# Author: KB 
# Date: 2026-03-06 
# Purpose: For doing the autoscaling for the simple Diffusion Model (2026.03.06) 
# File: autoscaler.py 

import time
import subprocess
import threading
import asyncio
import grpc
import query_pb2
import query_pb2_grpc

from collections import Counter

class AutoScaler:

    def __init__(self, scheduler, worker_pool, config):

        self.scheduler = scheduler
        self.worker_pool = worker_pool
        self.config = config

        self.min_workers = config.MIN_WORKERS
        self.max_workers = config.MAX_WORKERS

        self.scale_up_threshold = config.SCALE_UP_THRESHOLD
        self.scale_down_threshold = config.SCALE_DOWN_THRESHOLD

        self.check_interval = config.CHECK_INTERVAL

        self.running = False
        self.metrics = None

        ############################################################
        # PREDICTIVE AUTOSCALING STATE
        ############################################################

        self.queue_history = []
        self.max_history = 10
        self.prediction_horizon = 3
        self.startup_delay = 30      # seconds
        self.last_scale_time = 0
        self.scale_cooldown = 20

        ############################################################
        # Worker runtime cache
        ############################################################
        self.worker_status = {}

    ############################################################
    # WORKLOAD ANALYSIS
    ############################################################

    def update_queue_history(self, queue_length): # queue history

        self.queue_history.append(
            (
                time.time(),
                queue_length
            )
        )

        if len(self.queue_history) > self.max_history:
            self.queue_history.pop(0)
 
    def predict_queue(self): # predict future queue

        if len(self.queue_history) < 3:
            return 0

        growth = []

        for i in range(1, len(self.queue_history)):
            previous = self.queue_history[i-1][1]
            current = self.queue_history[i][1]
            growth.append(current - previous)

        average_growth = sum(growth) / len(growth)
        current_queue = self.queue_history[-1][1]

        predicted = current_queue + (average_growth * self.prediction_horizon)

        return max(predicted, 0)

  
    def can_scale(self): # scaling cooldoon
        return (time.time() - self.last_scale_time) > self.scale_cooldown

    def get_dominant_workload(self):

        workloads = []

        try:

            for q in self.scheduler.class_queues:

                items = list(q.queue)

                for req in items:

                    if hasattr(req, "workload_type"):
                        workloads.append(req.workload_type)

        except Exception as e:

            print(f"[AUTOSCALER WARNING] Workload inspection failed: {e}")

        if not workloads:
            return "balanced"

        dominant = Counter(workloads).most_common(1)[0][0]

        print(f"[AUTOSCALER] Dominant workload: {dominant}")

        return dominant        

    ############################################################
    # DOCKER CONTROL
    ############################################################

    def start_worker(self, workload_type="balanced"):
    
        print("[AUTOSCALER] Requesting worker_pool to add worker")
        scale_begin = time.perf_counter()

        before = self.worker_pool.size()

        # self.worker_pool.add_worker()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.worker_pool.add_worker(workload_type)
        )
        loop.close()

        time.sleep(5)  # give worker time to boot

        after = self.worker_pool.size()
        scale_latency = time.perf_counter() - scale_begin

        # ADDING FAILURE VISIBILITY
        if after == before:
            print("[AUTOSCALER ERROR] Worker failed to start!")
        else:
            print("[AUTOSCALER] Worker successfully added")
            self.last_scale_time = time.time() # record scaling time
            if self.metrics:
                self.metrics.log_autoscaler_event(
                    event="scale_up",
                    mode=self.config.SCALING_MODE,
                    queue_length=self.scheduler.get_total_queue_length(),
                    workers_before=before,
                    workers_after=after,
                    arrival_rate=self.scheduler.get_arrival_rate(),
                    service_rate=self.scheduler.get_service_rate(),
                    utilization=self.metrics.worker_utilization(
                        after,
                        self.scheduler.active_requests
                    ),
                    scale_latency=scale_latency
                )

    def stop_worker(self):
    
        print("[AUTOSCALER] Requesting worker_pool to remove worker")
        scale_begin = time.perf_counter()
        before = self.worker_pool.size()

        worker = self.choose_worker_to_remove()
        if worker is None:
            print("[AUTOSCALER] No removable worker found.")
            return

        if self.metrics: # record reason for removal    
            self.metrics.log_worker_removed(
                worker=worker["port"],
                workload=worker["workload_type"]
            )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.worker_pool.remove_worker(worker)
        )

        loop.close()
        
        after = self.worker_pool.size()
        self.last_scale_time = time.time() # recording scaling time
        scale_latency = time.perf_counter() - scale_begin

        if self.metrics:
            self.metrics.log_autoscaler_event(
                event="scale_down",
                mode=self.config.SCALING_MODE,
                queue_length=self.scheduler.get_total_queue_length(),
                workers_before=before,
                workers_after=after,
                arrival_rate=self.scheduler.get_arrival_rate(),
                service_rate=self.scheduler.get_service_rate(),
                utilization=self.metrics.worker_utilization(
                    max(after, 1),
                    self.scheduler.active_requests
                ),

                scale_latency=scale_latency

            )

            self.metrics.record_worker_count(
                self.worker_pool.size()
            )


    def get_worker_status(self, port):
        """
        Query a worker for runtime statistics.
        """

        try:
            with grpc.insecure_channel(f"localhost:{port}") as channel:
                stub = query_pb2_grpc.QueryStub(channel)

                response = stub.WorkerStatus(
                    query_pb2.WorkerStatusRequest(),
                    timeout=2
                )

                return response

        except Exception as e:
            print(f"Exception {e}")
            print(f"[AUTOSCALER WARNING] Worker {port} did not respond.")

            return None

    ############################################################
    # SCALING LOGIC
    ############################################################

    def scale(self):

        try:
            queue_length = self.scheduler.get_total_queue_length()
        except Exception as e:
            print(f"Exception with 'scheduler.get_total_queue_length()' in  autoscaler.py: {e}")
            queue_length = 0

        worker_count = self.worker_pool.size()

        # updating queue history every cycle
        self.update_queue_history(queue_length)
        predicted_queue = self.predict_queue()

        sla = self.scheduler.get_sla_violations()
        interactive_violations = sla["interactive"]
        balanced_violations = sla["balanced"]
        batch_violations = sla["batch"]
        total_sla_violations = interactive_violations + balanced_violations + batch_violations

        print(
            f"[AUTOSCALER] "
            f"Current={queue_length}, "
            f"Predicted={predicted_queue:.1f}"
        )

        ################################################################
        # Collect runtime status from all workers (Polling every worker)
        ################################################################
        
        self.worker_status.clear() # clear previously running cache

        worker_statuses = []

        for worker in self.worker_pool.get_workers():
            status = self.get_worker_status(worker["port"])
           
            if status is None:
                continue

            self.worker_status[worker["port"]] = status
            worker_statuses.append(status)

            print(
                f"[AUTOSCALER] Worker {worker['port']} | "
                f"Busy={status.busy} | "
                f"Active={status.active_requests} | "
                f"Latency={status.average_latency_ms:.2f} ms | "
                f"GPU={status.gpu_memory_mb:.2f} MB"
            )

        busy_workers = sum(1 for s in worker_statuses if s.busy)
        idle_workers = worker_count - busy_workers

        # -----------------------------------------
        # Determine dominant workload
        # -----------------------------------------

        dominant_workload = self.get_dominant_workload()

        if worker_count == 0:
            print("[AUTOSCALER WARNING] No workers available!")

        print(
            f"[AUTOSCALER] "
            f"Queue={queue_length}, "
            f"Predicted={predicted_queue:.1f}, "
            f"SLA={total_sla_violations}, "
            f"Workers={worker_count}, "
            f"Busy={busy_workers}, "
            f"Idle={idle_workers}"
        )

        if self.metrics:    
            self.metrics.log_cluster_state(
                workers=worker_count,
                busy_workers=busy_workers,
                idle_workers=idle_workers,
                queue_length=queue_length,
                predicted_queue=predicted_queue,
                arrival_rate=self.scheduler.get_arrival_rate(),
                service_rate=self.scheduler.get_service_rate(),
                mode=self.config.SCALING_MODE
            )   
       
        ############################################################
        # Utilization-based Horizontal Autoscaling
        ############################################################

        arrival_rate = self.scheduler.get_arrival_rate()
        service_rate = self.scheduler.get_service_rate()

        if service_rate <= 1e-6:
            rho = float("inf")
        else:
            rho = arrival_rate / (service_rate * max(worker_count, 1))

        print(
            f"[AUTOSCALER] "
            f"Workers={worker_count} | "
            f"λ={arrival_rate:.4f} | "
            f"μ={service_rate:.4f} | "
            f"ρ={rho:.2f}"
        )

        ############################################################
        # SCALE UP
        ############################################################

        if (
            rho > 0.80 and
            worker_count < self.max_workers and
            self.can_scale()
        ):

            print(
                f"[AUTOSCALER] SCALE UP "
                f"(ρ={rho:.2f}) "
                f"Launching {dominant_workload} worker"
            )

            self.start_worker(dominant_workload)

        ############################################################
        # SCALE DOWN
        ############################################################

        elif (
            rho < 0.30 and
            idle_workers > 0 and
            worker_count > self.min_workers and
            self.can_scale()
        ):

            print(
                f"[AUTOSCALER] SCALE DOWN "
                f"(ρ={rho:.2f})"
            )

            self.stop_worker()

    def scale_vertical(self):
        
        try:
            queue_length = self.scheduler.get_total_queue_length()
        except Exception as e:
            print(f'Exception: {e} in scale_vector() method')
            queue_length = 0

        print(
            f"[QUEUE DEBUG] "
            f"Active={self.scheduler.active_requests} "
            f"Queue={queue_length}"
        )

        # vertical scaling depends on the following
        arrival_rate = self.scheduler.get_arrival_rate()
        service_rate = self.scheduler.get_service_rate()

        print("=" * 60)
        print("[QUEUE DEBUG]")
        print(f"active reqs   : {self.scheduler.active_requests}")
        print(f"queue total   : {queue_length}")
        print(f"arrival rate  : {arrival_rate:.4f}")
        print(f"service rate  : {service_rate:.4f}")
        print("=" * 60)

        sla = self.scheduler.get_sla_violations()
        interactive_violations = sla["interactive"]
        balanced_violations = sla["balanced"]
        batch_violations = sla["batch"]
        total_sla_violations = interactive_violations + balanced_violations + batch_violations

        if self.metrics:
            
            ####################################################
            # Store metrics in memory
            ####################################################

            self.metrics.record_arrival_rate(arrival_rate)
            self.metrics.record_service_rate(service_rate)
            self.metrics.record_queue_length(queue_length)
            self.metrics.record_sla_violations(total_sla_violations)

            if hasattr(self.metrics, "record_queue_prediction"):
                self.metrics.record_queue_prediction(
                    self.predict_queue()
                )

            ####################################################
            # Write scheduler_metrics.csv
            ####################################################

            self.metrics.log_scheduler_metrics(
                queue_length=queue_length,
                arrival_rate=arrival_rate,
                service_rate=service_rate
            )

        print(
            f"[VERTICAL AUTOSCALER] Queue={queue_length}, "
            f"λ={arrival_rate:.2f}, μ={service_rate:.2f}"
        )

        # 🔥 Dynamic batching logic
        if queue_length > 10:
            self.scheduler.set_batch_size(8)
            print("[VERTICAL] Batch size → 8")

        elif queue_length > 5:
            self.scheduler.set_batch_size(4)
            print("[VERTICAL] Batch size → 4")

        else:
            self.scheduler.set_batch_size(1)
            print("[VERTICAL] Batch size → 1")

    def choose_worker_to_remove(self):
        """
        This is to chose a worker to remove
        """
        candidate = None
        best_score = float("inf")        

        for worker in self.worker_pool.get_workers():
            status = self.worker_status.get(worker["port"]) # instead of the previous "status = self.get_worker_status(worker["port"])" which will query gRPC again; unnecesssary as status has preciously been collected in the autoscaler cycle

            if status is None:
                continue
            
            # preserving workload diversity
            same_type = [
                    w for w in self.worker_pool.get_workers() 
                    if w["workload_type"] == worker["workload_type"]
                    ]

            if len(same_type) <= 1:
                continue

            score = self.compute_scale_down_score(
                worker,
                status
            )

            if score < best_score:
                best_score = score
                candidate = worker

        if candidate:
            print(
                f"[AUTOSCALER] Removing worker "
                f"{candidate['port']} "
                f"({candidate['workload_type']}) "
                f"score={best_score:.2f}"
            )

        return candidate

    def compute_scale_down_score(self, worker, status):
        """
        Lower score = remove first.
        """

        score = 0

        # Busy workers should stay alive
        score += status.active_requests * 100

        # High latency workers are probably doing work
        score += status.average_latency_ms

        # Large GPU usage usually means the model is active
        score += status.gpu_memory_mb * 0.05

        # Recently created workers should not immediately disappear
        age = self.worker_pool.get_worker_age(worker)

        if age < self.config.MIN_WORKER_AGE:
            score += 10000

        if self.metrics:
            self.metrics.log_scale_down_score(
                worker=worker["port"],
                score=score
            )

        return score

    ############################################################
    # MAIN LOOP
    ############################################################

    def run(self):
    
        # 🔥 Step 1: Start minimum workers (horizontal base)
        # Start workers ONLY for horizontal or hybrid scaling
        if self.config.SCALING_MODE in ["horizontal", "hybrid"]:

            for _ in range(self.min_workers):
                # self.start_worker()
                self.start_worker("balanced")

            print("[AUTOSCALER] Initial workers launched. Waiting for stabilization...")
            time.sleep(10)

        else:
            print("[AUTOSCALER] Vertical mode selected — skipping worker container startup")
        #time.sleep(10)

        # 🔥 Step 2: Continuous scaling loop
        while self.running:

            if self.metrics:
                # recording scaling mode every cycle    
                self.metrics.record_scaling_mode(
                    self.config.SCALING_MODE,
                    self.worker_pool.size()
                )

                # recording queueing statistics every cycle
                self.metrics.record_queue_length(self.scheduler.get_total_queue_length())
                if hasattr(self.metrics, "record_queue_prediction"):    
                    self.metrics.record_queue_prediction(self.predict_queue())

                self.metrics.record_worker_count(self.worker_pool.size())

                print(f"[AUTOSCALER] Monitoring {self.worker_pool.size()} workers")

                if self.metrics:
                    self.metrics.log_autoscaler_event(
                        event="monitor",
                        mode=self.config.SCALING_MODE,
                        queue_length=self.scheduler.get_total_queue_length(),
                        workers_before=self.worker_pool.size(),
                        workers_after=self.worker_pool.size(),
                        arrival_rate=self.scheduler.get_arrival_rate(),
                        service_rate=self.scheduler.get_service_rate(),
                        utilization=self.metrics.worker_utilization(
                            self.worker_pool.size(),
                            self.scheduler.active_requests
                        ),
                        scale_latency=0
                    )

                
            if self.config.SCALING_MODE == "vertical":
                self.scale_vertical()

            elif self.config.SCALING_MODE == "horizontal":
                self.scale()

            else:  # hybrid (recommended)
                self.scale_vertical()
                self.scale()

            time.sleep(self.check_interval)

    def start(self):

        self.running = True

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

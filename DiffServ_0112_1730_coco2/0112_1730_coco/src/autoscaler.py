# Author: KB 
# Date: 2026-03-06 
# Purpose: Unified INFaaS-style autoscaler — Strategy A/B cost comparison
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
        # INFaaS-style unified scaling state
        ############################################################
        self.per_model_qps = {}          # model_name -> QPS observed
        self.per_model_load_cost = {}    # model_name -> measured load time (s)
        self.scale_down_consecutive = 0  # consecutive idle cycles
        self.strategy_threshold = getattr(config, 'STRATEGY_THRESHOLD', 1.5)
        self.max_containers_per_model = getattr(config, 'MAX_CONTAINERS_PER_MODEL', 3)
        self.scale_down_delay = getattr(config, 'SCALE_DOWN_DELAY', 20)

    ############################################################
    # WORKLOAD ANALYSIS
    ############################################################

    def update_queue_history(self, queue_length):

        self.queue_history.append(
            (
                time.time(),
                queue_length
            )
        )

        if len(self.queue_history) > self.max_history:
            self.queue_history.pop(0)
 
    def predict_queue(self):

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

  
    def can_scale(self):
        return (time.time() - self.last_scale_time) > self.scale_cooldown

    def get_dominant_workload(self):

        workloads = []

        try:

            for q in self.scheduler.class_queues:

                items = list(q.queue)

                for req in items:

                    wt = self.scheduler._get_meta(req, "workload_type")
                    if wt is not None:
                        workloads.append(wt)

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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.worker_pool.add_worker(workload_type)
        )
        loop.close()

        time.sleep(5)  # give worker time to boot

        after = self.worker_pool.size()
        scale_latency = time.perf_counter() - scale_begin

        if after == before:
            print("[AUTOSCALER ERROR] Worker failed to start!")
        else:
            print("[AUTOSCALER] Worker successfully added")
            self.last_scale_time = time.time()
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

        if self.metrics:
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
        self.last_scale_time = time.time()
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
    # PER-MODEL QPS AND COST TRACKING
    ############################################################

    def collect_per_model_qps(self):
        """
        Poll all workers and aggregate per-model QPS and load costs.
        """
        model_worker_counts = {}  # model_name -> number of workers serving it
        model_total_qps = {}      # model_name -> aggregate QPS

        for worker in self.worker_pool.get_workers():
            status = self.worker_status.get(worker["port"])
            if status is None:
                continue

            model_name = status.model_name
            if model_name not in model_worker_counts:
                model_worker_counts[model_name] = 0
                model_total_qps[model_name] = 0.0
            model_worker_counts[model_name] += 1

            # Estimate per-worker QPS from throughput field
            model_total_qps[model_name] += getattr(status, 'throughput', 0.0)

        self.per_model_qps = model_total_qps

        # Count replicas per model across all workers
        self.worker_pool.model_worker_counts = model_worker_counts

        return model_worker_counts, model_total_qps

    def compute_model_cost(self, model_name, worker_count_for_model, qps):
        """
        INFaaS Strategy A: compute cost for a specific model variant.
        cost = load_time × worker_count_for_model + load_time / max(qps, 0.01)
        """
        load_cost = self.per_model_load_cost.get(
            model_name,
            getattr(self.config, 'MODEL_LOAD_COSTS', {}).get(model_name, 15.0)
        )

        cost = load_cost * worker_count_for_model + load_cost / max(qps, 0.01)
        return cost

    def compute_fastest_model_cost(self, model_worker_counts, model_qps):
        """
        INFaaS Strategy B: find the fastest (lowest load_time) model and compute
        cost of serving ALL demand through it.
        fastest_cost = fastest_load_time × total_count
        """
        total_workers = sum(model_worker_counts.values())
        total_qps = sum(model_qps.values())

        if total_workers == 0:
            return 0.0, None

        fastest_model = None
        fastest_load = float('inf')

        for model_name in model_worker_counts:
            load_cost = self.per_model_load_cost.get(
                model_name,
                getattr(self.config, 'MODEL_LOAD_COSTS', {}).get(model_name, 15.0)
            )
            if load_cost < fastest_load:
                fastest_load = load_cost
                fastest_model = model_name

        if fastest_model is None:
            return 0.0, None

        fastest_cost = fastest_load * total_workers
        return fastest_cost, fastest_model

    ############################################################
    # UNIFIED INFaaS SCALING LOGIC
    ############################################################

    def scale_infaas(self):
        """
        Unified INFaaS-style scaling algorithm combining horizontal
        (container-level) and vertical (model-switching) scaling.

        Per INFaaS autoscaler.cc:
        - Strategy A: sum_cost = Σ(load_lat × count) per variant
        - Strategy B: fastest_cost = fastest_loadlat × fastest_count
        - Decision: if sum_cost > 1.5 × fastest_cost → use fastest variant
        """

        try:
            queue_length = self.scheduler.get_total_queue_length()
        except Exception as e:
            print(f"[INFaaS] Queue length error: {e}")
            queue_length = 0

        worker_count = self.worker_pool.size()

        # Update queue history for prediction
        self.update_queue_history(queue_length)
        predicted_queue = self.predict_queue()

        # Collect SLA violations
        sla = self.scheduler.get_sla_violations()
        total_sla_violations = sum(sla.values())

        # Collect worker statuses
        self.worker_status.clear()
        worker_statuses = []

        for worker in self.worker_pool.get_workers():
            status = self.get_worker_status(worker["port"])
            if status is None:
                continue
            self.worker_status[worker["port"]] = status
            worker_statuses.append(status)

        busy_workers = sum(1 for s in worker_statuses if s.busy)
        idle_workers = worker_count - busy_workers

        # Collect per-model QPS from workers
        model_worker_counts, model_qps = self.collect_per_model_qps()

        # Update per-model load costs from worker measurements
        for worker in self.worker_pool.get_workers():
            status = self.worker_status.get(worker["port"])
            if status is not None:
                model_name = status.model_name
                switch_time_ms = status.last_switch_time_ms
                if switch_time_ms > 0:
                    self.per_model_load_cost[model_name] = switch_time_ms / 1000.0

        arrival_rate = self.scheduler.get_arrival_rate()
        service_rate = self.scheduler.get_service_rate()

        if service_rate <= 1e-6:
            rho = float("inf")
        else:
            rho = arrival_rate / (service_rate * max(worker_count, 1))

        dominant_workload = self.get_dominant_workload()

        print("=" * 70)
        print("[INFaaS UNIFIED SCALER]")
        print(f"  Workers={worker_count} | Busy={busy_workers} | Idle={idle_workers}")
        print(f"  Queue={queue_length} | Predicted={predicted_queue:.1f}")
        print(f"  ρ={rho:.2f} | λ={arrival_rate:.4f} | μ={service_rate:.4f}")
        print(f"  SLA violations={total_sla_violations}")
        print(f"  Per-model QPS: {model_qps}")
        print(f"  Per-model workers: {model_worker_counts}")
        print(f"  Per-model load costs: {self.per_model_load_cost}")
        print("=" * 70)

        if self.metrics:
            self.metrics.log_cluster_state(
                workers=worker_count,
                busy_workers=busy_workers,
                idle_workers=idle_workers,
                queue_length=queue_length,
                predicted_queue=predicted_queue,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                mode=self.config.SCALING_MODE
            )

        ############################################################
        # SLO-pressure signal: rolling SLO violation rate vs budget
        ############################################################

        try:
            slo_violation_rate = self.scheduler.get_slo_violation_rate()
        except Exception:
            slo_violation_rate = 0.0

        slo_budget = getattr(self.config, "SLO_VIOLATION_BUDGET", 0.01)
        slo_breached = slo_violation_rate > slo_budget

        print(f"[INFaaS] SLO violation rate={slo_violation_rate:.4f} "
              f"(budget={slo_budget:.4f}, breached={slo_breached})")

        ############################################################
        # INFaaS STRATEGY A vs B COMPARISON
        ############################################################

        if len(model_worker_counts) > 0:
            # Strategy A: sum of per-variant costs
            sum_cost = 0.0
            for model_name, count in model_worker_counts.items():
                qps = model_qps.get(model_name, 0.0)
                variant_cost = self.compute_model_cost(model_name, count, qps)
                sum_cost += variant_cost
                print(f"[INFaaS] Strategy A: {model_name} cost={variant_cost:.4f} "
                      f"(count={count}, qps={qps:.2f})")

            # Strategy B: fastest variant cost
            fastest_cost, fastest_model = self.compute_fastest_model_cost(
                model_worker_counts, model_qps
            )

            print(f"[INFaaS] Strategy A total={sum_cost:.4f}")
            print(f"[INFaaS] Strategy B fastest={fastest_model} cost={fastest_cost:.4f}")
            print(f"[INFaaS] Threshold: {self.strategy_threshold}x")

            if fastest_cost > 0 and sum_cost > self.strategy_threshold * fastest_cost:
                print(f"[INFaaS] DECISION: Switch to fastest model {fastest_model} "
                      f"(sum={sum_cost:.4f} > {self.strategy_threshold}×{fastest_cost:.4f})")
                self._trigger_vertical_switch(fastest_model)
            else:
                print(f"[INFaaS] DECISION: Keep current model mix "
                      f"(sum={sum_cost:.4f} <= {self.strategy_threshold}×{fastest_cost:.4f})")

        ############################################################
        # HORIZONTAL SCALING (container-level)
        ############################################################

        if (
            (rho > 0.80 or slo_breached) and
            worker_count < self.max_workers and
            self.can_scale()
        ):
            print(f"[INFaaS] SCALE UP (ρ={rho:.2f}, SLO breach={slo_breached}) "
                  f"Launching {dominant_workload} worker")
            self.start_worker(dominant_workload)
            self.scale_down_consecutive = 0

        elif (
            rho < 0.30 and
            idle_workers > 0 and
            worker_count > self.min_workers and
            self.can_scale()
        ):
            self.scale_down_consecutive += 1

            if self.scale_down_consecutive >= self.scale_down_delay:
                print(f"[INFaaS] SCALE DOWN (ρ={rho:.2f}) consecutive={self.scale_down_consecutive}")
                self.stop_worker()
                self.scale_down_consecutive = 0
            else:
                print(f"[INFaaS] Idle detected ({self.scale_down_consecutive}/{self.scale_down_delay})")
        else:
            self.scale_down_consecutive = 0

        ############################################################
        # VERTICAL SCALING (batch-size tuning)
        ############################################################

        if queue_length > 10:
            self.scheduler.set_batch_size(8)
            print("[INFaaS] Batch size → 8")
        elif queue_length > 5:
            self.scheduler.set_batch_size(4)
            print("[INFaaS] Batch size → 4")
        else:
            self.scheduler.set_batch_size(1)
            print("[INFaaS] Batch size → 1")

    def _trigger_vertical_switch(self, target_model):
        """
        Trigger a vertical model switch to target_model on the least-busy worker.
        """
        best_worker = None
        best_score = float('inf')

        for worker in self.worker_pool.get_workers():
            status = self.worker_status.get(worker["port"])
            if status is None:
                continue
            score = status.active_requests * 1000 + status.average_latency_ms
            if score < best_score:
                best_score = score
                best_worker = worker

        if best_worker is None:
            print("[INFaaS] No worker available for vertical switch")
            return

        print(f"[INFaaS] Vertical switch → {target_model} on worker {best_worker['port']}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.worker_pool.switch_model(
                f"localhost:{best_worker['port']}",
                self.worker_pool.get_model_id_from_name(target_model)
            )
        )
        loop.close()

    ############################################################
    # SCALE-DOWN SELECTION
    ############################################################

    def choose_worker_to_remove(self):
        """
        Choose a worker to remove. Preserves workload diversity and
        penalizes workers that serve popular models.
        """
        candidate = None
        best_score = float("inf")        

        for worker in self.worker_pool.get_workers():
            status = self.worker_status.get(worker["port"])

            if status is None:
                continue
            
            # preserving workload diversity
            same_type = [
                    w for w in self.worker_pool.get_workers() 
                    if w["workload_type"] == worker["workload_type"]
                    ]

            if len(same_type) <= 1:
                continue

            score = self.compute_scale_down_score(worker, status)

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

        score += status.active_requests * 100
        score += status.average_latency_ms
        score += status.gpu_memory_mb * 0.05

        # Penalize removal of workers serving popular models
        model_name = status.model_name
        model_qps = self.per_model_qps.get(model_name, 0.0)
        score += model_qps * 50

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
    
        if self.config.SCALING_MODE in ["horizontal", "hybrid"]:

            for _ in range(self.min_workers):
                self.start_worker("balanced")

            print("[AUTOSCALER] Initial workers launched. Waiting for stabilization...")
            time.sleep(10)

        else:
            print("[AUTOSCALER] Vertical mode selected — skipping worker container startup")

        while self.running:

            if self.metrics:
                self.metrics.record_scaling_mode(
                    self.config.SCALING_MODE,
                    self.worker_pool.size()
                )

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

            # Unified INFaaS scaling (replaces separate scale() + scale_vertical())
            self.scale_infaas()

            time.sleep(self.check_interval)

    def start(self):

        self.running = True

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

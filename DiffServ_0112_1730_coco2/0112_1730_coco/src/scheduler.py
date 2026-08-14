# Author: KB 
# Date: 2026-03-06 
# Purpose: Expose queue metrics 
# File: diffserv_scheduler.py 

import queue
import time
import threading
import uuid
from workload_classifier import WorkloadClassifier

import asyncio
import grpc

import query_pb2
import query_pb2_grpc


class Scheduler:

    def __init__(self, worker_pool, config, metrics=None):    
        
        self.batch_size = 4
        self.config = config
        self.metrics = metrics
        self.worker_pool = worker_pool
        self.worker_runtime = {} # Worker runtime cache

        self.sla_violations = { # for recording Service Level Agreement (SLA) violations
            "interactive": 0,
            "balanced": 0,
            "batch": 0
        }

        self.class_queues = [
            queue.Queue(),
            queue.Queue(),
            queue.Queue()
        ] # for scheduling / priority

        self.request_queue = queue.Queue() # for batching / throughput

        self.arrival_count = 0
        self.service_count = 0

        #
        # Beginning of the experiment
        #
        #self.arrival_window_start = time.time()
        #self.service_window_start = time.time()
        self.arrival_start_time = time.time()
        self.service_start_time = time.time()


        self.lock = threading.Lock()

        self.active_requests = 0
        self.classifier = WorkloadClassifier()

  ############################################################
    # ACTIVE REQUEST TRACKING (FOR AUTOSCALER)
  ############################################################        
        
    def increment_active(self):
        with self.lock:
            self.active_requests += 1
            print(f"[SCHEDULER] increment_active -> {self.active_requests}")

    def decrement_active(self):
        with self.lock:
            if self.active_requests > 0:
                self.active_requests -= 1
            print(f"[SCHEDULER] decrement_active -> {self.active_requests}")

    ############################################################
    # CORE METHODS (UNCHANGED)
    ############################################################ 
    
    # A dispatching coroutine added; function for scheduling
    async def dispatch_request(self, request):
        workload = getattr(request, "workload_type", "balanced")
        worker_addr = self.worker_pool.get_next_worker(workload)
        if worker_addr is None:    
            raise RuntimeError("No available worker.")
        print(f"[SCHEDULER] Dispatching to {worker_addr}")

        async with grpc.aio.insecure_channel(worker_addr) as channel:
            stub = query_pb2_grpc.QueryStub(channel)
            ############################################################
            # Health check
            ############################################################

            heartbeat = await stub.Heartbeat(
                query_pb2.HeartbeatRequest()
            )

            if heartbeat.status.status != 1:
                raise RuntimeError(
                    f"Worker {worker_addr} unhealthy."
                )

            ############################################################
            # Retry loop
            ############################################################

            for attempt in range(3):

                try:

                    response_stream = stub.QueryOnlineImage(request)

                    ####################################################
                    # Forward stream
                    ####################################################

                    async for response in response_stream:
                        yield response

                    ####################################################
                    # Runtime update AFTER request finishes
                    ####################################################

                    status = await stub.WorkerStatus(
                        query_pb2.WorkerStatusRequest()
                    )

                    self.worker_runtime[worker_addr] = {

                        "active_requests": status.active_requests,

                        "latency_ms": status.average_latency_ms,

                        "gpu_memory": status.gpu_memory_mb,

                        "busy": status.busy,

                        "queue_length": status.queue_length,

                        "timestamp": time.time()

                    }

                    return

                except grpc.RpcError:

                    print(f"[Scheduler] Retry {attempt+1}")

                    await asyncio.sleep(2)

            raise RuntimeError(
                f"Worker {worker_addr} unavailable."
            )
                # # Double exceptioning to prevent the scheduler from crashing   
                # except grpc.RpcError as e:    
                #     print(f"[Scheduler] Worker failed: {e}\n Worker Address: {worker_addr}")
                #     return None
                # except Exception as e:
                #     print(f"[Scheduler] Unexpected worker error: {e}")
                #     return None

    async def dispatch_batch(self, batch):
        
        worker_addr = self.worker_pool.get_next_worker(batch[0].workload_type)

        print(f"[SCHEDULER] Dispatching batch of {len(batch)} requests to {worker_addr}")

        prompts = []

        for req in batch:
            prompts.extend(req.Prompt)

        request = query_pb2.QueryOnlineRequest(
            Prompt=prompts,
            Steps=batch[0].Steps,
            CFG_Scale=batch[0].CFG_Scale,
            BatchSize=len(prompts),
            Seed=batch[0].Seed
        )

        dispatch_time = time.perf_counter()
        for req in batch:  # every request knows when it has left the scheduler
            req.dispatch_time = dispatch_time

        async with grpc.aio.insecure_channel(worker_addr) as channel:
            stub = query_pb2_grpc.QueryStub(channel)
            response = await stub.QueryOnlineImage(request)

            try: #update worker runtime cache
                status = await stub.WorkerStatus(
                    query_pb2.WorkerStatusRequest()
                )

                if self.metrics: # logging runtime statistics
                    self.metrics.log_worker_runtime(
                        worker=worker_addr,
                        active=status.active_requests,
                        latency=status.average_latency_ms,
                        gpu=status.gpu_memory_mb,
                        busy=status.busy
                    )

                self.worker_runtime[worker_addr] = {
                    "active_requests": status.active_requests,
                    "latency_ms": status.average_latency_ms,
                    "gpu_memory": status.gpu_memory_mb,
                    "busy": status.busy,
                    "timestamp": time.time()
                }
            except Exception:

                pass
        return response

    def handle_request(self, request):
    
        self.increment_active()
        try:
            print("[SCHEDULER] Processing request")

            batch = self.get_batch()
            if len(batch) == 0:
                return None

            response = asyncio.run(self.dispatch_batch(batch))

            finish_time = time.perf_counter()

            for req in batch:  # allow for later computations
                req.finish_time = finish_time
                total_latency = finish_time - req.arrival_time

                sla_target = self.config.SLA_TARGETS.get(req.workload_type, 2000)
                latency_ms = total_latency * 1000

                if latency_ms > sla_target:
                    self.sla_violations[req.workload_type] += 1

                if self.metrics:
                    self.metrics.log_request_latency(
                        request_id=req.request_id,
                        latency=total_latency,
                        workload=req.workload_type
                    )

            if response is None:    
                print("[SCHEDULER] Request failed")
            else:
                print("[SCHEDULER] Worker finished")
            return response
        finally:
            self.decrement_active()


    def start(self):

        print("[SCHEDULER] Scheduler started")

        while True:
            request = self.dequeue()

            if request is not None:
                response = self.handle_request(request) # later we will forward the request to the client
                print("[SCHEDULER] Response received")

            else:
                time.sleep(0.01)

    ############################################################
    # 🔥 UPDATED METRIC (KEY CHANGE)
    ############################################################                
    # def get_total_queue_length(self):
        
    #     if self.config.SCALING_MODE == "vertical":    
    #         #
    #         # In vertical mode the queue consists of:
    #         #   queued requests
    #         #   requests currently executing
    #         #
    #         queue_total = self.request_queue.qsize() + self.active_requests
    #     else:
    #         queue_total = sum(q.qsize() for q in self.class_queues)

    #     if self.config.SCALING_MODE != "vertical":
    #         with self.lock:
    #             queue_total += self.active_requests

    #     return queue_total
    
    def get_total_queue_length(self):
        """
        Current scheduler load.

        Because requests are dispatched immediately in the current
        architecture, qsize() is no longer meaningful.

        Use active requests instead.
        """

        return self.active_requests

    
    ############################################################
    # EXISTING METRICS (UNCHANGED)
    ############################################################
    def enqueue(self, request, cls=2):
        '''
        Goes into both queues
        '''

        print(f"[SCHEDULER DEBUG] enqueue class={cls}")
        
        # if self.batch_size > 1: # because vertical batching uses 'self.request_queue' instead of self.class_queues; (2026.06.19)
        #     self.class_queues[cls].put(request)
        # # -----------------------------------------
        # # Workload classification
        # # -----------------------------------------

        print("[DEBUG] enqueue entered")

        
        # Request tracing; timestamping every arriving request
        request.request_id = str(uuid.uuid4())
        request.arrival_time = time.perf_counter()
        request.dispatch_time = None
        request.queue_wait = None

        # self.class_queues[cls].put(request)

        try:
            workload_type = self.classifier.classify(request)
        except Exception as e:
            print(f"[CLASSIFIER ERROR] {e}")
            workload_type = "balanced"
        request.workload_type = workload_type

        if workload_type == "interactive":
            cls = 0
        elif workload_type == "balanced":
            cls = 1
        else:
            cls = 2

        # attach dynamic metadata
        print(f"[SCHEDULER] Classified request as: {workload_type}")
        
        if self.config.SCALING_MODE == "vertical":    
            self.request_queue.put(request)
        else:
            self.class_queues[cls].put(request)

        print("[DEBUG] request added to queues")

        with self.lock:
                self.arrival_count += 1
                print(f"[SCHEDULER] arrival_count={self.arrival_count}"
)

        if self.metrics:

            self.metrics.record_queue_length(

                self.get_total_queue_length()

            )


    def dequeue(self):

        # Vertical scaling
        if self.config.SCALING_MODE == "vertical":
            try:
                req = self.request_queue.get_nowait()

            except queue.Empty:
                return None

            req.dispatch_time = time.perf_counter()
            req.queue_wait = (req.dispatch_time - req.arrival_time)

            with self.lock:
                self.service_count += 1
            return req

        # Horizontal scheduling
        for q in self.class_queues:
            if not q.empty():

                req = q.get()
                req.dispatch_time = time.perf_counter()

                req.queue_wait = (req.dispatch_time - req.arrival_time)

                with self.lock:
                    self.service_count += 1
                    print(f"[SCHEDULER] service_count={self.service_count}"
)

                return req
        return None

    def get_arrival_rate(self):
    
        with self.lock:
            elapsed = time.time() - self.arrival_start_time

            if elapsed <= 0:
                return 0

            rate = self.arrival_count / elapsed

        if self.metrics:
            self.metrics.record_arrival_rate(rate)

        return rate

    def get_service_rate(self):
        
        with self.lock:
            elapsed = time.time() - self.service_start_time

            if elapsed <= 0:
                return 0

            rate = self.service_count / elapsed

        if self.metrics:
            self.metrics.record_service_rate(rate)

        return rate

    def get_sla_violations(self):
    
        return dict(self.sla_violations)

    def compute_worker_score(self, worker_addr): # worker performance score
    
        if worker_addr not in self.worker_runtime:
            return 0

        stats = self.worker_runtime[worker_addr]

        score = (stats["active_requests"] * 100 + stats["latency_ms"] + stats["gpu_memory"] * 0.01)

        if stats["busy"]:
            score += 1000

        return score

    def set_batch_size(self, size): # PNB(2026.04.23)
        self.batch_size = size

    def get_batch(self): # Core of vertical scaling. PNB(2026.04.23)
        batch = []

        deadline = time.time() + 0.05

        while len(batch) < self.batch_size:
            remaining = deadline - time.time()

            if remaining <= 0:
                break
            try:
                req = self.request_queue.get(timeout=remaining)
                batch.append(req)
            except Exception as e:
                print(f"Exception: {e}")
                break
        
        if self.metrics:
    
            self.metrics.record_batch_size(

                len(batch)

            )

        return batch

    def get_batch_size(self):
        return self.batch_size

    def get_cluster_statistics(self, worker_pool, autoscaler):

        sla = self.get_sla_violations()

        return {

            "worker_count": worker_pool.size(),
            "queue_length": self.get_total_queue_length(),
            "predicted_queue": autoscaler.predict_queue(),
            "arrival_rate": self.get_arrival_rate(),
            "service_rate":self.get_service_rate(),
            "active_requests": self.active_requests,
            "interactive": sla["interactive"],
            "balanced": sla["balanced"],
            "batch": sla["batch"],
            "total": ( sla["interactive"] + sla["balanced"]  + sla["batch"])
        }

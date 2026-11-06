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
import infaas_request_status_pb2

# SLO module for auto-configured latency targets
try:
    from slo import (
        resolve_latency_slo_ms,
        get_accuracy_level_from_steps,
        resolve_exec_time_ms,
    )
    import slo as slo_module
except ImportError:
    print("WARNING: SLO module not available - using default SLA fallbacks")
    def resolve_latency_slo_ms(model_id, steps):
        return 2000.0  # default fallback
    def get_accuracy_level_from_steps(steps):
        return "medium"
    def resolve_exec_time_ms(model_id, steps):
        return 10000.0  # default fallback
    slo_module = None

# PaniniServe merge seams (DIFS v3; config-gated, default OFF). See
# panini_seams.py for the ported order_batch / GPUUtilMonitor logic.
try:
    from panini_seams import GPUUtilMonitor, fair_share_step_reduction
except ImportError:
    print("WARNING: panini_seams module not available - seams disabled")
    class GPUUtilMonitor:
        def __init__(self, gpu_index=0):
            self.available = False
        def start(self):
            pass
        def stop(self):
            return 0
    def fair_share_step_reduction(steps, min_steps, overflow_units, time_per_step):
        return max(int(steps) - int(overflow_units * time_per_step), int(min_steps))


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

        # SLO tracking: keyed by (model_id, accuracy_level) -> [served, met]
        self.slo_stats = {}  # type: dict[tuple, list]

        ############################################################
        # PaniniServe merge seams (DIFS v3). All default OFF: with the
        # config flags left False, scheduling behavior is identical to v2.
        ############################################################
        self.panini_admission = bool(getattr(self.config, "PANINI_ADMISSION_CONTROL", False))
        self.panini_locality = bool(getattr(self.config, "PANINI_LOCALITY", False))
        self.panini_gpu_monitor = bool(getattr(self.config, "PANINI_GPU_MONITOR", False))
        if slo_module is not None:
            slo_module.PANINI_SLO_MULTIPLIER = bool(
                getattr(self.config, "PANINI_SLO_INTERPOLATED", False)
            )

        self.class_queues = [
            queue.Queue(),
            queue.Queue(),
            queue.Queue()
        ] # for scheduling / priority

        self.request_queue = queue.Queue() # for batching / throughput

        # Non-proto request bookkeeping (SLO deadline, workload, timestamps).
        # Proto messages reject arbitrary attribute assignment, so per-request
        # metadata lives in a side dict keyed by id(request).
        self._request_meta = {}

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
    def _set_meta(self, request, **kwargs):
        """Attach non-proto metadata to a request (SLO deadline, workload,
        timestamps). Proto messages reject arbitrary attributes, so metadata
        is stored in a side dict keyed by id(request); plain objects get
        real attributes set."""
        if hasattr(type(request), "DESCRIPTOR"):
            self._request_meta.setdefault(id(request), {}).update(kwargs)
        else:
            for k, v in kwargs.items():
                setattr(request, k, v)

    def _get_meta(self, request, key, default=None):
        """Read metadata attached by _set_meta()."""
        if hasattr(type(request), "DESCRIPTOR"):
            return self._request_meta.get(id(request), {}).get(key, default)
        return getattr(request, key, default)

    def _clear_meta(self, request):
        if hasattr(type(request), "DESCRIPTOR"):
            self._request_meta.pop(id(request), None)

    async def dispatch_request(self, request):
        req_start = time.perf_counter()
        model_id, accuracy, steps, slo_ms = self._slo_context(request)

        # Seam 1 (PaniniServe merge, default OFF): admission control.
        # Under overload, fair-share step reduction + deadline drops.
        if self.panini_admission:
            steps, slo_ms, dropped = self._panini_admission(model_id, accuracy, steps, slo_ms)
            if dropped:
                print("[SCHEDULER][SEAM1] Request dropped under overload")
                yield query_pb2.QueryOnlineImageResponse(
                    status=infaas_request_status_pb2.InfaasRequestStatus(
                        status=infaas_request_status_pb2.UNAVAILABLE,
                        msg="PaniniServe seam 1: dropped under overload"
                    )
                )
                self.decrement_active()
                return
            if int(steps) != int(self._slo_context(request)[2]):
                # Propagate the fair-share step reduction to the worker.
                try:
                    request.Steps = steps
                except Exception:
                    pass

        workload = self._get_meta(request, "workload_type", "balanced")
        if self.panini_locality:
            # Seam 3 (default OFF): prefer a worker already running this model.
            worker_addr = self.worker_pool.get_next_worker(workload, model_id=int(model_id))
        else:
            worker_addr = self.worker_pool.get_next_worker(workload)

        if worker_addr is None:
            print("[SCHEDULER] No workers available, returning UNAVAILABLE")
            yield query_pb2.QueryOnlineImageResponse(
                status=infaas_request_status_pb2.InfaasRequestStatus(
                    status=infaas_request_status_pb2.UNAVAILABLE,
                    msg="No workers available"
                )
            )
            if self.panini_admission:
                self.decrement_active()
            return

        actual_model_id = self.worker_pool.get_worker_model_id(worker_addr)
        if actual_model_id != model_id:
            model_id = actual_model_id
            try:
                slo_ms = resolve_latency_slo_ms(model_id, steps)
            except Exception:
                pass

        print(f"[SCHEDULER] Dispatching to {worker_addr}")

        # Seam 2 (default OFF): sample GPU utilization over this request's
        # execution window (10 ms polling) and log the average.
        gpu_monitor = None
        if self.panini_gpu_monitor:
            gpu_monitor = GPUUtilMonitor()
            gpu_monitor.start()

        try:
            async for response in self._dispatch_to_worker(
                request, worker_addr, model_id, accuracy, steps, slo_ms, req_start
            ):
                yield response
        finally:
            # The streaming path otherwise never decrements active_requests
            # (a v1/v2 gap).  Seam 1 depends on an accurate concurrency count,
            # so when it is enabled the counter is closed out here.
            if self.panini_admission:
                self.decrement_active()
            if gpu_monitor is not None:
                self._record_gpu_util(gpu_monitor)

    async def _dispatch_to_worker(self, request, worker_addr, model_id, accuracy, steps, slo_ms, req_start):
        """Forward one request to a worker (RPC + retry loop). Kept separate so
        dispatch_request can wrap it in seam try/finally bookkeeping."""
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

                        "timestamp": time.time()

                    }

                    # SLO evaluation: end-to-end latency vs auto-configured SLO
                    total_latency = time.perf_counter() - req_start
                    self._record_slo(model_id, accuracy, steps, slo_ms, total_latency)
                    self._clear_meta(request)

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
        
        worker_addr = self.worker_pool.get_next_worker(self._get_meta(batch[0], "workload_type"))

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
            self._set_meta(req, dispatch_time=dispatch_time)

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
                self._set_meta(req, finish_time=finish_time)
                total_latency = finish_time - self._get_meta(req, "arrival_time", finish_time)

                # SLO evaluation: auto-configured latency SLO per (model, steps).
                # Falls back to legacy SLA_TARGETS if the SLO module is absent.
                try:
                    model_id, accuracy, steps, slo_ms = self._slo_context(req)
                    self._record_slo(model_id, accuracy, steps, slo_ms, total_latency)
                    latency_ms = total_latency * 1000
                    if latency_ms > slo_ms:
                        self.sla_violations[self._get_meta(req, "workload_type", "balanced")] += 1
                except Exception:
                    sla_target = self.config.SLA_TARGETS.get(self._get_meta(req, "workload_type", "balanced"), 2000)
                    latency_ms = total_latency * 1000
                    if latency_ms > sla_target:
                        self.sla_violations[self._get_meta(req, "workload_type", "balanced")] += 1

                if self.metrics:
                    self.metrics.log_request_latency(
                        request_id=self._get_meta(req, "request_id"),
                        latency=total_latency,
                        workload=self._get_meta(req, "workload_type", "balanced")
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
        self._set_meta(request,
            request_id=str(uuid.uuid4()),
            arrival_time=time.perf_counter(),
            dispatch_time=None,
            queue_wait=None)

        # SLO deadline: derived latency SLO converted to a perf_counter deadline
        _arrival = self._get_meta(request, "arrival_time", time.perf_counter())
        try:
            _mid, _acc, _steps, _slo_ms = self._slo_context(request)
            self._set_meta(request, latency_slo_ms=_slo_ms,
                           deadline=_arrival + (_slo_ms / 1000.0))
        except Exception:
            self._set_meta(request, latency_slo_ms=2000.0,
                           deadline=_arrival + 2.0)

        # self.class_queues[cls].put(request)

        try:
            workload_type = self.classifier.classify(request)
        except Exception as e:
            print(f"[CLASSIFIER ERROR] {e}")
            workload_type = "balanced"
        self._set_meta(request, workload_type=workload_type)

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

            _dt = time.perf_counter()
            self._set_meta(req, dispatch_time=_dt)
            self._set_meta(req, queue_wait=_dt - self._get_meta(req, "arrival_time", _dt))

            with self.lock:
                self.service_count += 1
            return req

        # Horizontal scheduling with earliest-deadline-first (EDF) within
        # each priority class: preserve class priority (interactive first),
        # but within a class serve the request closest to its SLO deadline.
        for q in self.class_queues:
            if q.empty():
                continue

            # Drain the queue picking the earliest-deadline request
            candidates = []
            while not q.empty():
                candidates.append(q.get())

            if not candidates:
                return None

            def _deadline(r):
                return getattr(r, "deadline", float("inf"))

            req = min(candidates, key=_deadline)
            for r in candidates:
                if r is not req:
                    q.put(r)

            _dt = time.perf_counter()
            self._set_meta(req, dispatch_time=_dt)
            self._set_meta(req, queue_wait=_dt - self._get_meta(req, "arrival_time", _dt))

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

        # Order batch by SLO deadline (EDF) so urgent requests are served first
        batch.sort(key=lambda r: self._get_meta(r, "deadline", float("inf")))

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

    ############################################################
    # SLO SUPPORT (auto-configured latency SLO)
    ############################################################

    def _slo_context(self, request):
        """
        Derive the per-request SLO context:
          (model_id, accuracy_level, steps, latency_slo_ms)

        Latency SLO is NOT user-supplied: it is auto-configured as
        SLO_LATENCY_MULTIPLIER * profiled_exec_time(model, steps).
        """
        steps = int(getattr(request, "Steps", 0) or 0)
        if steps <= 0:
            steps = 30

        try:
            accuracy = str(getattr(request, "AccuracyLevel", "") or "")
            if not accuracy:
                accuracy = get_accuracy_level_from_steps(steps)
        except Exception:
            accuracy = "medium"

        model_id = getattr(request, "model_id", None)
        if model_id is None:
            model_id = getattr(self.config, "MODEL_ID", 3)

        try:
            slo_ms = resolve_latency_slo_ms(int(model_id), int(steps))
        except Exception:
            slo_ms = 2000.0

        return int(model_id), str(accuracy), int(steps), float(slo_ms)

    def _panini_admission(self, model_id, accuracy, steps, slo_ms):
        """
        Seam 1 (PaniniServe merge, default OFF) - admission control.

        Under overload (active_requests > PANINI_OVERLOAD_THRESHOLD), shrink the
        request's denoising steps fair-share style (never below its accuracy
        tier floor).  Then a queueing-aware deadline check predicts whether the
        reduced execution still fits inside the latency SLO (estimated wait for
        in-flight requests + profiled exec time); if not, the request is
        dropped.  Requests are also dropped outright once active_requests
        reaches PANINI_DROP_THRESHOLD.
        """
        active = self.active_requests
        threshold = int(getattr(self.config, "PANINI_OVERLOAD_THRESHOLD", 4))
        drop_threshold = int(getattr(self.config, "PANINI_DROP_THRESHOLD", threshold * 2))
        cut_per_unit = float(getattr(self.config, "PANINI_STEP_CUT_PER_OVERLOAD", 2))

        try:
            acc_ranges = getattr(self.config, "ACCURACY_STEP_RANGES", {})
            min_steps = int(acc_ranges.get(accuracy, (10, 20))[0])
        except Exception:
            min_steps = 10
        default_steps = int(steps)

        if active >= drop_threshold:
            print(f"[SCHEDULER][SEAM1] drop (active={active} >= drop_threshold={drop_threshold})")
            return steps, slo_ms, True

        reduced_steps = int(steps)
        if active > threshold:
            reduced_steps = fair_share_step_reduction(
                steps, min_steps, active - threshold, cut_per_unit
            )
            try:
                slo_ms = resolve_latency_slo_ms(int(model_id), int(reduced_steps))
            except Exception:
                slo_ms = 2000.0
            print(f"[SCHEDULER][SEAM1] step reduction {default_steps} -> {reduced_steps} "
                  f"(active={active} > threshold={threshold})")

        # Queueing-aware deadline check: predicted end-to-end latency
        # (estimated serialized wait for in-flight requests + this request's
        # profiled execution) must fit the latency SLO.
        try:
            exec_ms = resolve_exec_time_ms(int(model_id), int(reduced_steps))
        except Exception:
            exec_ms = slo_ms / float(getattr(self.config, "SLO_LATENCY_MULTIPLIER", 2.5))
        predicted_latency_ms = active * exec_ms
        if predicted_latency_ms > slo_ms:
            print(f"[SCHEDULER][SEAM1] drop (predicted={predicted_latency_ms:.0f}ms "
                  f"> slo={slo_ms:.0f}ms)")
            return reduced_steps, slo_ms, True

        return reduced_steps, slo_ms, False

    def _record_gpu_util(self, monitor):
        """Seam 2 (default OFF): stop the per-request GPU monitor and log the
        average utilization over the request's execution window."""
        try:
            util = monitor.stop()
        except Exception:
            util = 0
        if self.metrics:
            try:
                self.metrics.log_gpu_metrics(util)
            except Exception:
                pass

    def _record_slo(self, model_id, accuracy_level, steps, slo_ms, latency_sec):
        """Record a single SLO evaluation. Returns True if the SLO was met."""
        met = latency_sec * 1000 <= slo_ms

        key = (int(model_id), str(accuracy_level))
        if key not in self.slo_stats:
            self.slo_stats[key] = [0, 0]  # [served, met]
        self.slo_stats[key][0] += 1
        if met:
            self.slo_stats[key][1] += 1

        if self.metrics:
            try:
                self.metrics.record_slo(
                    model_id=model_id,
                    accuracy_level=accuracy_level,
                    steps=steps,
                    slo_ms=slo_ms,
                    latency_sec=latency_sec,
                    met=met
                )
            except Exception:
                pass

        return met

    def get_slo_stats(self):
        """Return SLO attainment counts keyed by (model_id, accuracy_level)."""
        return {k: tuple(v) for k, v in self.slo_stats.items()}

    def get_slo_violation_rate(self):
        """
        Rolling SLO violation rate across all requests.
        Returns a float in [0, 1] where 1.0 means every request missed its SLO.
        """
        served = sum(v[0] for v in self.slo_stats.values())
        if served == 0:
            return 0.0
        met = sum(v[1] for v in self.slo_stats.values())
        return 1.0 - (met / served)

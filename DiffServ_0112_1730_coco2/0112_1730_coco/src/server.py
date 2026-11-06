# Author: KB 
# Date: 2026-03-06 
# Purpose: Start autoscaler during system initialization 
# File: server.py 

import asyncio
import time
import os
import signal
from concurrent import futures
import grpc
import traceback
import threading

import config

import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2
import infaas_request_status_pb2_grpc

from scheduler import Scheduler
from worker_pool import WorkerPool
from autoscaler import AutoScaler
from metrics import Metrics # for recording metrics (2026.07.02)

try:
    from performance.optimizer import get_optimizer
except ImportError:  # optional; v6 behaviour must still work
    get_optimizer = None

############################################################
# 🔥 gRPC WRAPPER AROUND YOUR SCHEDULER
############################################################

class SchedulerService(query_pb2_grpc.QueryServicer):

    def __init__(self, scheduler, worker_pool, metrics):
        self.scheduler = scheduler
        self.worker_pool = worker_pool
        self.metrics = metrics

    

    async def QueryOnlineImage(self, request, context):
    #def QueryOnlineImage(self, request, context):
    
        # 🔥 START tracking load
        start_time = time.perf_counter()
        self.scheduler.increment_active()
        self.metrics.record_request()

        ############################################################
        # ENQUEUE REQUEST INTO SCHEDULER
        ############################################################
       
        print("[SERVER DEBUG] Request received")

        ############################################################
        # Record request arrival for scheduler statistics.
        # Do NOT enqueue protobuf requests.
        ############################################################
        with self.scheduler.lock:
            self.scheduler.arrival_count += 1

        print("[DEBUG-A] arrival counter updated")

        worker_addr = None

        print("[DEBUG] before worker selection")

        ############################################################
        # Scheduler owns dispatching
        ############################################################

        async for response in self.scheduler.dispatch_request(request):

            elapsed = time.perf_counter() - start_time

            self.metrics.record_latency(elapsed)

            self.metrics.record_service_time(elapsed)

            self.metrics.record_success()

            yield response



############################################################
# OPTIONAL PERFORMANCE OPTIMIZER
############################################################

def _start_optimizer_prefetch():
    """Log optimizer state and optionally warm the OS page cache.

    Prefetching only reads model files; it never loads them onto the GPU or
    changes which model a worker is serving.  Enable with:
        DIFS_PERF_OPTIMIZATION=1
        DIFS_PRELOAD_MODELS=all        (or e.g. "3,4,5")
    """
    if get_optimizer is None:
        return None
    optimizer = get_optimizer()
    if not optimizer.enabled:
        print("[SERVER] Performance optimizer DISABLED "
              "(set DIFS_PERF_OPTIMIZATION=1 to enable)")
        return optimizer

    print("[SERVER] Performance optimizer ENABLED")
    spec = os.environ.get("DIFS_PRELOAD_MODELS", "").strip()
    if not spec or spec.lower() in {"0", "none", "off"}:
        print("[SERVER] Model prefetch disabled "
              "(set DIFS_PRELOAD_MODELS=all or model ids)")
        return optimizer

    try:
        from model_registry import ALL_MODELS, get_model_filename
        from project_paths import MODEL_CACHE
    except Exception as exc:  # pragma: no cover - import environment dependent
        print(f"[SERVER] Prefetch skipped: {exc}")
        return optimizer

    if spec.lower() == "all":
        model_ids = sorted(ALL_MODELS)
    else:
        model_ids = []
        for token in spec.split(","):
            try:
                model_ids.append(int(token.strip()))
            except ValueError:
                print(f"[SERVER] Ignoring invalid prefetch model id: {token!r}")

    names = []
    for mid in model_ids:
        try:
            names.append(get_model_filename(mid))
        except ValueError:
            print(f"[SERVER] Unknown model id for prefetch: {mid}")

    summary = optimizer.prefetch_model_files(names, MODEL_CACHE)
    print(f"[SERVER] Model prefetch {summary}")
    return optimizer


############################################################
# 🔥 MAIN SERVER
############################################################

async def serve():

    # 0. METRICS: creating csv file to store performance metrics (2026.05.15)
    os.makedirs("metrics", exist_ok=True)

    # Global metrics object
    metrics = Metrics()
    print("[SERVER] Metrics subsystem initialized.")

    worker_pool = WorkerPool(metrics)

    # Opt-in optimizer: health-aware scheduling / circuit breakers / prefetch.
    _start_optimizer_prefetch()

    # 1. Initialize components (UNCHANGED)
    scheduler = Scheduler(worker_pool, config, metrics)
    

    # getting the server to consume requests; Vertical scaling does not need the queue-consumption thread
    if config.SCALING_MODE != "vertical":
        
        threading.Thread(
            target=scheduler.start,
            daemon=True
        ).start()

        print("[SERVER] Scheduler thread started")    


    ############################################################
    # VERTICAL MODE NEEDS ONE EXECUTION BACKEND
    ############################################################

    if config.SCALING_MODE == "vertical":
        
        ## REAL WORKER REGISTRATION
        # ===========================
        print("[DEBUG-B] entering vertical branch")
        print("[SERVER] Starting one Docker worker for vertical mode")
        await worker_pool.add_worker("balanced")

    else:
        # For horizontal/hybrid: register externally started workers
        import socket
        import json as _json

        _model_map_str = os.environ.get("DIFS_WORKER_MODEL_MAP", "{}")
        try:
            _model_map = {int(k): int(v) for k, v in _json.loads(_model_map_str).items()}
        except Exception:
            _model_map = {}

        # Collect ports: DIFS_EXTRA_WORKER_PORTS env var (comma-separated) + default 50060
        extra_ports_str = os.environ.get("DIFS_EXTRA_WORKER_PORTS", "")
        candidate_ports = [50060]
        if extra_ports_str:
            candidate_ports.extend(int(p.strip()) for p in extra_ports_str.split(",") if p.strip())

        for external_port in candidate_ports:
            sock = socket.socket()
            sock.settimeout(1)
            if sock.connect_ex(("localhost", external_port)) == 0:
                _mid = _model_map.get(external_port, 1)
                print(f"[SERVER] Found external worker on port {external_port} (model_id={_mid}), registering it")
                worker_pool.workers.append({
                    "container_id": f"external-{external_port}",
                    "port": external_port,
                    "workload_type": "balanced",
                    "model_id": _mid,
                    "sampler": 1,
                    "threads": 4,
                    "created_time": time.time()
                })
                worker_pool.next_port = max(worker_pool.next_port, external_port + 1)
            sock.close()

        print(f"[SERVER] Registered {len(worker_pool.workers)} external worker(s)")

    # Disabled either explicitly by the benchmark harness or by the hardcoded
    # baseline profile (e.g. the Clipper-style emulations have autoscaler off).
    no_autoscaler_env = os.environ.get("DIFS_NO_AUTOSCALER", "0") == "1"
    skip_autoscaler = no_autoscaler_env or not getattr(config, "AUTOSCALER_ENABLED", True)

    if skip_autoscaler:
        print("[SERVER] Autoscaler DISABLED (DIFS_NO_AUTOSCALER=1 or baseline profile)")
        autoscaler = None
    else:
        autoscaler = AutoScaler(
            scheduler,
            worker_pool,
            config
        )

        autoscaler.metrics = metrics
        autoscaler.start()

        print("[SERVER] Autoscaler started")

    # 2. Create gRPC server
    server = grpc.aio.server()

    query_pb2_grpc.add_QueryServicer_to_server(
        SchedulerService(scheduler, worker_pool, metrics),
        server
    )

    # 🔥 Scheduler now listens HERE
    server.add_insecure_port('[::]:50050')

    print("[SERVER] Scheduler gRPC running on port 50050")
    print(f"[SERVER] INFaaS config: GPU_MAX_REPLICAS={getattr(config, 'GPU_MAX_REPLICAS', 1)} "
          f"STRATEGY_THRESHOLD={getattr(config, 'STRATEGY_THRESHOLD', 1.5)} "
          f"SCALE_DOWN_DELAY={getattr(config, 'SCALE_DOWN_DELAY', 20)}")

    # Graceful shutdown
    await server.start()

    try:

        await server.wait_for_termination()

    except asyncio.CancelledError:

        pass

    finally:

        print("\n==============================")
        print("Stopping gRPC server...")
        print("==============================")

        await server.stop(grace=2)

        print("\n==============================")
        print("Generating Final Metrics Report")
        print("==============================")

        metrics.generate_summary()
        metrics.export_summary()
        metrics.close()

        print("Metrics saved successfully.")


############################################################
# OPTIONAL: background scheduler loop
############################################################

# async def run_scheduler_loop(scheduler):

#     while True:
#         try:
#             scheduler.schedule()   # or whatever your method is
#         except AttributeError:
#             pass

#         await asyncio.sleep(0.1)


############################################################
# ENTRY POINT
############################################################

if __name__ == "__main__":
    try:
        asyncio.run(serve())

    except KeyboardInterrupt:
        print("\n[SERVER] Graceful shutdown complete.")

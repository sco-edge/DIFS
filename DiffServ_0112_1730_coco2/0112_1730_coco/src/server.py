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

        try:
            async for response in self.scheduler.dispatch_request(request):

                yield response

        except Exception as e:

            self.metrics.record_error()
            self.metrics.record_failure()
            print(f"[SERVER ERROR] {e}")
            traceback.print_exc()

        finally:

            elapsed = time.perf_counter() - start_time
            self.metrics.record_latency(elapsed)
            self.metrics.record_service_time(elapsed)
            self.metrics.record_success()
            self.scheduler.decrement_active()



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
        
        # # FAKE WORKER REGISTRATION (FOR TESTING PURPOSES)
        # # ================================================
        # print("[DEBUG-B] entering vertical branch")
        # print("[SERVER] Using externally started worker")

        # worker_pool.workers = [
        #     {
        #         "container_id": "external",
        #         "port": 50060,
        #         "workload_type": "balanced",
        #         "model_id": 1,
        #         "sampler": 1,
        #         "threads": 4,
        #         "created_time": time.time()
        #     }
        # ]

        ## REAL WORKER REGISTRATION
        # ===========================
        print("[DEBUG-B] entering vertical branch")
        print("[SERVER] Starting one Docker worker for vertical mode")
        await worker_pool.add_worker("balanced")

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

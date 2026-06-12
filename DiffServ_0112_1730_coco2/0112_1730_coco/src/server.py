# Author: KB 
# Date: 2026-03-06 
# Purpose: Start autoscaler during system initialization 
# File: server.py 

import asyncio
import grpc
from concurrent import futures
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
import time
import csv
import os
import threading

############################################################
# 🔥 gRPC WRAPPER AROUND YOUR SCHEDULER
############################################################

class SchedulerService(query_pb2_grpc.QueryServicer):

    def __init__(self, scheduler, worker_pool):
        self.scheduler = scheduler
        self.worker_pool = worker_pool

    # async def QueryOnlineImage(self, request, context):

    #     # 🔥 Ask scheduler for next worker
    #     worker_addr = self.worker_pool.get_next_worker()

    #     print(f"[SCHEDULER] Routing request to {worker_addr}")

    #     try:
    #         async with grpc.aio.insecure_channel(worker_addr) as channel:

    #             stub = query_pb2_grpc.QueryStub(channel)

    #             async for response in stub.QueryOnlineImage(request):
    #                 yield response

    #     except Exception as e:
    #         print(f"[SCHEDULER ERROR] {e}")

    async def QueryOnlineImage(self, request, context):
    #def QueryOnlineImage(self, request, context):
    
        # 🔥 START tracking load
        start_time = time.perf_counter()
        self.scheduler.increment_active()

        ############################################################
        # ENQUEUE REQUEST INTO SCHEDULER
        ############################################################

        print("[SERVER DEBUG] Request received")
        self.scheduler.enqueue(request)

        worker_addr = None

        try:

            ############################################################
            # VERTICAL MODE → LOCAL EXECUTION
            ############################################################
            if config.SCALING_MODE == "vertical":
                print("[SERVER] Vertical mode active")
                worker_addr = self.worker_pool.get_next_worker()
                print(
                    f"[VERTICAL ROUTER] Using worker {worker_addr}"
                )

            ############################################################
            # HORIZONTAL/HYBRID MODE
            ############################################################
            if worker_addr is None:
                worker_addr = self.worker_pool.get_next_worker() # this preserve horizontal scaling
                print(f"[SCHEDULER] Routing request to {worker_addr}") # PNB (2026.04.10)
            
            ############################################################
            # gRPC HEALTH CHECK
            ############################################################

            try:

                async with grpc.aio.insecure_channel(worker_addr) as health_channel:

                    health_stub = query_pb2_grpc.QueryStub(health_channel)

                    heartbeat = await health_stub.Heartbeat(
                        query_pb2.HeartbeatRequest()
                    )

                    if heartbeat.status.status != 1:

                        print(f"[HEALTH CHECK] Worker unhealthy: {worker_addr}")
                        return

            except Exception as e:

                print(f"[HEALTH CHECK FAILED] {worker_addr}: {e}")
                return

            # 🔥 RETRY LOGIC
            # import asyncio

            success = False

            for attempt in range(3):

                try:
                    print(f"[RETRY] Attempt {attempt+1} → {worker_addr}")

                    async with grpc.aio.insecure_channel(worker_addr) as channel:

                        ## =======  1 CLIENT REQUEST -> 1 WORKER CALL ========

                        # stub = query_pb2_grpc.QueryStub(channel)

                        # async for response in stub.QueryOnlineImage(request): # for direct forwarding
                        #     elapsed = time.perf_counter() - start_time # PNB (2026.04.10)
                        #     print(f"[SERVER LATENCY] {worker_addr}: {elapsed:.3f}s") # PNB (2026.04.10)

                        #     yield response


                        # =======  MULTIPLE CLIENT REQUESTS -> batched -> 1 WORKER CALL ========
                        stub = query_pb2_grpc.QueryStub(channel)

                        async for response in stub.QueryOnlineImage(request):

                            elapsed = time.perf_counter() - start_time

                            print(f"[SERVER LATENCY] {worker_addr}: {elapsed:.3f}s")
                            print(f"[INFERENCE SUCCESS] Worker={worker_addr}")

                            throughput = 1 / elapsed

                            print(f"[THROUGHPUT] {throughput:.2f} req/sec")

                            ############################################################
                            # METRICS LOGGING
                            ############################################################

                            with open("metrics/server_metrics.csv", "a", newline="") as f:

                                writer = csv.writer(f)

                                writer.writerow([
                                    time.time(),
                                    worker_addr,
                                    elapsed,
                                    self.scheduler.get_total_queue_length(),
                                    getattr(self.scheduler, "arrival_rate", 0),
                                    getattr(self.scheduler, "service_rate", 0),
                                    len(self.worker_pool.workers)
                                ])

                            yield response

                    success = True
                    break

                except Exception as e:
                    print(f"[RETRY {attempt+1}] Failed:")
                    traceback.print_exc()
                    await asyncio.sleep(2)

            if not success:
                print(f"[SCHEDULER ERROR] Worker {worker_addr} failed after retries")

        except Exception as e:
            print(f"[SCHEDULER ERROR] {e}")

        finally:
            # 🔥 END tracking load
            self.scheduler.decrement_active()


############################################################
# 🔥 MAIN SERVER
############################################################

async def serve():

    # 0. METRICS: creating csv file to store performance metrics (2026.05.15)
    os.makedirs("metrics", exist_ok=True)
    
    server_metrics_file = "metrics/server_metrics.csv"

    if not os.path.exists(server_metrics_file):

        with open(server_metrics_file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "worker",
                "latency",
                "queue_length",
                "arrival_rate",
                "service_rate",
                "workers"
            ])


    # 1. Initialize components (UNCHANGED)
    scheduler = Scheduler()

    ## getting the server to consume requests
    threading.Thread(
        target=scheduler.start,
        daemon=True
    ).start()

    print("[SERVER] Scheduler thread started")

    worker_pool = WorkerPool()


    ############################################################
    # VERTICAL MODE NEEDS ONE EXECUTION BACKEND
    ############################################################

    if config.SCALING_MODE == "vertical":

        print("[SERVER] Starting single worker for vertical mode")

        await worker_pool.add_worker("balanced")


    autoscaler = AutoScaler(
        scheduler,
        worker_pool,
        config
    )

    autoscaler.start()

    print("[SERVER] Autoscaler started")

    # 2. Create gRPC server
    server = grpc.aio.server()

    query_pb2_grpc.add_QueryServicer_to_server(
        SchedulerService(scheduler, worker_pool),
        server
    )

    # 🔥 Scheduler now listens HERE
    server.add_insecure_port('[::]:50050')

    print("[SERVER] Scheduler gRPC running on port 50050")

    await server.start()

    # # 🔥 run scheduler loop in background if needed
    # asyncio.create_task(run_scheduler_loop(scheduler))

    await server.wait_for_termination()


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
    asyncio.run(serve())

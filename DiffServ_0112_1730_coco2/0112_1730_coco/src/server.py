# Author: KB 
# Date: 2026-03-06 
# Purpose: Start autoscaler during system initialization 
# File: server.py 

import asyncio
import grpc
from concurrent import futures

import config

import query_pb2
import query_pb2_grpc

from scheduler import Scheduler
from worker_pool import WorkerPool
from autoscaler import AutoScaler
import time


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

        worker_addr = None

        try:
            worker_addr = self.worker_pool.get_next_worker()
            print(f"[SCHEDULER] Routing request to {worker_addr}") # PNB (2026.04.10)
            # 🔥 HEALTH CHECK
            import socket

            def is_port_open(host, port):
                s = socket.socket()
                try:
                    s.settimeout(1)
                    s.connect((host, port))
                    return True
                except Exception:
                    return False
                finally:
                    s.close()

            # 🔥 Extract host + port
            host, port = worker_addr.split(":")
            port = int(port)

            # 🔥 Health check
            if not is_port_open(host, port):
                print(f"[HEALTH CHECK] Worker {worker_addr} not ready")
                return

            print(f"[SCHEDULER] Active={self.scheduler.get_total_queue_length()} → {worker_addr}")

            # 🔥 RETRY LOGIC
            # import asyncio

            success = False

            for attempt in range(3):

                try:
                    print(f"[RETRY] Attempt {attempt+1} → {worker_addr}")

                    async with grpc.aio.insecure_channel(worker_addr) as channel:

                        stub = query_pb2_grpc.QueryStub(channel)

                        async for response in stub.QueryOnlineImage(request):
                            elapsed = time.perf_counter() - start_time # PNB (2026.04.10)
                            print(f"[SERVER LATENCY] {worker_addr}: {elapsed:.3f}s") # PNB (2026.04.10)

                            yield response

                    success = True
                    break

                except Exception as e:
                    print(f"[RETRY {attempt+1}] Failed: {e}")
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

    # 1. Initialize components (UNCHANGED)
    scheduler = Scheduler()
    worker_pool = WorkerPool()

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

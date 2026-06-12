# Author: KB 
# Date: 2026-03-06 
# Purpose: For adding scaling capability 
# File: worker_pool.py 

import subprocess
import time
import os

# 2026.05.15
import asyncio
import grpc
import query_pb2
import query_pb2_grpc

from model_registry import MODEL_CONFIGS


class WorkerPool:

    def __init__(self):

        self.workers = []
        self.base_port = 50060
        self.next_port = self.base_port


    ############################################################
    # WAIT UNTIL WORKER IS FULLY READY
    ############################################################

    async def wait_until_worker_ready(self, addr, timeout=600): # PNB: increase to 600 because stable diffusion loads can easily exceed 300 s
        '''
        Health wait function; delays or blocks worker or pool manager until specific health criteria is met
        '''

        start = time.time()

        while time.time() - start < timeout:

            try:
                async with grpc.aio.insecure_channel(addr) as channel:

                    stub = query_pb2_grpc.QueryStub(channel)

                    response = await stub.Heartbeat(
                        query_pb2.HeartbeatRequest()
                    )

                    if response.status.status == 1:
                        print(f"[WORKER_POOL] Worker READY: {addr}")
                        return True

            except Exception as e:
                print(f"[WORKER_POOL HEALTHCHECK] {e}")

            print(f"[WORKER_POOL] Waiting for worker {addr}...")
            await asyncio.sleep(2)

        print(f"[WORKER_POOL ERROR] Worker timeout: {addr}")
        return False

    def size(self):
        return len(self.workers)

    async def add_worker(self, workload_type="balanced"):
        # -----------------------------------------
        # Dynamic model selection
        # -----------------------------------------
        config = MODEL_CONFIGS[workload_type]

        model_id = str(config["model_id"])
        sampler = str(config["sampler"])
        threads = str(config["threads"])


        print(f"[WORKER_POOL] Launching workload type: {workload_type}")
        print(f"[WORKER_POOL] Model={model_id}, Sampler={sampler}")
        

        #port = self.base_port + len(self.workers)# new port generated for new worker being added
        
        port = self.next_port
        self.next_port += 1

        # worker = InferenceWorker()
        #  worker.start()
        print(f"Execution Started In Directory: {os.getcwd()}")

        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",

            #-------------------------------------
            # Docker directory
            #-------------------------------------
            "-v", f"{os.getcwd()}:/workspace", # -v mounts my project in container
            "-w", "/workspace", # -w sets working directory

            #-------------------------------------
            # Port mapping
            #-------------------------------------
            # "-p", f"{port}:{port}",
            "-p", f"{port}:50060",

            #-------------------------------------
            # Volume mounts
            #-------------------------------------
            "-v", "/tmp/model:/tmp/model",
            "-v", "/tmp/diffusion_output:/tmp/diffusion_output",

            #-------------------------------------
            # Image name
            #-------------------------------------

            # Use an EXISTING base image
            "difs-diffusion-worker",

            #"pytorch-diffusion-server",
            "python", "src/1/pytorch_container_diffusion_v3.py",

            #-------------------------------------
            # Arguments passed to python server
            #-------------------------------------
            # "-model", "4",
            # #"-model", "/tmp/model/sd-v1-4.saftetensors", # explicitly pass full model path
            # "-sampler", "2",
            # "-thread", "10",
            "-model", model_id,
            "-sampler", sampler,
            "-thread", threads,
            
            "-port", "50060",
            "-npz", "/tmp/model/real_stats.npz"
        ]

        # Handle Docker failure gracefully
        try:
            # container_id = subprocess.check_output(cmd).decode().strip()

            # Adding debugging visibility (2026.04.23)
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print("[WORKER ERROR]")
                print(result.stderr)
                print(result.stdout)
                return

            container_id = result.stdout.strip()
            print(f"[WORKER_POOL] Container ID: {container_id}")

        except subprocess.CalledProcessError as e:
            print(f"[WORKER_POOL ERROR] Failed to start container: {e}")
            subprocess.call(["docker", "logs", container_id])
            return

        print(f"[WORKER_POOL] Waiting for worker on port {port}...")

        # ---------------------------------------------------
        # DEBUG: show worker logs during startup
        # ---------------------------------------------------
        time.sleep(5)

        print("\n[WORKER_POOL DEBUG] Docker container logs:\n")

        subprocess.call([
            "docker",
            "logs",
            container_id
        ])

        print("\n[WORKER_POOL DEBUG END]\n")

        #time.sleep(8)   # 🔥 increase to 8–10 seconds

        # PNB Instead of using sleep to wait, run only if worker is healthy or passes health check test # (2026.06.15)
        worker_addr = f"localhost:{port}"

        # ready = asyncio.run(
        #     self.wait_until_worker_ready(worker_addr)
        # )
        

        ready = await  self.wait_until_worker_ready(worker_addr)
        

        #loop.close()


        if not ready:
            print("[WORKER_POOL ERROR] Worker failed startup")
            
            # To prevent dead containers from accumulating
            subprocess.call(["docker", "logs", container_id])
            subprocess.call(["docker", "rm","-f", container_id])

            return

        print(f"[WORKER_POOL] Worker READY: {worker_addr}")
        
        #self.workers.append((container_id, port))
        self.workers.append({
            "container_id": container_id,
            "port": port,
            "workload_type": workload_type,
            "model_id": model_id
        })
        

        print(f"[WORKER_POOL] Started container {container_id[:8]} on port {port}")

    def remove_worker(self):

        if not self.workers:
            return

        container_id, port = self.workers.pop()
        worker = self.workers.pop()

        container_id = worker["container_id"]
        port = worker["port"]

        subprocess.call(["docker", "stop", container_id])

        print(f"[WORKER_POOL] Stopped container {container_id[:8]} on port {port}")

    def get_worker_ports(self):

        #return [port for _, port in self.workers]
        return [w["port"] for w in self.workers]

    def get_next_worker(self):
    
        if not self.workers:
            raise RuntimeError("No workers available!")

        if not hasattr(self, "idx"):
            self.idx = 0

        worker = self.workers[self.idx % len(self.workers)]
        self.idx += 1

        #return f"localhost:{worker[1]}"
        return f"localhost:{worker['port']}"

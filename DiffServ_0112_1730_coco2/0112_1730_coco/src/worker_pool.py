# Author: KB 
# Date: 2026-03-06 
# Purpose: For adding scaling capability 
# File: worker_pool.py 

import subprocess
import time

class WorkerPool:

    def __init__(self):

        self.workers = []
        self.base_port = 50060
        self.next_port = self.base_port

    def size(self):
        return len(self.workers)

    def add_worker(self):

        #port = self.base_port + len(self.workers)# new port generated for new worker being added
        
        port = self.next_port
        self.next_port += 1

        # worker = InferenceWorker()
        #  worker.start()

        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",
            # "-p", f"{port}:{port}",
            "-p", f"{port}:50060",
            "-v", "/tmp/model:/tmp/model",
            "-v", "/tmp/diffusion_output:/tmp/diffusion_output",
            "pytorch-diffusion-server",
            #"python", "pytorch_container_diffusion_async_v2.py",
            "-model", "4",
            "-sampler", "2",
            "-thread", "10",
            "-port", "50060",
            "-npz", "/tmp/model/real_stats.npz"
        ]

        # Handle Docker failure gracefully
        try:
            container_id = subprocess.check_output(cmd).decode().strip()
        except subprocess.CalledProcessError as e:
            print(f"[WORKER_POOL ERROR] Failed to start container: {e}")
            return

        print(f"[WORKER_POOL] Waiting for worker on port {port}...")

        time.sleep(8)   # 🔥 increase to 8–10 seconds

        self.workers.append((container_id, port))

        print(f"[WORKER_POOL] Started container {container_id[:8]} on port {port}")

    def remove_worker(self):

        if not self.workers:
            return

        container_id, port = self.workers.pop()

        subprocess.call(["docker", "stop", container_id])

        print(f"[WORKER_POOL] Stopped container {container_id[:8]} on port {port}")

    def get_worker_ports(self):

        return [port for _, port in self.workers]

    def get_next_worker(self):
    
        if not self.workers:
            raise RuntimeError("No workers available!")

        if not hasattr(self, "idx"):
            self.idx = 0

        worker = self.workers[self.idx % len(self.workers)]
        self.idx += 1

        return f"localhost:{worker[1]}"

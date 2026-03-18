# Author: KB 
# Date: 2026-03-06 
# Purpose: For adding scaling capability 
# File: worker_pool.py 

import subprocess

class WorkerPool:

    def __init__(self):

        self.workers = []
        self.base_port = 50060

    def size(self):
        return len(self.workers)

    def add_worker(self):

        port = self.base_port + len(self.workers)# new port generated for new worker being added

        # worker = InferenceWorker()
        #  worker.start()

        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",
            "-p", f"{port}:{port}",
            "-v", "/tmp/model:/tmp/model",
            "-v", "/tmp/diffusion_output:/tmp/diffusion_output",
            "pytorch-diffusion-server",
            "-model", "4",
            "-sampler", "2",
            "-thread", "10",
            "-port", str(port),
            "-npz", "/tmp/model/real_stats.npz"
        ]

        container_id = subprocess.check_output(cmd).decode().strip()

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

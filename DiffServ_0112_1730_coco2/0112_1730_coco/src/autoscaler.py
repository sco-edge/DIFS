# Author: KB 
# Date: 2026-03-06 
# Purpose: For doing the autoscaling for the simple Diffusion Model (2026.03.06) 
# File: autoscaler.py 

import time
import subprocess
import threading


class AutoScaler:

    def __init__(self, scheduler, worker_pool, config):

        self.scheduler = scheduler
        self.worker_pool = worker_pool

        self.min_workers = config.MIN_WORKERS
        self.max_workers = config.MAX_WORKERS

        self.scale_up_threshold = config.SCALE_UP_THRESHOLD
        self.scale_down_threshold = config.SCALE_DOWN_THRESHOLD

        self.check_interval = config.CHECK_INTERVAL

        self.running = False

        # track ports
        self.next_port = 50060
        self.active_ports = []

    ############################################################
    # DOCKER CONTROL
    ############################################################

    def start_worker(self):

        port = self.next_port
        self.next_port += 1

        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-p", f"{port}:50060",
            "-v", "/tmp/model:/tmp/model",
            "-v", "/tmp/diffusion_output:/tmp/diffusion_output",
            "pytorch-diffusion-server",
            "-model", "4",
            "-sampler", "2",
            "-thread", "10",
            "-port", "50060",
            "-npz", "/tmp/model/real_stats.npz"
        ]

        print(f"[AUTOSCALER] Starting worker on port {port}")

        subprocess.Popen(cmd)

        self.worker_pool.add_worker(port)
        self.active_ports.append(port)

    def stop_worker(self):

        if not self.active_ports:
            return

        port = self.active_ports.pop()

        print(f"[AUTOSCALER] Stopping worker on port {port}")

        # kill container using port
        subprocess.call(f"docker ps | grep {port} | awk '{{print $1}}' | xargs docker stop", shell=True)

        self.worker_pool.remove_worker()

    ############################################################
    # SCALING LOGIC
    ############################################################

    def scale(self):

        try:
            queue_length = self.scheduler.get_total_queue_length()
        except:
            queue_length = 0

        worker_count = self.worker_pool.get_worker_count()

        print(f"[AUTOSCALER] Queue={queue_length}, Workers={worker_count}")

        # SCALE UP
        if queue_length > self.scale_up_threshold and worker_count < self.max_workers:
            self.start_worker()

        # SCALE DOWN
        elif queue_length < self.scale_down_threshold and worker_count > self.min_workers:
            self.stop_worker()

    ############################################################
    # MAIN LOOP
    ############################################################

    def run(self):

        # start with minimum workers
        for _ in range(self.min_workers):
            self.start_worker()

        while self.running:
            self.scale()
            time.sleep(self.check_interval)

    def start(self):

        self.running = True

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

# Author: KB 
# Date: 2026-03-06 
# Purpose: For doing the autoscaling for the simple Diffusion Model (2026.03.06) 
# File: autoscaler.py 

import time
import subprocess
import threading
from collections import Counter

class AutoScaler:

    def __init__(self, scheduler, worker_pool, config):

        self.scheduler = scheduler
        self.worker_pool = worker_pool
        self.config = config

        self.min_workers = config.MIN_WORKERS
        self.max_workers = config.MAX_WORKERS

        self.scale_up_threshold = config.SCALE_UP_THRESHOLD
        self.scale_down_threshold = config.SCALE_DOWN_THRESHOLD

        self.check_interval = config.CHECK_INTERVAL

        self.running = False

    ############################################################
    # WORKLOAD ANALYSIS
    ############################################################

    def get_dominant_workload(self):

        workloads = []

        try:

            for q in self.scheduler.class_queues:

                items = list(q.queue)

                for req in items:

                    if hasattr(req, "workload_type"):
                        workloads.append(req.workload_type)

        except Exception as e:

            print(f"[AUTOSCALER WARNING] Workload inspection failed: {e}")

        if not workloads:
            return "balanced"

        dominant = Counter(workloads).most_common(1)[0][0]

        print(f"[AUTOSCALER] Dominant workload: {dominant}")

        return dominant

        

    ############################################################
    # DOCKER CONTROL
    ############################################################

    def start_worker(self, workload_type="balanced"):
    
        print("[AUTOSCALER] Requesting worker_pool to add worker")

        before = self.worker_pool.size()

        #self.worker_pool.add_worker()
        self.worker_pool.add_worker(workload_type)

        time.sleep(5)  # give worker time to boot

        after = self.worker_pool.size()

        # ADDING FAILURE VISIBILITY
        if after == before:
            print("[AUTOSCALER ERROR] Worker failed to start!")
        else:
            print("[AUTOSCALER] Worker successfully added")

    def stop_worker(self):
    
        print("[AUTOSCALER] Requesting worker_pool to remove worker")

        self.worker_pool.remove_worker()

    ############################################################
    # SCALING LOGIC
    ############################################################

    def scale(self):

        try:
            queue_length = self.scheduler.get_total_queue_length()
        except:
            queue_length = 0

        worker_count = self.worker_pool.size()

        # -----------------------------------------
        # Determine dominant workload
        # -----------------------------------------

        dominant_workload = self.get_dominant_workload()

        if worker_count == 0:
            print("[AUTOSCALER WARNING] No workers available!")

        print(f"[AUTOSCALER] Queue={queue_length}, Workers={worker_count}")

        # SCALE UP
        # if queue_length > self.scale_up_threshold and worker_count < self.max_workers:
        #     self.start_worker()

        if queue_length > self.scale_up_threshold and worker_count < self.max_workers:    
            print(f"[AUTOSCALER] Scaling for workload: {dominant_workload}")
            self.start_worker(dominant_workload)

        # SCALE DOWN
        elif queue_length < self.scale_down_threshold and worker_count > self.min_workers:
            self.stop_worker()


    def scale_vertical(self):
        
        try:
            queue_length = self.scheduler.get_total_queue_length()
        except:
            queue_length = 0

        # vertical scaling depends on the following
        arrival_rate = self.scheduler.get_arrival_rate()
        service_rate = self.scheduler.get_service_rate()

        print(
            f"[VERTICAL AUTOSCALER] Queue={queue_length}, "
            f"λ={arrival_rate:.2f}, μ={service_rate:.2f}"
        )

        # 🔥 Dynamic batching logic
        if queue_length > 10:
            self.scheduler.set_batch_size(8)
            print("[VERTICAL] Batch size → 8")

        elif queue_length > 5:
            self.scheduler.set_batch_size(4)
            print("[VERTICAL] Batch size → 4")

        else:
            self.scheduler.set_batch_size(1)
            print("[VERTICAL] Batch size → 1")

    ############################################################
    # MAIN LOOP
    ############################################################

    def run(self):
    
        # 🔥 Step 1: Start minimum workers (horizontal base)
        # Start workers ONLY for horizontal or hybrid scaling
        if self.config.SCALING_MODE in ["horizontal", "hybrid"]:

            for _ in range(self.min_workers):
                # self.start_worker()
                self.start_worker("balanced")

            print("[AUTOSCALER] Initial workers launched. Waiting for stabilization...")
            time.sleep(10)

        else:
            print("[AUTOSCALER] Vertical mode selected — skipping worker container startup")

        print("[AUTOSCALER] Initial workers launched. Waiting for stabilization...")
        time.sleep(10)

        # 🔥 Step 2: Continuous scaling loop
        while self.running:

            if self.config.SCALING_MODE == "vertical":
                self.scale_vertical()

            elif self.config.SCALING_MODE == "horizontal":
                self.scale()

            else:  # hybrid (recommended)
                self.scale_vertical()
                self.scale()

            time.sleep(self.check_interval)

    def start(self):

        self.running = True

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

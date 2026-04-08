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

        

    ############################################################
    # DOCKER CONTROL
    ############################################################

    def start_worker(self):
    
        print("[AUTOSCALER] Requesting worker_pool to add worker")

        before = self.worker_pool.size()

        self.worker_pool.add_worker()

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

        if worker_count == 0:
            print("[AUTOSCALER WARNING] No workers available!")

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

        print("[AUTOSCALER] Initial workers launched. Waiting for stabilization...")
        time.sleep(20)

        # while self.running:
        #     self.scale()
        #     time.sleep(self.check_interval)

    def start(self):

        self.running = True

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

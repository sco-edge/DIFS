# Author: KB 
# Date: 2026-03-06 
# Purpose: For doing the autoscaling for the simple Diffusion Model (2026.03.06) 
# File: autoscaler.py 

import threading
import time

class AutoScaler:

    def __init__(self, scheduler, worker_pool, config):

        self.scheduler = scheduler
        self.worker_pool = worker_pool
        self.config = config
        self.running = True

    def scale_logic(self):

        while self.running:

            qlen = self.scheduler.get_total_queue_length()
            workers = self.worker_pool.size()

            if qlen > self.config.SCALE_UP_THRESHOLD:
                if workers < self.config.MAX_WORKERS:
                    self.worker_pool.add_worker()

            elif qlen < self.config.SCALE_DOWN_THRESHOLD:
                if workers > self.config.MIN_WORKERS:
                    self.worker_pool.remove_worker()

            time.sleep(self.config.AUTOSCALE_INTERVAL)

    def start(self):

        t = threading.Thread(target=self.scale_logic)
        t.daemon = True
        t.start()

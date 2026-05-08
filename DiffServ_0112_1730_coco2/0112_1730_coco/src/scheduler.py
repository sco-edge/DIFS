# Author: KB 
# Date: 2026-03-06 
# Purpose: Expose queue metrics 
# File: diffserv_scheduler.py 

import queue
import time
import threading


class Scheduler:

    def __init__(self):

        self.batch_size = 1

        self.class_queues = [
            queue.Queue(),
            queue.Queue(),
            queue.Queue()
        ] # for scheduling / priority

        self.request_queue = queue.Queue() # for batching / throughput

        self.arrival_count = 0
        self.service_count = 0

        self.arrival_window_start = time.time()
        self.service_window_start = time.time()

        self.lock = threading.Lock()

        self.active_requests = 0;

  ############################################################
    # ACTIVE REQUEST TRACKING (FOR AUTOSCALER)
  ############################################################        
        
    def increment_active(self):
        with self.lock:
            self.active_requests += 1

    def decrement_active(self):
        with self.lock:
            if self.active_requests > 0:
                self.active_requests -= 1

    ############################################################
    # CORE METHODS (UNCHANGED)
    ############################################################                
    def handle_request(self, request):

        print("[SCHEDULER] Processing request:", request)


    def start(self):

        print("[SCHEDULER] Scheduler started")

        while True:

            request = self.dequeue()

            if request is not None:

                self.handle_request(request)

            else:
                time.sleep(0.01)

    ############################################################
    # 🔥 UPDATED METRIC (KEY CHANGE)
    ############################################################                
    def get_total_queue_length(self):
        queue_total = sum(q.qsize() for q in self.class_queues)
        
        with self.lock:
            active = self.active_requests

        return queue_total + active # now metric becomes TOTAL LOAD = queued requests + active requests

    
    ############################################################
    # EXISTING METRICS (UNCHANGED)
    ############################################################
    def enqueue(self, request, cls=2):
        '''
        Goes into both queues
        '''

        self.class_queues[cls].put(request)

        self.request_queue.put(request)

        with self.lock:
            self.arrival_count += 1


    def dequeue(self):

        for q in self.class_queues:

            if not q.empty():

                with self.lock:
                    self.service_count += 1

                return q.get()

        return None


    def get_arrival_rate(self):

        now = time.time()

        with self.lock:

            elapsed = now - self.arrival_window_start

            if elapsed == 0:
                return 0

            rate = self.arrival_count / elapsed

            self.arrival_count = 0
            self.arrival_window_start = now

        return rate


    def get_service_rate(self): 

        now = time.time()

        with self.lock:

            elapsed = now - self.service_window_start

            if elapsed == 0:
                return 0

            rate = self.service_count / elapsed

            self.service_count = 0
            self.service_window_start = now

        return rate

    def set_batch_size(self, size): # PNB(2026.04.23)
        self.batch_size = size

    def get_batch(self): # Core of vertical scaling. PNB(2026.04.23)
        batch = []

        while len(batch) < self.batch_size:
            try:
                req = self.request_queue.get(timeout=0.01)
                batch.append(req)
            except:
                break
        
        return batch

    def get_batch_size(self):
        return self.batch_size

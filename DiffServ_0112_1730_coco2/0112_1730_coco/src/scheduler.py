# Author: KB 
# Date: 2026-03-06 
# Purpose: Expose queue metrics 
# File: diffserv_scheduler.py 

import queue
import time
import threading


class Scheduler:

    def __init__(self):

        self.class_queues = [
            queue.Queue(),
            queue.Queue(),
            queue.Queue()
        ]

        self.arrival_count = 0
        self.service_count = 0

        self.arrival_window_start = time.time()
        self.service_window_start = time.time()

        self.lock = threading.Lock()

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

    def get_total_queue_length(self):

        total = 0

        for q in self.class_queues:
            total += q.qsize()

        return total


    def enqueue(self, request, cls=2):

        self.class_queues[cls].put(request)

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

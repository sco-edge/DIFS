# Author: KB 
# Date: 2026-03-06 
# Purpose: For adding scaling capability 
# File: worker_pool.py 

class WorkerPool:

    def __init__(self):

        self.workers = []

    def add_worker(self):

        worker = InferenceWorker()
        worker.start()
        self.workers.append(worker)

    def remove_worker(self):

        if self.workers:
            worker = self.workers.pop()
            worker.stop()

    def size(self):

        return len(self.workers)

# Author: KB 
# Date: 2026-03-06 
# Purpose: Expose queue metrics 
# File: diffserv_scheduler.py 

def get_total_queue_length(self):

    total = 0

    for q in self.class_queues:
        total += q.qsize()

    return total

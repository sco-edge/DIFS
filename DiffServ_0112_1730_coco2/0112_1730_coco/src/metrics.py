# Author: KB 
# Date: 2026-03-06 
# Purpose: For metrics calculations 
# File: metrics.py 

class Metrics:

    def __init__(self):

        self.latencies = []
        self.request_count = 0

    def record_latency(self, latency):

        self.latencies.append(latency)

    def avg_latency(self):

        if not self.latencies:
            return 0

        return sum(self.latencies) / len(self.latencies)

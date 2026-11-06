# Author: KB
# Date: 2026-06-21 
# Purpose:
#

import csv
import os
import threading
import time


class WorkerPoolMetrics:

    def __init__(self, output_dir="metrics"):

        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.file = os.path.join(
            output_dir,
            "worker_pool_metrics.csv"
        )

        self.lock = threading.Lock()

        if not os.path.exists(self.file):

            with open(self.file,
                      "w",
                      newline="",
                      encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "worker_port",
                    "worker_id",
                    "model_name",
                    "busy",
                    "active_requests",
                    "gpu_memory_mb",
                    "throughput",
                    "average_latency_ms",
                    "uptime_sec",
                    "event"
                ])

    ###########################################################
    # CSV helper
    ###########################################################

    def append(self, row):

        with self.lock:

            with open(
                    self.file,
                    "a",
                    newline="",
                    encoding="utf-8") as f:

                writer = csv.writer(f)
                writer.writerow(row)

    ###########################################################
    # Worker created
    ###########################################################

    def worker_created(
            self,
            worker_port,
            worker_id):

        self.append([
            time.time(),
            worker_port,
            worker_id,
            "",
            False,
            0,
            0,
            0,
            0,
            0,
            "created"
        ])

    ###########################################################
    # Worker removed
    ###########################################################

    def worker_removed(
            self,
            worker_port,
            worker_id):

        self.append([
            time.time(),
            worker_port,
            worker_id,
            "",
            False,
            0,
            0,
            0,
            0,
            0,
            "removed"
        ])

    ###########################################################
    # Runtime update
    ###########################################################

    def worker_runtime(
            self,
            worker_port,
            worker_id,
            model_name,
            busy,
            active_requests,
            gpu_memory,
            throughput,
            latency,
            uptime):

        self.append([
            time.time(),
            worker_port,
            worker_id,
            model_name,
            busy,
            active_requests,
            gpu_memory,
            throughput,
            latency,
            uptime,
            "heartbeat"
        ])
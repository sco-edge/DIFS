# Author: KB
# Date: 2026-06-20
# Purpose:
# Worker-side metrics logging for DIFS
#

import csv
import os
import threading
import time

import uuid
import torch


class WorkerMetrics:

    def __init__(self,
                 output_dir="metrics"):

        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.file = os.path.join(
            output_dir,
            "worker_metrics.csv"
        )

        self.lock = threading.Lock()
        ###########################################################
        # Worker ID
        ###########################################################

        self.worker_id = str(uuid.uuid4())[:8]

        if not os.path.exists(self.file):

            with open(self.file,
                      "w",
                      newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "worker_id",
                    "worker_port",
                    "model_id",
                    "model_name",
                    "sampler",
                    "batch_size",
                    "steps",
                    "latency_sec",
                    "gpu_memory_MB",
                    "gpu_reserved_MB",
                    "gpu_peak_MB",
                    "fid",
                    "images_generated",
                    "success",
                    "request_id"
                ])

        self.total_requests = 0
        self.total_images = 0
        self.total_latency = 0.0
        self.last_fid = 0.0
        self.start_time = time.time()


    ###########################################################
    # GPU MEMORY
    ###########################################################

    def gpu_memory(self):

        if not torch.cuda.is_available():

            return (0, 0, 0)

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)

        reserved = torch.cuda.memory_reserved() / (1024 ** 2)

        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return (
            round(allocated, 2),
            round(reserved, 2),
            round(peak, 2)
        )


    ###########################################################
    # MAIN LOGGER
    ###########################################################

    def log_inference(
                self,
                worker_port,
                model_id,
                model_name,
                sampler,
                batch_size,
                steps,
                latency,
                fid,
                images_generated,
                success,
                request_id=None):

        allocated, reserved, peak = self.gpu_memory()

        with self.lock:

            with open(self.file, "a", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    time.time(),
                    self.worker_id,
                    worker_port,
                    model_id,
                    model_name,
                    sampler,
                    batch_size,
                    steps,
                    latency,
                    allocated,
                    reserved,
                    peak,
                    fid,
                    images_generated,
                    success,
                    request_id
                ])
            self.total_requests += 1
            self.total_images += images_generated
            self.total_latency += latency
            self.last_fid = fid
    
    ###########################################################
    # CSV APPEND HELPER
    ###########################################################

    def append_csv(self, filename, row):

        with self.lock:

            with open(filename, "a", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)
                writer.writerow(row)

    ###########################################################
    # MODEL LOAD LOGGER
    ###########################################################

    def log_model_load(self, model, load_time):

        file = os.path.join(
            self.output_dir,
            "model_load_metrics.csv"
        )

        if not os.path.exists(file):

            with open(file, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "model",
                    "load_time"
                ])

        self.append_csv(
            file,
            [
                time.time(),
                model,
                load_time
            ]
        )

    ###########################################################
    # MODEL SWITCH LOGGER
    ###########################################################

    def log_model_switch(
            self,
            old_model,
            new_model,
            switch_time):

        file = os.path.join(
            self.output_dir,
            "model_switch_metrics.csv"
        )

        if not os.path.exists(file):

            with open(file, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "old_model",
                    "new_model",
                    "switch_time"
                ])

        self.append_csv(
            file,
            [
                time.time(),
                old_model,
                new_model,
                switch_time
            ]
        )

    ###########################################################
    # GPU MEMORY LOGGER
    ###########################################################

    def log_gpu_memory(
            self,
            allocated_mb,
            reserved_mb):

        file = os.path.join(
            self.output_dir,
            "gpu_memory_metrics.csv"
        )

        if not os.path.exists(file):

            with open(file, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "allocated_mb",
                    "reserved_mb"
                ])

        self.append_csv(
            file,
            [
                time.time(),
                allocated_mb,
                reserved_mb
            ]
        )

    ###########################################################
    # FID LOGGER
    ###########################################################

    def log_fid(self, fid):
    
        file = os.path.join(
            self.output_dir,
            "fid_metrics.csv"
        )

        if not os.path.exists(file):

            with open(file, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "worker_id",
                    "fid"
                ])

        self.append_csv(
            file,
            [
                time.time(),
                self.worker_id,
                fid
            ]
        )

    ###########################################################
    # HEARTBEAT
    ###########################################################

    def record_heartbeat(self):
    
        file = os.path.join(
            self.output_dir,
            "heartbeat_metrics.csv"
        )

        if not os.path.exists(file):

            with open(file,
                    "w",
                    newline="",
                    encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "worker_id"
                ])

        self.append_csv(
            file,
            [
                time.time(),
                self.worker_id
            ]
        )
    
    def current_throughput(self):
        elapsed = max(time.time() - self.start_time, 1e-6)
        return self.total_images / elapsed


    def average_latency(self):
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests


    def current_fid(self):
        return self.last_fid


    def total_images_generated(self):
        return self.total_images
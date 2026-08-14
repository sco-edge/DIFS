# ================================================================
# Author: KB
# Updated: 2026
# Purpose:
# Central Metrics Manager for DIFS
#
# Backward compatible with previous Metrics class.
#

# =====================PART 1A=====================================

# Used by:
#   server.py
#   scheduler.py
#   autoscaler.py
#   worker_pool.py
#   worker containers
#
# ================================================================

import csv
import os
import threading
import time
import statistics
import json

from collections import deque

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True

except Exception:
    GPU_AVAILABLE = False


class Metrics:
    """
    Central metrics engine for DIFS.

    Used by:
        Server, Scheduler, WorkerPool, Autoscaler, ModelManager
    """

    ###############################################################
    # INITIALIZATION
    ###############################################################

    def __init__(self):
    
        self.lock = threading.RLock()

        ####################################################
        # Request metrics
        ####################################################

        self.request_count = 0
        self.completed_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_start_times = {}

        ####################################################
        # Latency metrics
        ####################################################

        self.latencies = deque(maxlen=10000)

        self.queue_delays = deque(maxlen=10000)
        self.queue_lengths = deque(maxlen=10000)
        self.batch_sizes = deque(maxlen=10000)
        self.arrival_rates = deque(maxlen=10000)
        self.service_rates = deque(maxlen=10000)
        self.service_times = deque(maxlen=10000)
        self.worker_counts = deque(maxlen=10000)
        self.active_requests_history = deque(maxlen=10000)

        ####################################################
        # Throughput
        ####################################################

        self.start_time = time.time()

        ####################################################
        # Resource metrics
        ####################################################

        self.cpu_history = deque(maxlen=10000)

        self.memory_history = deque(maxlen=10000)

        self.gpu_history = deque(maxlen=10000)

        if GPU_AVAILABLE:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            self.gpu_handle = None

        ####################################################
        # Model metrics
        ####################################################

        self.model_load_times = deque(maxlen=1000)

        self.model_switch_times = deque(maxlen=1000)

        self.model_switch_count = 0

        ####################################################
        # Autoscaling metrics
        ####################################################

        self.scaling_history = deque(maxlen=10000)

        self.scale_up_events = 0
        self.scale_down_events = 0
        self.sla_violations = 0

        self.scale_latencies = []

        self.autoscaler_log = []

        self.worker_history = []
        self.worker_busy_history=[]
        self.worker_idle_history=[]

        self.utilization_history = []

        ####################################################
        # Scheduler metrics
        ####################################################

        self.scheduler_events = []

        ####################################################
        # Worker metrics
        ####################################################

        self.worker_events = []
        self.worker_startup_times = deque(maxlen=1000)
        self.worker_startup_failures = 0
        self.worker_startup_successes = 0

        ####################################################
        # Initialize CSV files
        ####################################################

        self.initialize_csv_files()

    ###############################################################
    # CSV INITIALIZATION
    ###############################################################

    def initialize_csv_files(self):
    
        os.makedirs("metrics", exist_ok=True)

        files = {

            "metrics/server_metrics.csv":
            [
                "timestamp",
                "latency",
                "throughput",
                "queue_delay"
            ],

            "metrics/scheduler_metrics.csv":
            [
                "timestamp",
                "queue_length",
                "arrival_rate",
                "service_rate"
            ],

            "metrics/legacy_worker_metrics.csv":
            [
                "timestamp",
                "worker",
                "latency",
                "cpu",
                "memory",
                "gpu"
            ],

           "metrics/autoscaler_metrics.csv":
                
            [
                "timestamp",
                "event",
                "mode",
                "queue_length",
                "workers_before",
                "workers_after",
                "arrival_rate",
                "service_rate",
                "utilization",
                "scale_latency"
            ],

            "metrics/switch_metrics.csv":
            [
                "timestamp",
                "event",
                "old_model",
                "new_model",
                "switch_time",
                "reason"
            ],

            "metrics/gpu_metrics.csv":
            [
                "timestamp",
                "gpu_utilization"
            ],

            "metrics/cluster_metrics.csv":
            [
                "timestamp",
                "workers",
                "busy_workers",
                "idle_workers",
                "queue_length",
                "predicted_queue",
                "arrival_rate",
                "service_rate",
                "mode"
            ],

            "metrics/summary.csv":
            [
                "metric",
                "value"
            ]
        }

        for filename, header in files.items():

            if not os.path.exists(filename):

                with open(filename, "w", newline="") as f:

                    csv.writer(f).writerow(header)

    ###############################################################

    

    ###############################################################
    # BACKWARD COMPATIBLE FUNCTIONS
    ###############################################################

    def record_latency(self, latency):
    
        with self.lock:

            self.latencies.append(latency)

            self.completed_requests += 1

    ###############################################################

    def avg_latency(self):

        with self.lock:    
            if not self.latencies:
                return 0
            return statistics.mean(self.latencies)

    def p50_latency(self):    
        return self.latency_percentile(50)
    
    def p90_latency(self):    
        return self.latency_percentile(90)

    def p95_latency(self):    
        return self.latency_percentile(95)

    def p99_latency(self):
        return self.latency_percentile(99)

    ###############################################################

    def record_request(self):
    
        with self.lock:

            self.request_count += 1

    def throughput(self):
        
        elapsed = time.time() - self.start_time

        if elapsed <= 0:
            return 0

        with self.lock:
            completed = self.completed_requests

        return completed / elapsed


    
    def record_cpu(self):
    
        if psutil is None:
            return 0

        value = psutil.cpu_percent(interval=None)

        self.cpu_history.append(value)

        return value

    def cpu_usage(self):
        return self.record_cpu()



    def record_memory(self):
        if psutil is None:
            return 0
        else:
            value = psutil.virtual_memory().percent

        self.memory_history.append(value)

        return value

    def memory_usage(self):
        return self.record_memory()



    def record_gpu(self):
        
        if not GPU_AVAILABLE:
            return 0

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            value = util.gpu
        except Exception:
            value = 0

        self.gpu_history.append(value)

        return value

    def gpu_usage(self):
        return self.record_gpu()

    ###############################################################

    def record_success(self):

        with self.lock:

            self.successful_requests += 1

    ###############################################################

    def record_failure(self):

        with self.lock:

            self.failed_requests += 1

    ###############################################################

    def request_started(self, request_id):
    
        with self.lock:
            self.request_start_times[request_id] = time.time()

    ###############################################################

    def request_finished(self, request_id):
        
        with self.lock:
            if request_id not in self.request_start_times:
                return 0

            latency = time.time() - self.request_start_times.pop(request_id)

            self.record_latency(latency)

            return latency
        
    ###############################################################

    def success_rate(self):

        total = self.successful_requests + self.failed_requests

        if total == 0:
            return 0

        return self.successful_requests / total


# ================ PART 2=======================================================================================
# This section adds

# latency percentiles
# server logger
# scheduler logger
# worker logger
# autoscaler logger
# model loading logger
# model switching logger
# GPU logger
# CSV append utilities

# FOr:
# Percentile latency calculations (P50, P95, P99).
# Server, scheduler, worker, and autoscaler logging methods.
# Model loading and switching metrics.
# Automatic summary generation and CSV export, all while keeping compatibility with your existing DIFS codebase.
# ================================================================================================================


    ###############################################################
    # LATENCY STATISTICS
    ###############################################################

    def min_latency(self):
        with self.lock:
    
            if not self.latencies:
                return 0

            return min(self.latencies)

    ###############################################################

    def max_latency(self):
        
        with self.lock:
        
            if not self.latencies:
                return 0

            return max(self.latencies)

    ###############################################################

    def median_latency(self):

        if not self.latencies:
            return 0

        return statistics.median(self.latencies)

    ###############################################################

    def std_latency(self):

        if len(self.latencies) < 2:
            return 0

        return statistics.stdev(self.latencies)

    ###############################################################

    def latency_percentile(self, percentile):

        if not self.latencies:
            return 0

        return np.percentile(
            np.asarray(self.latencies),
            percentile
        )

    ###############################################################

    
    ###############################################################
    # QUEUE / BATCH STATISTICS
    ###############################################################

    def record_queue_length(self, queue_length):

        with self.lock:

            self.queue_lengths.append(queue_length)

    def record_queue_delay(self, delay):
    
        with self.lock:

            self.queue_delays.append(delay)

    def record_queue_prediction(self, prediction):
    
        with self.lock:

            if not hasattr(self, "queue_predictions"):
                self.queue_predictions = deque(maxlen=10000)

            self.queue_predictions.append(prediction)

    def avg_queue_delay(self):
    
        if len(self.queue_delays)==0:
            return 0

        return statistics.mean(
            self.queue_delays
        )

    def max_queue_delay(self):
    
        if len(self.queue_delays)==0:
            return 0

        return max(self.queue_delays)

    ###############################################################

    def record_batch_size(self, batch_size):

        with self.lock:

            self.batch_sizes.append(batch_size)

    ###############################################################

    def record_arrival_rate(self, arrival_rate):

        with self.lock:

            self.arrival_rates.append(arrival_rate)

    ###############################################################

    def record_service_rate(self, service_rate):

        with self.lock:

            self.service_rates.append(service_rate)

    def record_service_time(self, service_time):
    
        with self.lock:
            self.service_times.append(service_time)

    ###############################################################

    def record_worker_count(self, worker_count):

        with self.lock:

            self.worker_counts.append(worker_count)

    ###############################################################
    # SERVER LOGGER
    ###############################################################

    def log_server_metrics(        

            self,

            latency,

            throughput,

            queue_delay

    ):
        '''
        Then server.py simply becomes

        metrics.log_server_metrics(
            latency,
            throughput,
            queue_delay
        )

        instead of opening CSV files.
        '''

        self.append_csv(

            "metrics/server_metrics.csv",

            [

                time.time(),

                latency,

                throughput,

                queue_delay

            ]

        )                

    ###############################################################
    # SCHEDULER LOGGER
    ###############################################################

    def log_scheduler_metrics(

        self,

        queue_length,

        arrival_rate,

        service_rate

    ):
        '''
        Scheduler Logging
        '''

        self.append_csv(

            "metrics/scheduler_metrics.csv",

            [

                time.time(),

                queue_length,

                arrival_rate,

                service_rate

            ]

        )

    ###############################################################
    # WORKER LOGGER
    ###############################################################

    def log_worker_metrics(

            self,

            worker,

            latency,

            cpu,

            memory,

            gpu

    ):
        '''
        Worker Logger
        '''

        self.append_csv(

            "metrics/legacy_worker_metrics.csv",

            [

                time.time(),

                worker,

                latency,

                cpu,

                memory,

                gpu

            ]

        )
    ###############################################################
    # WORKER STARTUP LOGGER
    ###############################################################

    def log_worker_startup(

            self,

            startup_time,

            success

    ):

        with self.lock:

            if success:
                self.worker_startup_successes += 1
                self.worker_startup_times.append(
                    startup_time
                )

            else:
                self.worker_startup_failures += 1

    ###############################################################
    # AUTOSCALER LOGGER
    ###############################################################

    def log_autoscaler_event(
        self,
        event,
        mode,
        queue_length,
        workers_before,
        workers_after,
        arrival_rate,
        service_rate,
        utilization,
        scale_latency
    ):

        if event == "scale_up":
            self.scale_up_events += 1

        elif event == "scale_down":
            self.scale_down_events += 1

        self.scale_latencies.append(scale_latency)

        self.worker_history.append(workers_after)

        self.utilization_history.append(utilization)

        self.scaling_history.append({
            "time": time.time(),
            "mode": mode,
            "workers": workers_after,
            "queue": queue_length,
            "event": event
        })

        self.append_csv(

            "metrics/autoscaler_metrics.csv",

            [

                time.time(),
                event,
                mode,
                queue_length,
                workers_before,
                workers_after,
                arrival_rate,
                service_rate,
                utilization,
                scale_latency

            ]

        )

        
    ###############################################################
    # SCALE-DOWN SCORE LOGGER
    ###############################################################

    def record_worker_removed(self):

        pass

    def log_scale_down_score(
        self,
        worker,
        score
    ):
        """
        Records why a worker was selected for removal.
        """

        self.scheduler_events.append({
            "time": time.time(),
            "worker": worker,
            "score": score,
            "event": "scale_down_score"
        })

    ###############################################################
    # CLUSTER STATE LOGGER
    ###############################################################

    def log_cluster_state(
        self,
        workers,
        busy_workers,
        idle_workers,
        queue_length,
        predicted_queue,
        arrival_rate,
        service_rate,
        mode
    ):
        """
        Records the entire cluster state once every autoscaler cycle.
        """

        self.append_csv(

            "metrics/cluster_metrics.csv",

            [
                time.time(),
                workers,
                busy_workers,
                idle_workers,
                queue_length,
                predicted_queue,
                arrival_rate,
                service_rate,
                mode
            ]

        )

    ###############################################################
    # GPU LOGGER
    ###############################################################

    def log_gpu_metrics(

        self,

        utilization

    ):
        
        self.append_csv(

            "metrics/gpu_metrics.csv",

            [

                time.time(),

                utilization

            ]

        )


    ###############################################################
    # MODEL LOADING AND SWITCHING LOGGERS
    ###############################################################

    def log_model_load(
            self,
            model,
            load_time
    ):
        '''
        Example of use:
            
        metrics.log_model_load(
            "sd-v1-4",
            elapsed
        )
        '''

        self.model_load_times.append(load_time)

        self.append_csv(

            "metrics/switch_metrics.csv",

            [
                time.time(),
                "LOAD",
                "",
                model,
                load_time,
                "startup"
            ]

        )

    def log_model_switch(

            self,
            old_model,
            new_model,
            switch_time
            ):
            
            self.model_switch_count += 1

            self.model_switch_times.append(

                switch_time

            )

            self.append_csv(

                "metrics/switch_metrics.csv",

                [
                    time.time(),
                    "SWITCH",
                    old_model,
                    new_model,
                    switch_time,
                    "autoscaler"
                ]

            )

    ########################################################################################
    # TO PREVENT DUPLICATED CSV-WRITING CODE EVERYWHERE; WILL REPLACE ALL "with open(...)"
    #######################################################################################
    def append_csv(
            self,
            filename,
            row
    ):

        with self.lock:

            with open(
                filename,
                "a",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow(row)                

    ###############################################################
    # SIMPLE GETTERS
    ###############################################################

    def average_queue_length(self):

        with self.lock:
        
            if not self.queue_lengths:
                return 0

            return np.mean(self.queue_lengths)

    ###############################################################

    def average_batch_size(self):

        if len(self.batch_sizes) == 0:
            return 0

        return np.mean(self.batch_sizes)

    ###############################################################

    def average_arrival_rate(self):

        if len(self.arrival_rates) == 0:
            return 0

        return np.mean(self.arrival_rates)

    ###############################################################

    def average_service_rate(self):

        if len(self.service_rates) == 0:
            return 0

        return np.mean(self.service_rates)

    ###############################################################

    def average_worker_count(self):

        if len(self.worker_counts) == 0:
            return 0

        return np.mean(self.worker_counts)

    def worker_utilization(
        self,
        active_workers,
        busy_workers
    ):

        if active_workers == 0:
            return 0

        return busy_workers / active_workers

    def record_worker_busy(self, busy):    
        with self.lock:
            self.worker_busy_history.append(busy)

    def record_worker_idle(self, idle):
        with self.lock:
            self.worker_idle_history.append(idle)

    def average_busy_workers(self):
    
        if len(self.worker_busy_history) == 0:
            return 0

        return statistics.mean(self.worker_busy_history)

    def average_idle_workers(self):

        if len(self.worker_idle_history) == 0:
            return 0

        return statistics.mean(self.worker_idle_history)


    ###############################################################
    # OTHER METHODS
    ###############################################################
    def avg_model_load(self):
        
        if len(self.model_load_times)==0:
            return 0

        return statistics.mean(
            self.model_load_times
        )

    def avg_switch_time(self):
    
        if len(self.model_switch_times)==0:
            return 0

        return statistics.mean(
            self.model_switch_times
        )

    def average_scaling_latency(self):
    
        if len(self.scale_latencies) == 0:
            return 0

        return statistics.mean(self.scale_latencies)

    def avg_cpu(self):
    
        with self.lock:
        
            if not self.cpu_history:
                return 0

            return statistics.mean(
                self.cpu_history
            )

    def avg_memory(self):
    
        with self.lock:
        
            if not self.memory_history:
                return 0

            return statistics.mean(
                self.memory_history
            )

    def avg_worker_startup_time(self):
    
        if len(self.worker_startup_times) == 0:

            return 0

        return statistics.mean(

            self.worker_startup_times

        )

    def avg_gpu(self):
    
        with self.lock:
        
            if not self.gpu_history:
                return 0

            return statistics.mean(
                self.gpu_history
            )

    def snapshot(self):
    
        return {

            "requests": self.request_count,
            "completed": self.completed_requests,
            "throughput": self.throughput(),
            "avg_latency": self.avg_latency(),
            "p95_latency": self.p95_latency(),
            "p99_latency": self.p99_latency(),
            "avg_queue_delay": self.avg_queue_delay(),
            "avg_cpu": self.avg_cpu(),
            "avg_memory": self.avg_memory(),
            "avg_gpu": self.avg_gpu(),
            "avg_switch_time": self.avg_switch_time(),
            "avg_scaling_latency": self.average_scaling_latency(),
            "success_rate": self.success_rate(),
            "min_latency": self.min_latency(),
            "max_latency": self.max_latency(),
            "median_latency": self.median_latency(),
            "std_latency": self.std_latency(),
            "avg_queue_length": self.average_queue_length(),
            "avg_batch_size": self.average_batch_size(),
            "avg_arrival_rate": self.average_arrival_rate(),
            "avg_service_rate": self.average_service_rate(),
            "avg_workers": self.average_worker_count(),
            "scale_up_events": self.scale_up_events,
            "scale_down_events": self.scale_down_events,
            "avg_model_load": self.avg_model_load(),
            "avg_worker_startup_time": self.avg_worker_startup_time(),
            "worker_startup_successes": self.worker_startup_successes,
            "worker_startup_failures": self.worker_startup_failures,
            "avg_busy_workers": self.average_busy_workers(),
            "avg_idle_workers": self.average_idle_workers(),
        }

    def print_snapshot(self):
    
        s = self.snapshot()

        print("="*60)

        for k,v in s.items():

            print(f"{k:20}: {v}")

        print("="*60)

    def generate_summary(self):
        return self.snapshot()

    def export_json(self, filename="metrics/report.json"):
        os.makedirs("metrics", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:

            json.dump(

                self.snapshot(),

                f,

                indent=4

            )

    def export_summary(self):
        self.write_summary()
        self.export_json()

    def reset(self):
        '''
        Reset to prevent multiple experiments from contaminating each other
        '''
        self.latencies.clear()
        self.queue_lengths.clear()
        self.queue_delays.clear()
        self.batch_sizes.clear()
        self.arrival_rates.clear()
        self.service_rates.clear()
        self.worker_counts.clear()
        self.cpu_history.clear()
        self.memory_history.clear()
        self.gpu_history.clear()
        self.model_load_times.clear()
        self.model_switch_times.clear()
        self.scale_latencies.clear()
        self.scaling_history.clear()
        self.worker_history.clear()
        self.utilization_history.clear()
        self.worker_busy_history.clear()
        self.worker_idle_history.clear()
        self.request_start_times.clear()
        self.active_requests_history.clear()
        self.start_time = time.time()
        self.worker_startup_times.clear()

        self.request_count = 0
        self.completed_requests = 0 # completed_requests = successful_requests + failures
        self.successful_requests = 0
        self.failed_requests = 0
        self.model_switch_count = 0
        self.scale_up_events = 0
        self.scale_down_events = 0
        self.worker_startup_successes = 0
        self.worker_startup_failures = 0
        

        

    def write_summary(self):
        
        with open("metrics/summary.csv","w",newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            snap = self.snapshot()

            for k, v in snap.items():

                writer.writerow([k, v])

    def record_scaling_mode(
        self,
        mode,
        workers
    ):
        '''
        scaling_history is only updated when a scaling event happens. Suppose horizontal
        scaling never happens, there will be no history. This is to allow the autoscaler
        every cycle to address that.
        '''

        self.scaling_history.append({

            "time": time.time(),

            "mode": mode,

            "workers": workers

        })

    def experiment_summary(self):
    
        return {

            "requests": self.request_count,

            "throughput": self.throughput(),
            "latency_avg": self.avg_latency(),
            "latency_p95": self.p95_latency(),
            "latency_p99": self.p99_latency(),
            "avg_cpu": self.avg_cpu(),
            "avg_gpu": self.avg_gpu(),
            "avg_switch": self.avg_switch_time(),
            "scale_up": self.scale_up_events,
            "scale_down": self.scale_down_events
        }

    def record_error(self):
        with self.lock:
            self.failed_requests += 1

    def record_active_requests(self, active):    
        with self.lock:
            self.active_requests_history.append(active)

    def record_completed_worker(self, worker):
    
        with self.lock:

            self.worker_events.append({
                "time": time.time(),
                "worker": worker,
                "event": "completed"
            })

    ############################################################
    # REQUEST COMPLETION
    ############################################################

    def record_request_completion(
            self,
            latency):
        '''
        This is a convenience wrapper. Instead of calling 
        record_latency() and record_completed_requests() everywhere, the worker
        can simply call this function
        '''

        self.record_latency(latency)

        self.record_completed_requests(1)

    ###############################################################
    # SLA VIOLATIONS
    ###############################################################

    def record_sla_violations(self, count):

        with self.lock:

            if not hasattr(self, "sla_violations"):
                self.sla_violations = 0

            self.sla_violations += count

    def close(self):
        pass

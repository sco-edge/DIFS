# Author: KB 
# Date: 2026-03-06 
# Purpose: For adding scaling capability 
# File: worker_pool.py 

import subprocess
import time
import os

import config

# 2026.05.15
import asyncio
import grpc
import query_pb2
import query_pb2_grpc
import traceback

from model_registry import MODEL_CONFIGS, get_model_filename, get_model_short_name, ALL_MODELS
from project_paths import MODEL_CACHE
from worker_pool_metrics import WorkerPoolMetrics


class WorkerPool:

    def __init__(self, metrics=None):

        self.metrics = metrics

        self.pool_metrics = WorkerPoolMetrics() # WorkerPool metrics
        self.workers = []
        self.base_port = 50060
        self.next_port = self.base_port

        ############################################################
        # Prevent concurrent worker creation
        ############################################################
        self.worker_lock = asyncio.Lock()

        ############################################################
        # INFaaS-style model replica tracking
        ############################################################
        self.model_worker_counts = {}  # model_name -> worker count (set by autoscaler)


    ############################################################
    # WAIT UNTIL WORKER IS FULLY READY
    ############################################################

    async def wait_until_worker_ready(self, addr, timeout=600): # PNB: increase to 600 because stable diffusion loads can easily exceed 300 s
        '''
        Health wait function; delays or blocks worker or pool manager until specific health criteria is met
        '''

        start = time.time()

        while time.time() - start < timeout:

            try:
                async with grpc.aio.insecure_channel(addr) as channel:

                    stub = query_pb2_grpc.QueryStub(channel)

                    response = await stub.Heartbeat(
                        query_pb2.HeartbeatRequest()
                    )

                    if response.status.status == 1:
                        print(f"[WORKER_POOL] Worker READY: {addr}")
                        return True

            except Exception as e:
    
                print("=" * 70)
                print("[WORKER_POOL HEALTHCHECK FAILED]")
                print("Worker :", addr)
                print("Elapsed:", round(time.time() - start, 1), "sec")
                print(type(e).__name__)
                print(e)
                print("=" * 70)

            print(f"[WORKER_POOL] Waiting for worker {addr}...")
            await asyncio.sleep(2)

        print(f"[WORKER_POOL ERROR] Worker timeout: {addr}")
        return False

    def size(self):
        return len(self.workers)

    async def add_worker(self, workload_type="balanced", model_id=None, sample_id=1, thread_count=10):
        async with self.worker_lock:
            startup_begin = time.perf_counter()
            # -----------------------------------------
            # Dynamic model selection
            # -----------------------------------------
            if model_id is not None:
                model_cfg = ALL_MODELS.get(model_id, {})
                model_id_str = str(model_id)
                sampler = str(model_cfg.get("sampler", 2))
                threads = str(model_cfg.get("threads", 10))
            else:
                config = MODEL_CONFIGS[workload_type]
                model_id_str = str(config["model_id"])
                sampler = str(config["sampler"])
                threads = str(config["threads"])

            print(f"[WORKER_POOL] Launching workload type: {workload_type}")
            print(f"[WORKER_POOL] Model={model_id_str}, Sampler={sampler}")
            

            #port = self.base_port + len(self.workers)# new port generated for new worker being added
            
            port = self.next_port
            print("="*60)
            print("HOST PORT =", port)
            print("CONTAINER PORT = 50060")
            print("="*60)
            self.next_port += 1

            # worker = InferenceWorker()
            #  worker.start()
            print(f"Execution Started In Directory: {os.getcwd()}")

            MODEL_ROOT = os.environ.get(
                "DIFS_MODEL_DIR",
                "/tmp/model"
            )

            cmd = [
                "docker", "run", "-d",
                "--gpus", "all",

                #-------------------------------------
                # Docker directory
                #-------------------------------------
                "-v", f"{os.getcwd()}:/workspace", # -v mounts my project in container
                "-w", "/workspace", # -w sets working directory

                #-------------------------------------
                # Port mapping
                #-------------------------------------
                # "-p", f"{port}:{port}",
                "-p", f"{port}:50060",

                #-------------------------------------
                # Volume mounts
                #-------------------------------------
                #"-v", f"{MODEL_CACHE}:{MODEL_CACHE}",
                "-v", f"{str(MODEL_CACHE)}:{str(MODEL_CACHE)}",
                "-e", f"DIFS_MODEL_DIR={MODEL_CACHE}",
                #"-e", f"DIFS_MODEL_DIR={MODEL_DIR}",
                "-v", "/tmp/diffusion_output:/tmp/diffusion_output",

                #-------------------------------------
                # Image name
                #-------------------------------------

                # Use an EXISTING base image
                "difs-diffusion-worker",

                #"pytorch-diffusion-server",
                "python", "src/1/pytorch_container_diffusion_v3.py",

                #-------------------------------------
                # Arguments passed to python server
                #-------------------------------------
                # "-model", "4",
                # #"-model", "/tmp/model/sd-v1-4.saftetensors", # explicitly pass full model path
                # "-sampler", "2",
                # "-thread", "10",
                "-model", model_id_str,
                "-sampler", sampler,
                "-thread", threads,
                
                #"-port", "50060",
                "-port", str(port),
                #"-npz", f"{MODEL_CACHE}/real_stats.npz"
                "-npz", f"{str(MODEL_CACHE)}/real_stats.npz"
            ]
            
            print("\n================ DOCKER COMMAND ================")
            for i, item in enumerate(cmd):
                print(f"{i:02d}: {item}")
            print("===============================================\n")
            # Handle Docker failure gracefully
            try:                
                # Adding debugging visibility (2026.04.23)
                result = subprocess.run(cmd, capture_output=True, text=True)

                print("=" * 60)
                print("[DOCKER STDERR]")
                print(result.stderr)
                print("=" * 60)

                print("=" * 60)
                print("[DOCKER STDOUT]")
                print(result.stdout)
                print("=" * 60)

                if result.returncode != 0:
                    print("[WORKER ERROR]")
                    print(result.stderr)
                    print(result.stdout)
                    return

                container_id = result.stdout.strip()
                print(subprocess.check_output(
                    ["docker","port",container_id],
                    text=True
                ))
                print(f"[WORKER_POOL] Container ID: {container_id}")

            except subprocess.CalledProcessError as e:
                print(f"[WORKER_POOL ERROR] Failed to start container: {e}")
                subprocess.call(["docker", "logs", container_id])
                return

            print(f"[WORKER_POOL] Waiting for worker on port {port}...")

            # ---------------------------------------------------
            # DEBUG: show worker logs during startup
            # ---------------------------------------------------
            print("\n============================================")
            print("[WORKER_POOL] Waiting for worker startup...")
            print("============================================")

            for i in range(15):

                time.sleep(2)

                print(f"\n------ Docker Logs ({2*(i+1)} sec) ------")

                result = subprocess.run(
                    [
                        "docker",
                        "logs",
                        "--tail",
                        "30",
                        container_id
                    ],
                    capture_output=True,
                    text=True
                )

                print(result.stdout)

                if result.stderr:
                    print(result.stderr)

                print("----------------------------------------")


            # PNB Instead of using sleep to wait, run only if worker is healthy or passes health check test # (2026.06.15)
            worker_addr = f"localhost:{port}"   

            ready = await self.wait_until_worker_ready(worker_addr)
            
            startup_time = time.perf_counter() - startup_begin

            if not ready:
                if self.metrics:
                    self.metrics.log_worker_startup(
                        startup_time,
                        False
                    )
                print("[WORKER_POOL ERROR] Worker failed startup")
                subprocess.call(["docker", "logs", container_id])
                subprocess.call(["docker", "rm", "-f", container_id])
                return

            print(f"[WORKER_POOL] Worker READY: {worker_addr}")

            self.workers.append({
                "container_id": container_id,
                "port": port,
                "workload_type": workload_type,
                "model_id": model_id_str,
                "sampler": sampler,
                "threads": threads,
                "created_time": time.time()
            })

            ############################################################
            # Log worker creation
            ############################################################

            self.pool_metrics.worker_created(
                worker_port=port,
                worker_id=container_id[:8]
            )

            print(f"[WORKER_POOL] Started container {container_id[:8]} on port {port}")

            startup_time = time.perf_counter() - startup_begin

            if self.metrics:

                self.metrics.log_worker_startup(
                    startup_time,
                    True
                )

                self.metrics.record_worker_count(
                    len(self.workers)
                )

                self.metrics.log_model_load(
                    model=model_id_str,
                    load_time=startup_time
                )          

    async def remove_worker(self, worker=None):

        async with self.worker_lock:    
            if worker is None:
                if not self.workers:
                    return
                worker = self.workers.pop()
            else:
                if worker in self.workers:
                    self.workers.remove(worker)

            container_id = worker["container_id"]
            port = worker["port"]

            subprocess.call([
                "docker",
                "stop",
                container_id
            ])

            subprocess.call([
                "docker",
                "rm",
                "-f",
                container_id
            ])

            self.pool_metrics.worker_removed(  # Worker removal metric
                worker_port=port,
                worker_id=container_id[:8]
            )

            if self.metrics:
                self.metrics.record_worker_count(len(self.workers))

            print(f"[WORKER_POOL] Removed worker {container_id[:8]} port={port}")


    def select_worker(self):
    
        if len(self.workers) == 0:
            return None
        return self.workers[0]

    def get_worker_ports(self):

        #return [port for _, port in self.workers]
        return [w["port"] for w in self.workers]

    def get_worker_model_id(self, worker_addr):
        """Return the integer model_id for the worker at *worker_addr*.

        Falls back to 3 (SD-v1-5) when the address is not found or the
        record lacks a model_id — matching the legacy default so that
        behaviour is unchanged for callers that do not pass a worker address.
        """
        for w in self.workers:
            if f"localhost:{w['port']}" == worker_addr:
                try:
                    return int(w.get("model_id", 3))
                except (TypeError, ValueError):
                    return 3
        return 3

    def get_workers(self):
        """
        Return worker metadata for autoscaler.
        """
        return self.workers

    def get_worker_age(self, worker):
        """
        Return worker age in seconds.
        """
        return time.time() - worker["created_time"]

    def get_worker_runtime(self, worker):
        """
        Query runtime statistics from a worker.
        """

        try:
            with grpc.insecure_channel(f"localhost:{worker['port']}") as channel:
                stub = query_pb2_grpc.QueryStub(channel)
                response = stub.WorkerStatus(
                    query_pb2.WorkerStatusRequest(),
                    timeout=2
                )

                ############################################################
                # Log runtime
                ############################################################

                self.pool_metrics.worker_runtime(
                    worker_port=worker["port"],
                    worker_id=worker["container_id"][:8],
                    model_name=response.model_name,
                    busy=response.busy,
                    active_requests=response.active_requests,
                    gpu_memory=response.gpu_memory_mb,
                    throughput=response.throughput,
                    latency=response.average_latency_ms,
                    uptime=time.time() - worker["created_time"]
                )

                return response

        except Exception:

            return None

    def get_next_worker(self, workload_type="balanced", model_id=None):
        # FIFO mode (Clipper/INFaaS baselines): simple round-robin, no scoring.
        if config.DISPATCH_POLICY == "fifo":
            if not self.workers:
                return None
            if not hasattr(self, "_fifo_idx"):
                self._fifo_idx = 0
            workers = self.workers
            w = workers[self._fifo_idx % len(workers)]
            self._fifo_idx += 1
            return f"localhost:{w['port']}"

        if not self.workers:
            print("====================================")
            print("Registered workers:")
            for w in self.workers:
                print(w)
            print("====================================")

            print("[WORKER_POOL] No workers running.")
            return None
        
        candidate_workers = [worker for worker in self.workers if worker["workload_type"] == workload_type ]  # build a list of candidate workers

        if len(candidate_workers) == 0: # auto-create missing workers
    
            print(f"[WORKER_POOL] No existing" f"{workload_type} worker.")
            asyncio.run(self.add_worker(workload_type))
            candidate_workers = [worker for worker in self.workers if worker["workload_type"] == workload_type ]

            if len(candidate_workers) == 0:
                raise RuntimeError(f"Unable to create worker for workload '{workload_type}'")

        if not hasattr(self, "idx"):
            self.idx = 0

        best_worker = None
        best_score = float("inf")

        for worker in candidate_workers:
            status = self.get_worker_runtime(worker)

            if status is None:
                score = 10000
            else:
                score = (
                    status.active_requests * 1000
                    + status.average_latency_ms
                    + status.gpu_memory_mb * 0.05
                )
                if status.busy:
                    score += 1000
                if worker["workload_type"] != workload_type:
                    score += 5000
                # PaniniServe seam 3 (DIFS v3, default OFF): locality hint.
                # The scheduler passes model_id only when PANINI_LOCALITY is
                # on; a worker already running that model is preferred
                # (front-block preference) so the request avoids a switch.
                if model_id is not None and str(worker.get("model_id")) == str(model_id):
                    score -= 2000

            if score < best_score:
                best_score = score
                best_worker = worker

        if best_worker is None:
            # Fallback
            best_worker = candidate_workers[self.idx % len(candidate_workers)]

        print("[WORKER_POOL] Selected", best_worker["workload_type"], "worker", best_worker["port"],)

        self.idx += 1

        return f"localhost:{best_worker['port']}"

    async def switch_model(self, worker_addr, model_id):

        start = time.perf_counter()

        model_filename = get_model_filename(model_id)

        async with grpc.aio.insecure_channel(worker_addr) as channel:
            
            stub = query_pb2_grpc.QueryStub(channel)

            response = await stub.LoadModel(
                query_pb2.LoadModelRequest(
                    model_id=model_id,
                    model_name=model_filename
                )
            )
        end = time.perf_counter()
        return end-start

    ############################################################
    # INFaaS-STYLE MODEL REPLICATION
    ############################################################

    def get_model_id_from_name(self, model_filename):
        """
        Resolve a model filename (e.g. 'sd-v1-4.safetensors') back to its numeric model_id.
        """
        for model_id, meta in ALL_MODELS.items():
            if meta["filename"] == model_filename:
                return model_id
        print(f"[WORKER_POOL WARNING] Unknown model filename: {model_filename}")
        return 1

    def get_workers_for_model(self, model_filename):
        """
        Return all workers that currently serve a given model.
        """
        return [
            w for w in self.workers
            if w.get("model_id") == model_filename
        ]

    def get_replica_count(self, model_filename):
        """
        Return the number of workers currently serving a given model.
        """
        return len(self.get_workers_for_model(model_filename))

    def get_all_model_replicas(self):
        """
        Return dict of model_filename -> worker count.
        """
        counts = {}
        for w in self.workers:
            mid = w.get("model_id", "unknown")
            counts[mid] = counts.get(mid, 0) + 1
        return counts

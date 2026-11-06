"""
process_manager.py

Starts and stops every component required for DIFS experiments.

Supports two worker launch modes:
  - "direct": runs worker as a Python subprocess (fast, no Docker rebuild)
  - "docker": uses the difs-diffusion-worker Docker image

Author: KB
"""

import subprocess
import signal
import socket
import sys
import time
import os

from .experiment_config import *

from model_registry import get_model_filename, ALL_MODELS
from project_paths import MODEL_CACHE, OUTPUT_DIR


class ProcessManager:

    def __init__(self, run_direct=None):

        self.worker_process = None
        self.server_process = None
        self.client_process = None

        if run_direct is None:
            self.run_direct = RUN_DIRECT
        else:
            self.run_direct = run_direct

        self.worker_port_counter = WORKER_PORT

        ############################################################
        # Multi-worker tracking (for multiworker benchmark)
        ############################################################
        self.worker_processes = {}   # port -> subprocess.Popen or container_id string

    ############################################################
    # Port Utilities
    ############################################################

    def is_port_open(self, host, port):

        sock = socket.socket()
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()

        return result == 0

    ############################################################
    # Wait for Port
    ############################################################

    def wait_for_port(
            self,
            host,
            port,
            timeout=120):

        print(f"[PROCESS] Waiting for {host}:{port}")
        start = time.time()

        while time.time() - start < timeout:

            if self.is_port_open(host, port):

                print(f"[PROCESS] Port {port} is ready")

                return True

            time.sleep(1)

        raise RuntimeError(
            f"Timeout waiting for port {port}"
        )

    ############################################################
    # Worker (Direct Python Subprocess Mode)
    ############################################################

    def start_worker_direct(self, model_id=None, port=None, sampler=1, threads=4):

        if self.worker_process is not None:

            print("[PROCESS] Worker already running")

            return

        if model_id is None:
            model_id = MODEL_ID

        if port is None:
            port = self.worker_port_counter
            self.worker_port_counter += 1

        model_filename = get_model_filename(model_id)

        cmd = [
            sys.executable,
            str(WORKER_SCRIPT),
            "-model", str(model_id),
            "-sampler", str(sampler),
            "-thread", str(threads),
            "-port", str(port),
            "-npz", str(PROJECT_ROOT / "real_stats.npz")
        ]

        print(f"[PROCESS] Starting worker (direct) - model={model_filename}, port={port}")

        env = os.environ.copy()
        env["DIFS_MODEL_DIR"] = str(MODEL_CACHE)
        env["DIFS_OUTPUT_DIR"] = str(OUTPUT_DIR)

        self.worker_process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env
        )

        self.wait_for_port(
            SCHEDULER_HOST,
            port,
            timeout=300
        )

        print(f"[PROCESS] Worker started on port {port}")

        return port

    ############################################################
    # Worker (Docker Mode)
    ############################################################

    def start_worker_docker(self, model_id=None, port=None, workload_type="balanced", sampler=1, threads=10):

        if self.worker_process is not None:

            print("[PROCESS] Worker already running (docker)")

            return

        if model_id is None:
            model_id = MODEL_ID

        if port is None:
            port = self.worker_port_counter
            self.worker_port_counter += 1

        MODEL_ROOT = os.environ.get("DIFS_MODEL_DIR", str(MODEL_CACHE))

        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",
            "-v", f"{PROJECT_ROOT}:/workspace",
            "-w", "/workspace",
            "-p", f"{port}:50060",
            "-v", f"{MODEL_ROOT}:{MODEL_ROOT}",
            "-e", f"DIFS_MODEL_DIR={MODEL_ROOT}",
            "-v", "/tmp/diffusion_output:/tmp/diffusion_output",
            "difs-diffusion-worker",
            "python", "src/1/pytorch_container_diffusion_v3.py",
            "-model", str(model_id),
            "-sampler", str(sampler),
            "-thread", str(threads),
            "-port", str(port),
            "-npz", f"{MODEL_ROOT}/real_stats.npz"
        ]

        print(f"[PROCESS] Starting worker (docker) - model={model_id}, port={port}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[PROCESS ERROR] Docker failed: {result.stderr}")
            return None

        container_id = result.stdout.strip()
        self.worker_process = container_id  # store for cleanup

        print(f"[PROCESS] Container {container_id[:8]} started on port {port}")

        time.sleep(10)

        self.wait_for_port(
            SCHEDULER_HOST,
            port,
            timeout=300
        )

        print(f"[PROCESS] Worker ready on port {port}")

        return port

    ############################################################
    # Worker (Unified Entry Point)
    ############################################################

    def start_worker(self, model_id=None, port=None, workload_type="balanced", sampler=None, threads=None):

        if self.run_direct:
            if sampler is None:
                cfg = ALL_MODELS.get(model_id or MODEL_ID, {})
                sampler = cfg.get("sampler", 1)
                threads = cfg.get("threads", 4)
            return self.start_worker_direct(model_id, port, sampler, threads)
        else:
            if sampler is None:
                cfg = ALL_MODELS.get(model_id or MODEL_ID, {})
                sampler = cfg.get("sampler", 1)
                threads = cfg.get("threads", 10)
            return self.start_worker_docker(model_id, port, workload_type, sampler, threads)

    ############################################################
    # Stop Worker
    ############################################################

    def stop_worker(self):

        if self.worker_process is None:

            return

        print("[PROCESS] Stopping worker")

        if self.run_direct:
            # Direct mode: it's a subprocess
            if isinstance(self.worker_process, subprocess.Popen):
                if self.worker_process.poll() is None:
                    self.worker_process.terminate()
                try:
                    self.worker_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.worker_process.kill()
                    self.worker_process.wait()
        else:
            # Docker mode: it's a container ID string
            container_id = self.worker_process
            subprocess.call(["docker", "stop", container_id])
            subprocess.call(["docker", "rm", "-f", container_id])

        self.worker_process = None
        print("[PROCESS] Worker stopped")


    ############################################################
    # SERVER
    ############################################################

    def start_server(self):

        if self.server_process is not None:

            print("[PROCESS] Server already running")

            return

        print("[PROCESS] Starting scheduler/server")

        env = os.environ.copy()
        env["DIFS_NO_AUTOSCALER"] = "1"

        self.server_process = subprocess.Popen(SERVER_COMMAND, cwd=PROJECT_ROOT, env=env)

        self.wait_for_port("localhost", SERVER_PORT)
        print("[PROCESS] Scheduler ready")

    def stop_server(self):
    
        if self.server_process is None:

            return

        print("[PROCESS] Stopping scheduler")

        self.server_process.terminate()

        try:
            self.server_process.wait(timeout=5)

        except subprocess.TimeoutExpired:
            print("[PROCESS] Scheduler did not exit, killing...")
            self.server_process.kill()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        self.server_process = None
        print("[PROCESS] Scheduler stopped")

    ############################################################
    # CLIENT
    ############################################################

    def run_client(self, workload, sampler, steps, seed):
    
        cmd = [
            sys.executable,
            CLIENT_SCRIPT,
            "-prompt_count",
            str(workload),
            "-host",
            SCHEDULER_HOST,
            "-port",
            str(SCHEDULER_PORT),
            "-steps",
            str(steps),
            "-sampler",
            str(sampler),
            "-seed",
            str(seed)
        ]

        print(
            f"[CLIENT] Running workload={workload}, "
            f"sampler={sampler}, "
            f"steps={steps}, "
            f"seed={seed}"
        )
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        return result.returncode

    ############################################################
    # Workload Experiment
    ############################################################

    def run_workload_experiment(self, scaling_mode=DEFAULT_SCALING_MODE):

        print()
        print("=" * 70)
        print("Running workload experiment")
        print("=" * 70)

        self.start_worker()
        self.start_server()

        try:
            for workload in WORKLOADS:
                print()
                print(f"[EXPERIMENT] workload={workload}")

                self.run_client(
                    workload=workload,
                    sampler=DEFAULT_SAMPLER,
                    steps=DEFAULT_STEPS,
                    seed=SEED
                )

                time.sleep(POST_EXPERIMENT_WAIT)
                print(f"[EXPERIMENT] Completed workload={workload}")

        finally:
            self.stop_server()
            self.stop_worker()
        return True

    ############################################################
    # Multi-Worker Management (for multiworker benchmark)
    ############################################################

    def start_worker_at_port(self, model_id, port, sampler=None, threads=None):
        """
        Start a single worker on a specific port (multi-worker mode).
        Tracks in self.worker_processes dict.
        Returns the port on success, None on failure.
        """
        if port in self.worker_processes:
            print(f"[PROCESS] Worker already running on port {port}")
            return port

        if sampler is None:
            cfg = ALL_MODELS.get(model_id, {})
            sampler = cfg.get("sampler", 1)
            threads = cfg.get("threads", 4)

        model_filename = get_model_filename(model_id)

        if self.run_direct:
            cmd = [
                sys.executable,
                str(WORKER_SCRIPT),
                "-model", str(model_id),
                "-sampler", str(sampler),
                "-thread", str(threads),
                "-port", str(port),
                "-npz", str(PROJECT_ROOT / "real_stats.npz")
            ]

            print(f"[PROCESS] Starting worker at port {port} - model={model_filename}")

            env = os.environ.copy()
            env["DIFS_MODEL_DIR"] = str(MODEL_CACHE)
            env["DIFS_OUTPUT_DIR"] = str(OUTPUT_DIR)

            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env
            )

            self.worker_processes[port] = proc

            self.wait_for_port(SCHEDULER_HOST, port, timeout=300)

            print(f"[PROCESS] Worker ready on port {port} (pid={proc.pid})")
            return port

        # Docker mode: launch one container per worker port. The worker
        # binds [::]:<port> inside the container, so expose <port>:<port>.
        MODEL_ROOT = os.environ.get("DIFS_MODEL_DIR", str(MODEL_CACHE))

        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",
            "-v", f"{PROJECT_ROOT}:/workspace",
            "-w", "/workspace",
            "-p", f"{port}:{port}",
            "-v", f"{MODEL_ROOT}:{MODEL_ROOT}",
            "-e", f"DIFS_MODEL_DIR={MODEL_ROOT}",
            "-v", "/tmp/diffusion_output:/tmp/diffusion_output",
            "difs-diffusion-worker",
            "python", "src/1/pytorch_container_diffusion_v3.py",
            "-model", str(model_id),
            "-sampler", str(sampler),
            "-thread", str(threads),
            "-port", str(port),
            "-npz", f"{MODEL_ROOT}/real_stats.npz"
        ]

        print(f"[PROCESS] Starting worker (docker) at port {port} - model={model_filename}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[PROCESS ERROR] Docker failed for port {port}: {result.stderr}")
            return None

        container_id = result.stdout.strip()
        self.worker_processes[port] = container_id

        self.wait_for_port(SCHEDULER_HOST, port, timeout=300)

        print(f"[PROCESS] Container {container_id[:8]} ready on port {port}")
        return port

    def stop_worker_at_port(self, port):
        """
        Stop a specific worker by port.
        """
        proc = self.worker_processes.pop(port, None)
        if proc is None:
            return

        print(f"[PROCESS] Stopping worker on port {port}")

        if isinstance(proc, subprocess.Popen):
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            # Docker container ID
            subprocess.call(["docker", "stop", proc])
            subprocess.call(["docker", "rm", "-f", proc])

        print(f"[PROCESS] Worker on port {port} stopped")

    def stop_all_multi_workers(self):
        """
        Stop all workers tracked in worker_processes.
        """
        ports = list(self.worker_processes.keys())
        for port in ports:
            self.stop_worker_at_port(port)

    def emergency_stop_all(self):
        """
        Nuclear cleanup: stop server + every tracked worker, then sweep
        for any remaining container_diffusion orphans.  Safe to call
        multiple times and during exceptions.
        """
        import signal as _sig

        print("[PROCESS] EMERGENCY STOP: cleaning up all processes...")

        # 1. Server
        try:
            self.stop_server()
        except Exception:
            pass

        # 2. All tracked multi-workers
        try:
            self.stop_all_multi_workers()
        except Exception:
            pass

        # 3. Single worker
        try:
            self.stop_worker()
        except Exception:
            pass

        # 4. Sweep for orphaned container_diffusion processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", "container_diffusio[n]"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid), _sig.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(2)
                result2 = subprocess.run(
                    ["pgrep", "-f", "container_diffusio[n]"],
                    capture_output=True, text=True, timeout=5,
                )
                if result2.stdout.strip():
                    for pid in result2.stdout.strip().split("\n"):
                        try:
                            os.kill(int(pid), _sig.SIGKILL)
                        except (ProcessLookupError, ValueError):
                            pass
        except Exception:
            pass

        # 5. Kill anything bound to SERVER_PORT
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{SERVER_PORT}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid), _sig.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass

        print("[PROCESS] EMERGENCY STOP complete.")

    def get_worker_ports(self):
        """
        Return list of ports for all multi-worker tracked processes.
        """
        return list(self.worker_processes.keys())

    def start_server_multi(self, extra_worker_ports=None, worker_model_map=None):
        """
        Start the server with DIFS_EXTRA_WORKER_PORTS env var.
        extra_worker_ports: list of int ports to pass to server.
        worker_model_map: dict mapping port (int) -> model_id (int), passed
            via DIFS_WORKER_MODEL_MAP so the server registers each external
            worker with its actual model_id (required for correct per-request
            latency SLO computation).
        """
        if self.server_process is not None:
            print("[PROCESS] Server already running")
            return

        print("[PROCESS] Starting scheduler/server (multi-worker mode)")

        env = os.environ.copy()
        env["DIFS_NO_AUTOSCALER"] = "1"

        if extra_worker_ports:
            env["DIFS_EXTRA_WORKER_PORTS"] = ",".join(str(p) for p in extra_worker_ports)
            print(f"[PROCESS] Server will register workers on ports: {extra_worker_ports}")

        if worker_model_map:
            import json
            env["DIFS_WORKER_MODEL_MAP"] = json.dumps({str(k): v for k, v in worker_model_map.items()})
            print(f"[PROCESS] Worker model map: {worker_model_map}")

        self.server_process = subprocess.Popen(SERVER_COMMAND, cwd=PROJECT_ROOT, env=env)

        self.wait_for_port("localhost", SERVER_PORT)
        print("[PROCESS] Scheduler ready")
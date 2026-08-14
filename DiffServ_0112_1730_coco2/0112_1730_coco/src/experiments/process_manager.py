"""
process_manager.py

Starts and stops every component required for DIFS experiments.

Worker (Terminal 1: python <path>bootstrap.py)
Scheduler (Terminal 2: python <path>/server.py)
Client (later; Terminal 3: python <path>/client_async_v3.py)

Author: KB
"""

import subprocess
import signal
import socket
import sys
import time

from pathlib import Path

from .experiment_config import *


class ProcessManager:

    def __init__(self):

        self.worker_process = None
        self.server_process = None
        self.client_process = None

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
            timeout=60):

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
    # Worker
    ############################################################

    def start_worker(self):

        if self.worker_process is not None:

            print("[PROCESS] Worker already running")

            return

        cmd = [
            sys.executable,
            BOOTSTRAP_SCRIPT, # Note that this is set in experiment_config.py
            "-model",
            str(MODEL_ID),
            "-sampler",
            "1",
            "-thread",
            str(THREADS),
            "-port",
            str(WORKER_PORT),
            "-npz",
            "real_stats.npz"
        ]

        print("[PROCESS] Starting worker")

        self.worker_process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT
        )

        self.wait_for_port(
            SCHEDULER_HOST,
            WORKER_PORT
        )

        print("[PROCESS] Worker started")

    
    def stop_worker(self):

        if self.worker_process is None:

            return

        print("[PROCESS] Stopping worker")

        if self.worker_process.poll() is None:
            self.worker_process.terminate()

        try:
            self.worker_process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            print("[PROCESS] Worker did not exit, killing...")
            self.worker_process.kill()
            self.worker_process.wait()

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

        self.server_process = subprocess.Popen(SERVER_COMMAND, cwd=PROJECT_ROOT) # since server is long-running "process.Popen" rather than "process.run"

        self.wait_for_port("localhost", SERVER_PORT)
        print("[PROCESS] Scheduler ready")

    def stop_server(self):
    
        if self.server_process is None:

            return

        print("[PROCESS] Stopping scheduler")
        # ask it to terminate cleanly
        self.server_process.terminate()

        try:
            self.server_process.wait(timeout=5)

        except subprocess.TimeoutExpired:
            print("[PROCESS] Scheduler did not exit, killing...")
            self.server_process.kill()
            self.server_process.wait()

        self.server_process = None
        print("[PROCESS] Scheduler stopped")

    ############################################################
    # CLIENT
    ############################################################

    def run_client(self, workload, sampler, steps,seed):
    
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
            f"steps={steps}",
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
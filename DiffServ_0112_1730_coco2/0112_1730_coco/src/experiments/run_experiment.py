"""
run_experiment.py

Main experiment orchestrator.

Author: KB
"""

import time
import subprocess
import shutil
import json
from pathlib import Path

from .process_manager import ProcessManager
from .experiment_config import *


class ExperimentRunner:
    
    def __init__(self):

        self.pm = ProcessManager()

    def run_client(
            self,
            workload=DEFAULT_WORKLOAD,
            sampler=DEFAULT_SAMPLER,
            steps=DEFAULT_STEPS,
            seed=SEED
            ):

        cmd = [
            "python",
            CLIENT_SCRIPT,
            "-prompt_count",
            str(workload),
            "-host",
            SCHEDULER_HOST,
            "-port",
            str(SERVER_PORT),
            "-steps",
            str(steps),
            "-sampler",
            str(sampler),
            "-seed",
            str(seed)
        ]

        print()
        print("=" * 60)
        print("[RUNNER] Launching client")
        print("=" * 60)
        subprocess.run(cmd, check=True)

    def run(
        self,
        workload=DEFAULT_WORKLOAD,
        sampler=DEFAULT_SAMPLER,
        steps=DEFAULT_STEPS,
        seed=SEED
    ):

        print()
        print("=" * 70)
        print("Starting experiment")
        print("=" * 70)

        self.pm.start_worker()

        self.pm.start_server()

        ########################################################
        # Execute workload
        ########################################################

        self.run_client(workload, sampler, steps, seed)

        ########################################################
        # Wait for metrics to flush
        ########################################################

        time.sleep(POST_EXPERIMENT_WAIT)

        ########################################################
        # Save everything
        ########################################################

        run_dir = self.create_run_directory(workload)

        self.save_experiment_config(
            run_dir,
            workload,
            sampler,
            steps,
            DEFAULT_SCALING_MODE,
            seed
        )

        self.copy_metrics(run_dir)

        self.copy_images(run_dir)

        ########################################################
        # Shutdown
        ########################################################

        self.pm.stop_server()

        self.pm.stop_worker()

        print()
        print("=" * 70)
        print("Experiment completed")
        print("=" * 70)

    def create_run_directory(self, workload):
        '''
        CREATE THE FOLDERS FOR THE EXPERIMENTS:
            
        To create multiple directories as we'll be running multiple experiments
        '''
        
        run_dir = RESULTS_DIR / f"workload_{workload}"
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def copy_metrics(self, run_dir):
        
        metrics_out = run_dir / "metrics"
        metrics_out.mkdir(exist_ok=True)

        for file in METRICS_DIR.glob("*.csv"):

            shutil.copy(file, metrics_out / file.name)

    def copy_images(self, run_dir):
        
        image_dir = run_dir / "images"
        image_dir.mkdir(exist_ok=True)

        for image in CLIENT_DIR.glob("*.png"):

            shutil.copy( image, image_dir / image.name)

    def save_experiment_config(
            self,
            run_dir,
            workload,
            sampler,
            steps,
            scaling_mode,
            seed
    ):

        config = {
            "workload": workload,
            "sampler": sampler,
            "steps": steps,
            "scaling_mode": scaling_mode,
            "model": MODEL_ID,
            "threads": THREADS,
            "seed": SEED,
            "scheduler_port": SCHEDULER_PORT,
            "worker_port": WORKER_PORT
        }

        with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:

            json.dump(config, f, indent=4)


if __name__ == "__main__":

    runner = ExperimentRunner()
    runner.run()
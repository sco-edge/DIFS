"""
run_experiment.py

Main experiment orchestrator.

Author: KB
Updated: 2026-07 - Added model_id parameter support
"""

import time
import subprocess
import shutil
import json
from pathlib import Path

from .process_manager import ProcessManager
from .experiment_config import *

from model_registry import get_model_filename, get_model_short_name, ALL_MODELS


class ExperimentRunner:
    
    def __init__(self, run_direct=None):

        self.pm = ProcessManager(run_direct=run_direct)

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
        seed=SEED,
        model_id=None
    ):

        if model_id is None:
            model_id = MODEL_ID

        model_name = get_model_short_name(model_id)

        print()
        print("=" * 70)
        print(f"Starting experiment - Model: {model_name} (id={model_id})")
        print("=" * 70)

        self.pm.start_worker(model_id=model_id)

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

        run_dir = self.create_run_directory(workload, model_id)

        self.save_experiment_config(
            run_dir,
            workload,
            sampler,
            steps,
            DEFAULT_SCALING_MODE,
            seed,
            model_id
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

    def create_run_directory(self, workload, model_id=None):
        '''
        CREATE THE FOLDERS FOR THE EXPERIMENTS:
        '''
        
        if model_id is not None:
            model_name = get_model_short_name(model_id)
            run_dir = RESULTS_DIR / f"model_{model_name}" / f"workload_{workload}"
        else:
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
            seed,
            model_id=None
    ):

        cfg = {
            "workload": workload,
            "sampler": sampler,
            "steps": steps,
            "scaling_mode": scaling_mode,
            "model_id": model_id or MODEL_ID,
            "model_name": get_model_short_name(model_id) if model_id else "unknown",
            "threads": THREADS,
            "seed": seed,
            "scheduler_port": SCHEDULER_PORT,
            "worker_port": WORKER_PORT
        }

        with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:

            json.dump(cfg, f, indent=4)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=int, default=None, help="Model ID to use")
    parser.add_argument("-workload", type=int, default=DEFAULT_WORKLOAD)
    parser.add_argument("-sampler", type=int, default=DEFAULT_SAMPLER)
    parser.add_argument("-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("-seed", type=int, default=SEED)
    parser.add_argument("--docker", action="store_true", help="Use Docker mode")
    args = parser.parse_args()

    runner = ExperimentRunner(run_direct=not args.docker)
    runner.run(
        workload=args.workload,
        sampler=args.sampler,
        steps=args.steps,
        seed=args.seed,
        model_id=args.model
    )
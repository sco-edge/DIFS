# Purpose: Only a launcher that lunches or gRPC worker process which is handled by "pytorch_container_diffusion_v3.py"

import os
import sys
import runpy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

WORKER = os.path.join(
    CURRENT_DIR,
    "pytorch_container_diffusion_v3.py"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.chdir(SRC_DIR)

runpy.run_path(
    WORKER,
    run_name="__main__"
)
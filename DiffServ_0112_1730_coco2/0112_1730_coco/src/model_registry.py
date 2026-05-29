# Author: KB 
# Date: 2026-03-06 
# Purpose: To register different possible model requirements. Together with "workload_classifier.py", it helps determine which model to load based on workload.
# File:

MODEL_CONFIGS = {

    # -----------------------------------------
    # FAST MODEL
    # -----------------------------------------
    "fast": {
        "model_id": 1,
        "sampler": 1,
        "threads": 4,
    },

    # -----------------------------------------
    # BALANCED MODEL
    # -----------------------------------------
    "balanced": {
        "model_id": 4,
        "sampler": 2,
        "threads": 10,
    },

    # -----------------------------------------
    # HIGH QUALITY MODEL
    # -----------------------------------------
    "quality": {
        "model_id": 7,
        "sampler": 2,
        "threads": 20,
    }
}
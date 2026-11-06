# Author: KB 
# Date: 2026-03-06 
# Purpose: To classify different workload types 
# File: workload_classifier.py 

class WorkloadClassifier:

    def classify(self, request):

        try:
            prompt = " ".join(request.Prompt).lower()
        except Exception:
            prompt = ""

        # ----------------------------------------
        # FAST / LOW LATENCY
        # ----------------------------------------
        if len(prompt) < 20:
            return "fast"

        # ----------------------------------------
        # HIGH QUALITY
        # ----------------------------------------
        if "ultra realistic" in prompt:
            return "quality"

        if "8k" in prompt:
            return "quality"

        if "cinematic" in prompt:
            return "quality"

        # ----------------------------------------
        # DEFAULT
        # ----------------------------------------
        return "balanced"
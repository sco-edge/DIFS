"""
baseline_profiles.py

Hardcoded architecture profiles used to emulate the Clipper/INFaaS baselines
inside DIFS, plus the project's own default configuration.

WHY THIS FILE EXISTS
--------------------
The baseline behaviour used to be selected by several independent environment
variables (DIFS_DISPATCH_POLICY, DIFS_AUTOSCALER, DIFS_PANINI_ADMISSION,
DIFS_PANINI_LOCALITY, DIFS_STEP_MAP).  A leftover variable in a shell could
silently change the behaviour of a manual run, which made experiments
non-deterministic.

The values are now hardcoded here (single source of truth) and selected with a
single variable:

    DIFS_BASELINE_PROFILE=default|clipper_large|clipper_small|infaas|difs_v3

When the variable is unset the ``default`` profile is used, which reproduces
the project's original no-override defaults exactly (EDF, autoscaler on,
PaniniServe seams off, stock SLO step map).  The legacy per-toggle variables
are intentionally ignored.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

PROFILE_ENV_VAR = "DIFS_BASELINE_PROFILE"
DEFAULT_PROFILE = "default"

# ---------------------------------------------------------------------------
# Hardcoded profiles.  Do not edit these at runtime; change the file and
# re-run so that every manual run uses the same values.
# ---------------------------------------------------------------------------
PROFILES: Dict[str, Dict] = {
    # Project defaults (equivalent to the old no-env behaviour).
    "default": {
        "description": "DIFS defaults: EDF, autoscaler on, PaniniServe seams off, stock step map",
        "dispatch_policy": "edf",
        "autoscaler": "1",
        "panini_admission": "0",
        "panini_locality": "0",
        # None => keep slo.ACCURACY_STEP_MAP as defined in slo.py
        "step_map": None,
    },
    # ---- Clipper-style Large-Only emulation -------------------------------
    "clipper_large": {
        "description": "Clipper-style Large-Only emulation (within DIFS): FIFO, model 9 only, no switching, no adaptation",
        "dispatch_policy": "fifo",
        "model_ids": [9, 9, 9, 9, 9],
        "autoscaler": "0",
        "panini_admission": "0",
        "panini_locality": "0",
        "step_map": {"low": 10, "medium": 20, "high": 30, "very_high": 40},
        "extra_levels": {"very_high": 40},
    },
    # ---- Clipper-style Small-Only emulation -------------------------------
    "clipper_small": {
        "description": "Clipper-style Small-Only emulation (within DIFS): FIFO, models 3+4, switching, no adaptation",
        "dispatch_policy": "fifo",
        "model_ids": [3, 4, 3, 4, 3],
        "autoscaler": "0",
        "panini_admission": "0",
        "panini_locality": "0",
        "step_map": {"low": 10, "medium": 20, "high": 30, "very_high": 40},
        "extra_levels": {"very_high": 40},
    },
    # ---- INFaaS-style emulation -------------------------------------------
    "infaas": {
        "description": "INFaaS-style emulation (within DIFS): FIFO, autoscaler cost-based switching, no step adaptation",
        "dispatch_policy": "fifo",
        "model_ids": [3, 4, 5, 8, 9],
        "autoscaler": "1",
        "panini_admission": "0",
        "panini_locality": "0",
        "step_map": {"low": 10, "medium": 20, "high": 30, "very_high": 40},
        "extra_levels": {"very_high": 40},
    },
    # ---- DIFS_OC_v3 -------------------------------------------------------
    "difs_v3": {
        "description": "DIFS_OC_v3: EDF, locality-aware, fair-share step adaptation 10-40",
        "dispatch_policy": "edf",
        "model_ids": [3, 4, 5, 8, 9],
        "autoscaler": "1",
        "panini_admission": "1",
        "panini_locality": "1",
        "step_map": {"low": 10, "medium": 20, "high": 30, "very_high": 40},
        "extra_levels": {"very_high": 40},
    },
}

# Alias kept for callers that expect a mapping of baseline name -> config.
BASELINES = PROFILES

# Profiles that the comparison benchmark may execute (excludes "default").
RUNNABLE_BASELINES = ["clipper_large", "clipper_small", "infaas", "difs_v3"]


def profile_name() -> str:
    """Return the requested profile name (default when unset)."""
    return os.environ.get(PROFILE_ENV_VAR, DEFAULT_PROFILE) or DEFAULT_PROFILE


def get_profile(name: Optional[str] = None) -> Dict:
    """Return the hardcoded profile for *name*.

    Unknown names fall back to ``default`` so a bad environment value can
    never crash a manual run; the fallback is reported on stderr.
    """
    if not name:
        name = profile_name()
    if name not in PROFILES:
        import sys
        print(
            f"[baseline_profiles] WARNING: unknown profile '{name}'; "
            f"using '{DEFAULT_PROFILE}'",
            file=sys.stderr,
        )
        name = DEFAULT_PROFILE
    return PROFILES[name]


def active_profile() -> Dict:
    """Return the currently selected hardcoded profile."""
    return get_profile(profile_name())

#!/usr/bin/env python3
# Author: KB
# Date: 2026-08-10
# Purpose: DIFS v3 - PaniniServe merge seams (all config-gated, default OFF)
# File: panini_seams.py

"""
panini_seams.py - PaniniServe merge seams for DIFS v3.

Three seams are ported from Templates/DIFS-Trade-off-main/PaniniServe_scheduler.py
and wired into the DIFS scheduler behind config flags that default to OFF:

  Seam 1 (config.PANINI_ADMISSION_CONTROL)
    PaniniServe window scheduling shrinks each request's step count
    proportionally (fair-share) when the predicted total exceeds the window
    budget, never going below the accuracy-tier floor, then drops requests
    (latest deadline first) until the window fits.  DIFS dispatches requests
    immediately (no windows), so the same policy is applied per request:
    overload = (active_requests - PANINI_OVERLOAD_THRESHOLD) drives a
    fair-share step reduction, and a queueing-aware deadline check drops the
    request when even the reduced step count cannot meet its latency SLO.
    See Scheduler._panini_admission() in scheduler.py.

  Seam 2 (config.PANINI_GPU_MONITOR)
    GPUUtilMonitor: a background thread polls GPU utilization every 10 ms
    while a request executes and reports the average, exactly like the
    PaniniServe monitor.  Falls back gracefully (reports 0) when the NVIDIA
    Management Library is unavailable (e.g. CPU-only benchmarking).

  Seam 3 (config.PANINI_LOCALITY)
    order_batch(): PaniniServe orders one window's requests front/back/auto
    so the queue only ever pays for a single model switch.  DIFS dispatches
    one request at a time, so the seam becomes a locality hint: when enabled,
    the scheduler asks the worker pool for a worker already running the
    request's model (front-block preference) instead of blindly picking the
    least-loaded worker.  See WorkerPool.get_next_worker(model_id=...).

All seams default to OFF.  With the flags left False, DIFS v3 scheduling
behavior is identical to DIFS v2 (only this module is added).
"""

import threading
import time

# ---------------------------------------------------------------------------
# Seam 3 (order_batch) profile constants, ported verbatim from PaniniServe.
# Used only by order_batch()/total_window_time() when predicting timings in
# the deterministic replay harness; the live scheduler never calls them.
# ---------------------------------------------------------------------------

TIME_PER_STEP = {"small": 0.08093, "large": 0.39624}   # seconds per denoising step
BASE_OVERHEAD = {"small": 0.09463, "large": 0.27221}    # fixed per-request overhead, seconds
SWITCH_TIME = {("small", "large"): 2.30, ("large", "small"): 4.50}   # model-swap overhead, seconds (asymmetric)


def exec_time(model, steps):
    """Predicted wall-clock time (seconds) to run `model` for `steps` steps."""
    return BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]


def switch_cost(prev_model, next_model):
    """Predicted overhead (seconds) of swapping GPU-resident models.  Zero if
    they match, or if prev_model is None (very first request)."""
    if prev_model is None or prev_model == next_model:
        return 0.0
    return SWITCH_TIME[(prev_model, next_model)]


def total_window_time(queue, current_model, get_model, get_steps):
    """Sum of switch cost + exec time to run `queue` in order, starting from
    current_model.  get_model(r) / get_steps(r) extract the (possibly reduced)
    assigned model/steps for each request."""
    t, prev = 0.0, current_model
    for r in queue:
        m = get_model(r)
        t += switch_cost(prev, m)
        t += exec_time(m, get_steps(r))
        prev = m
    return t


def order_batch(batch, current_model, model_pref_attr="model_pref",
                deadline_attr="deadline", assignment=None):
    """
    Seam 3 - locality-aware batch ordering (PaniniServe order_batch port).

      1) Requests that already want the currently-loaded model go first,
         sorted by deadline (Earliest Deadline First).
      2) Requests that want the other model go next, also sorted by deadline
         (the whole queue only pays for ONE model switch, at the boundary).
      3) "auto" requests (model not yet decided) are inserted one at a time,
         most-urgent-deadline first, into the latest position that still lets
         them meet their own latency_slo -- inheriting whichever model is
         adjacent, so they never introduce an extra switch.

    Reads model preference / deadline via getattr() with safe defaults, so it
    works on DIFS proto requests (which reject attribute assignment).  Returns
    the ordered list; `assignment` (dict keyed by id(request)) is filled with
    each request's inherited model, and is also returned as a convenience.
    """
    models = {r for r in batch if getattr(r, model_pref_attr, "auto") != "auto"}
    # Infer the two candidate models from the batch (or fall back to the
    # small/large naming PaniniServe uses).
    known = sorted(m for m in ("small", "large") if m in models)
    if not known:
        other = None
    else:
        other = "large" if known[0] == "small" else "small"

    front = [r for r in batch if getattr(r, model_pref_attr, "auto") == current_model]
    back = [r for r in batch if getattr(r, model_pref_attr, "auto") == other]
    auto_reqs = [r for r in batch if getattr(r, model_pref_attr, "auto") == "auto"]

    def _dl(r):
        return getattr(r, deadline_attr, float("inf"))

    front.sort(key=_dl)
    back.sort(key=_dl)

    assignment = {} if assignment is None else assignment
    for r in front:
        assignment[id(r)] = current_model
    for r in back:
        assignment[id(r)] = other

    queue = front + back
    for req in sorted(auto_reqs, key=_dl):
        best_pos = None
        # Scan from the back of the queue forward, looking for the latest
        # (least disruptive) position where inserting still meets the SLO.
        for pos in range(len(queue), -1, -1):
            left = assignment[id(queue[pos - 1])] if pos > 0 else current_model
            right = assignment[id(queue[pos])] if pos < len(queue) else None
            assignment[id(req)] = left or right or current_model
            t = total_window_time(
                queue[:pos] + [req], current_model,
                get_model=lambda r: assignment[id(r)],
                get_steps=lambda r: getattr(r, "default_steps", getattr(r, "Steps", 30)),
            )
            if t <= _dl(req):
                best_pos = pos
                break
        if best_pos is None:
            best_pos = len(queue)
            assignment[id(req)] = assignment[id(queue[-1])] if queue else current_model
        queue = queue[:best_pos] + [req] + queue[best_pos:]
    return queue, assignment


def fair_share_step_reduction(steps, min_steps, overflow_units, time_per_step):
    """
    Seam 1 - fair-share step reduction (PaniniServe plan_schedule step 1).

    Split the overload (in "execution units") evenly and shrink the step count
    accordingly, never going below min_steps.  Returns the reduced step count.
    """
    step_reduction = int(overflow_units * time_per_step)
    return max(int(steps) - step_reduction, int(min_steps))


def slo_multiplier(steps):
    """Linearly interpolate the latency-SLO multiplier across the 10-40 step
    range: 2.0x at 10 steps, 3.0x at 40 steps (PaniniServe slo_multiplier)."""
    lo, hi = 10, 40
    m_lo, m_hi = 2.0, 3.0
    frac = (steps - lo) / (hi - lo) if hi > lo else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return m_lo + frac * (m_hi - m_lo)


# ---------------------------------------------------------------------------
# Seam 2 - GPU utilization monitor (PaniniServe GPUUtilMonitor port).
# ---------------------------------------------------------------------------

class GPUUtilMonitor:
    """Polls GPU utilization every 10 ms on a background thread while a request
    executes and reports the average over the window.

    Degrades gracefully when pynvml is unavailable or no GPU is present: the
    monitor starts/stops normally and stop() returns 0 (CPU-only runs).
    """

    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.utils = []
        self.stopped = False
        self._handle = None
        self._nvml = None
        self.available = self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            return True
        except Exception:
            self._nvml = None
            self._handle = None
            return False

    def start(self):
        self.utils = []
        self.stopped = False
        if not self.available:
            return
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while not self.stopped:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.utils.append(util.gpu)
            except Exception:
                pass
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        if self.available:
            self.thread.join()
        if not self.utils:
            return 0
        return sum(self.utils) / len(self.utils)


if __name__ == "__main__":
    # Quick self-test of the three seams.
    import random

    random.seed(0)

    # Seam 3: order_batch on plain records
    class _R:
        def __init__(self, rid, pref, steps, slo):
            self.rid = rid
            self.model_pref = pref
            self.deadline = slo
            self.default_steps = steps

    batch = [
        _R(0, "small", 30, 6.0),
        _R(1, "large", 40, 20.0),
        _R(2, "auto", 30, 3.5),
        _R(3, "small", 20, 2.0),
        _R(4, "large", 40, 30.0),
    ]
    ordered, assign = order_batch(batch, current_model="small")
    print("[SEAM] order_batch:", [r.rid for r in ordered])
    print("[SEAM] assignments:", [assign[id(r)] for r in ordered])

    # Seam 1: interpolation + fair share
    for s in (10, 20, 25, 30, 40):
        print(f"[SEAM] slo_multiplier({s}) = {slo_multiplier(s):.3f}")
    print("[SEAM] fair_share_step_reduction(40, 30, 5, 2) =",
          fair_share_step_reduction(40, 30, 5, 2))

    # Seam 2: monitor (no GPU -> 0)
    m = GPUUtilMonitor()
    m.start()
    time.sleep(0.1)
    print(f"[SEAM] GPUUtilMonitor.available={m.available} avg={m.stop():.1f}")
    print("[SEAM] module OK.")

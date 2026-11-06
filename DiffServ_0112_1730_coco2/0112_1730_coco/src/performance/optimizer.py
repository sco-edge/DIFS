"""
Performance optimizer for DIFS_OC_v7.

Everything here is opt-in.  With ``DIFS_PERF_OPTIMIZATION`` unset (the default)
the optimizer reports ``enabled == False`` and the scheduler / worker pool behave
exactly as they did in DIFS_OC_v6.  This keeps the v6 benchmark semantics
intact while allowing v7 to run an "optimized" configuration.

What is implemented (not static placeholders):
  * health-aware worker scoring from EWMA latency + recent failures
  * per-worker circuit breakers (closed -> open -> half-open probe)
  * exponential backoff with jitter for the scheduler retry loop
  * arrival-rate prediction (EWMA inter-arrival time)
  * OS page-cache prefetch of model files (reduces cold-start disk reads)
  * preload candidate selection from observed model demand
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ENABLE_ENV = "DIFS_PERF_OPTIMIZATION"
FAILURE_THRESHOLD_ENV = "DIFS_CIRCUIT_FAILURE_THRESHOLD"
RECOVERY_SEC_ENV = "DIFS_CIRCUIT_RECOVERY_SEC"
RETRY_BASE_ENV = "DIFS_RETRY_BASE_SEC"
RETRY_MAX_ENV = "DIFS_RETRY_MAX_SEC"
EWMA_ALPHA_ENV = "DIFS_EWMA_ALPHA"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class CircuitBreaker:
    """Minimal thread-safe circuit breaker.

    CLOSED     -> requests pass
    OPEN       -> requests are rejected until the recovery timeout elapses
    HALF_OPEN  -> exactly one probe request is allowed
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 3, recovery_sec: float = 30.0):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_sec = max(0.0, float(recovery_sec))
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._last_error = ""

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def allow(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN:
                if now - self._opened_at >= self.recovery_sec:
                    self._state = self.HALF_OPEN
                    self._probe_in_flight = True
                    return True
                return False
            # HALF_OPEN: only one probe at a time
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED
            self._probe_in_flight = False
            self._last_error = ""

    def record_failure(self, error: str = "") -> None:
        with self._lock:
            self._failures += 1
            self._last_error = error
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._opened_at = time.time()
                self._probe_in_flight = False
            elif self._failures >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.time()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "state": self._state,
                "failures": self._failures,
                "error": self._last_error,
                "opened_at": self._opened_at,
            }


@dataclass
class _WorkerHealth:
    ewma_latency_ms: float = 0.0
    samples: int = 0
    successes: int = 0
    failures: int = 0
    last_seen: float = field(default_factory=time.time)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class PerformanceOptimizer:
    """Shared optimizer state for one server process."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = _env_bool(ENABLE_ENV, False) if enabled is None else bool(enabled)
        self.alpha = min(1.0, max(0.01, _env_float(EWMA_ALPHA_ENV, 0.3)))
        self.failure_threshold = _env_int(FAILURE_THRESHOLD_ENV, 3)
        self.recovery_sec = _env_float(RECOVERY_SEC_ENV, 30.0)
        self.retry_base_sec = max(0.0, _env_float(RETRY_BASE_ENV, 0.25))
        self.retry_max_sec = max(self.retry_base_sec, _env_float(RETRY_MAX_ENV, 4.0))
        self.min_samples = _env_int("DIFS_HEALTH_MIN_SAMPLES", 3)
        self._lock = threading.Lock()
        self._workers: Dict[str, _WorkerHealth] = {}
        self._arrival_ewma: Optional[float] = None
        self._last_arrival: Optional[float] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_summary: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Worker health / circuit breakers
    # ------------------------------------------------------------------
    def _health(self, worker_addr: str) -> _WorkerHealth:
        with self._lock:
            health = self._workers.get(worker_addr)
            if health is None:
                health = _WorkerHealth(
                    breaker=CircuitBreaker(self.failure_threshold, self.recovery_sec)
                )
                self._workers[worker_addr] = health
            return health

    def allow_worker(self, worker_addr: str) -> bool:
        if not self.enabled:
            return True
        return self._health(worker_addr).breaker.allow()

    def record_worker_success(self, worker_addr: str, latency_ms: float) -> None:
        if not self.enabled:
            return
        health = self._health(worker_addr)
        with self._lock:
            latency_ms = max(0.0, float(latency_ms))
            if health.samples == 0:
                health.ewma_latency_ms = latency_ms
            else:
                health.ewma_latency_ms = (
                    self.alpha * latency_ms + (1.0 - self.alpha) * health.ewma_latency_ms
                )
            health.samples += 1
            health.successes += 1
            health.last_seen = time.time()
        health.breaker.record_success()

    def record_worker_failure(self, worker_addr: str, error: str = "") -> None:
        if not self.enabled:
            return
        health = self._health(worker_addr)
        with self._lock:
            health.failures += 1
            health.last_seen = time.time()
        health.breaker.record_failure(error)

    def predicted_latency_ms(self, worker_addr: str, fallback_ms: float = 0.0) -> float:
        if not self.enabled:
            return float(fallback_ms)
        health = self._health(worker_addr)
        with self._lock:
            if health.samples < self.min_samples:
                return float(fallback_ms)
            return float(health.ewma_latency_ms)

    def health_penalty(self, worker_addr: str) -> float:
        """Extra score added for recently failing / unhealthy workers."""
        if not self.enabled:
            return 0.0
        health = self._health(worker_addr)
        with self._lock:
            penalty = 250.0 * health.failures
            age = time.time() - health.last_seen
            if health.failures and age > self.recovery_sec:
                # stale failure history should not penalise forever
                penalty *= 0.5
            return penalty

    # ------------------------------------------------------------------
    # Retry / backoff
    # ------------------------------------------------------------------
    def retry_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter (attempt is 0-based)."""
        if not self.enabled:
            return 2.0
        ceiling = min(self.retry_max_sec, self.retry_base_sec * (2 ** max(0, attempt)))
        if ceiling <= 0:
            return 0.0
        return random.uniform(ceiling / 2.0, ceiling)

    # ------------------------------------------------------------------
    # Arrival prediction
    # ------------------------------------------------------------------
    def observe_arrival(self, now: Optional[float] = None) -> None:
        if not self.enabled:
            return
        now = time.time() if now is None else now
        with self._lock:
            if self._last_arrival is not None:
                gap = max(0.0, now - self._last_arrival)
                if self._arrival_ewma is None:
                    self._arrival_ewma = gap
                else:
                    self._arrival_ewma = (
                        self.alpha * gap + (1.0 - self.alpha) * self._arrival_ewma
                    )
            self._last_arrival = now

    def predicted_interarrival_sec(self) -> Optional[float]:
        with self._lock:
            return self._arrival_ewma

    # ------------------------------------------------------------------
    # Preload / prefetch
    # ------------------------------------------------------------------
    @staticmethod
    def select_preload_models(
        demand: Dict[int, int], limit: int = 3
    ) -> List[int]:
        """Return the highest-demand model ids (ties broken by id)."""
        ranked = sorted(demand.items(), key=lambda kv: (-kv[1], kv[0]))
        return [model_id for model_id, _ in ranked[: max(0, limit)]]

    def prefetch_model_files(
        self,
        model_names: Sequence[str],
        model_dir: Path,
        chunk_size: int = 1 << 20,
        max_bytes_per_model: int = 64 << 20,
    ) -> Dict[str, object]:
        """Read the head of model files to warm the OS page cache.

        This is safe: it never touches the GPU and never changes which model a
        worker is serving.  Runs in a daemon thread and returns immediately.
        """
        if not self.enabled or not model_names:
            return {"scheduled": False, "reason": "optimizer disabled or no models"}

        def _worker() -> None:
            started = time.time()
            warmed = []
            errors = []
            for name in model_names:
                path = Path(model_dir) / name
                try:
                    if path.is_dir():
                        files = [p for p in path.rglob("*") if p.is_file()]
                    elif path.is_file():
                        files = [path]
                    else:
                        errors.append(f"missing:{name}")
                        continue
                    remaining = max_bytes_per_model
                    read_bytes = 0
                    for f in files:
                        if remaining <= 0:
                            break
                        with open(f, "rb") as fh:
                            while remaining > 0:
                                chunk = fh.read(min(chunk_size, remaining))
                                if not chunk:
                                    break
                                remaining -= len(chunk)
                                read_bytes += len(chunk)
                    warmed.append({"model": name, "bytes": read_bytes})
                except Exception as exc:  # pragma: no cover - host dependent
                    errors.append(f"{name}:{exc}")
            self._prefetch_summary = {
                "scheduled": True,
                "finished": True,
                "elapsed_sec": round(time.time() - started, 3),
                "warmed": warmed,
                "errors": errors,
            }

        thread = threading.Thread(target=_worker, name="difsv7-prefetch", daemon=True)
        self._prefetch_thread = thread
        self._prefetch_summary = {"scheduled": True, "finished": False}
        thread.start()
        return self._prefetch_summary

    def prefetch_status(self) -> Dict[str, object]:
        return dict(self._prefetch_summary)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            workers = {
                addr: {
                    "ewma_latency_ms": round(h.ewma_latency_ms, 3),
                    "samples": h.samples,
                    "successes": h.successes,
                    "failures": h.failures,
                    "breaker": h.breaker.snapshot(),
                }
                for addr, h in self._workers.items()
            }
            return {
                "enabled": self.enabled,
                "alpha": self.alpha,
                "workers": workers,
                "predicted_interarrival_sec": self._arrival_ewma,
                "prefetch": dict(self._prefetch_summary),
            }


_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_optimizer() -> PerformanceOptimizer:
    """Return the per-process optimizer singleton."""
    global _optimizer
    with _optimizer_lock:
        if _optimizer is None:
            _optimizer = PerformanceOptimizer()
        return _optimizer


def reset_optimizer_for_tests(enabled: Optional[bool] = None) -> PerformanceOptimizer:
    """Replace the singleton (used by unit tests and dry runs)."""
    global _optimizer
    with _optimizer_lock:
        _optimizer = PerformanceOptimizer(enabled=enabled)
        return _optimizer


# Backwards-compatible module-level handle used by older code.
optimizer = get_optimizer()

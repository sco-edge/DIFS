"""Performance optimization hooks for DIFS_OC_v7."""

from .optimizer import (  # noqa: F401
    CircuitBreaker,
    PerformanceOptimizer,
    get_optimizer,
    reset_optimizer_for_tests,
)

__all__ = [
    "CircuitBreaker",
    "PerformanceOptimizer",
    "get_optimizer",
    "reset_optimizer_for_tests",
]

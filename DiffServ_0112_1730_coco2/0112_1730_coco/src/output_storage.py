"""
output_storage.py

Resolve a safe directory for generated inference images.

Workers historically wrote to ``/tmp/diffserv_output``.  On this machine
``/tmp`` lives on the root filesystem, which has filled up before (~78 GB of
PNG files caused ``OSError: [Errno 28] No space left on device`` and corrupted
a benchmark cell).  This module keeps that from happening again:

1. if the requested output filesystem has less than the required free space,
   delete the disposable image/``.npz`` files already in the output directory;
2. re-check the free space;
3. if it is still too small, move the output directory to the first writable
   fallback filesystem that has enough free space.

The chosen directory is exported as ``DIFS_OUTPUT_DIR`` so every subprocess
(workers, server, client) inherits the same location.

Configuration
-------------
DIFS_OUTPUT_DIR             requested output directory (default /tmp/diffserv_output)
DIFS_OUTPUT_MIN_FREE_GIB    required free space in GiB (default 30)
DIFS_OUTPUT_FALLBACK_DIR    extra fallback directory/path-list (os.pathsep separated)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_OUTPUT_DIR = "/tmp/diffserv_output"
DEFAULT_MIN_FREE_GIB = 30.0
MIN_FREE_ENV = "DIFS_OUTPUT_MIN_FREE_GIB"
FALLBACK_ENV = "DIFS_OUTPUT_FALLBACK_DIR"

DISPOSABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".npz"}

DEFAULT_FALLBACKS = (
    "/mnt/ssd4t/diffserv_output",
    "/mnt/my_ssd/diffserv_output",
    "/var/tmp/diffserv_output",
)


def _log(message: str) -> None:
    print(f"[OUTPUT] {message}", file=sys.stderr)


def min_free_gib(value: Optional[float] = None) -> float:
    if value is not None:
        return float(value)
    raw = os.environ.get(MIN_FREE_ENV)
    if raw:
        try:
            return float(raw)
        except ValueError:
            _log(f"invalid {MIN_FREE_ENV}={raw!r}; using {DEFAULT_MIN_FREE_GIB}")
    return DEFAULT_MIN_FREE_GIB


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def free_gib(path: os.PathLike | str) -> float:
    """Free space in GiB on the filesystem that holds *path* (or its parent)."""
    ancestor = _existing_ancestor(Path(path))
    return shutil.disk_usage(ancestor).free / (1024 ** 3)


def _disposable_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    found = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in DISPOSABLE_SUFFIXES and not p.is_symlink():
                found.append(p)
    return found


def free_disposable_files(root: os.PathLike | str) -> tuple[int, int]:
    """Delete disposable image/``.npz`` files under *root*.

    Returns ``(count, bytes_freed)``.
    """
    count = 0
    freed = 0
    for p in _disposable_files(Path(root)):
        try:
            size = p.stat().st_size
            p.unlink()
        except OSError:
            continue
        count += 1
        freed += size
    return count, freed


def _fallback_candidates(requested: Path) -> list[Path]:
    raw = os.environ.get(FALLBACK_ENV, "")
    candidates: list[Path] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if item:
            candidates.append(Path(item).expanduser())
    for item in DEFAULT_FALLBACKS:
        candidates.append(Path(item))
    candidates.append(Path.home() / "diffserv_output")

    unique: list[Path] = []
    seen: set[Path] = set()
    requested_resolved = requested.resolve()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved == requested_resolved or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(cand)
    return unique


def resolve_output_dir(
    requested: Optional[os.PathLike | str] = None,
    *,
    min_gib: Optional[float] = None,
    dry_run: bool = False,
    cleanup: bool = True,
    export: bool = True,
) -> Path:
    """Return a writable output directory with enough free space.

    The decision is deterministic for a given filesystem state and is logged to
    stderr whenever a cleanup or relocation happens.
    """
    target = Path(requested or os.environ.get("DIFS_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    needed = min_free_gib(min_gib)

    if dry_run:
        _log(f"[dry-run] would require {needed:g} GiB free for {target}")
        return target

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log(f"cannot create {target}: {exc}")
        raise

    available = free_gib(target)
    if available >= needed:
        if export:
            os.environ["DIFS_OUTPUT_DIR"] = str(target)
        return target

    _log(
        f"only {available:.1f} GiB free on the filesystem holding {target} "
        f"(need {needed:g} GiB)"
    )

    if cleanup:
        count, freed = free_disposable_files(target)
        if count:
            _log(
                f"deleted {count} disposable image/npz file(s) "
                f"({freed / (1024 ** 3):.2f} GiB) from {target}"
            )
            available = free_gib(target)
            _log(f"free space after cleanup: {available:.1f} GiB")
            if available >= needed:
                if export:
                    os.environ["DIFS_OUTPUT_DIR"] = str(target)
                return target

    for candidate in _fallback_candidates(target):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if not os.access(candidate, os.W_OK):
            continue
        candidate_free = free_gib(candidate)
        if candidate_free < needed:
            continue
        _log(
            f"relocating output images from {target} to {candidate} "
            f"({candidate_free:.1f} GiB free)"
        )
        if export:
            os.environ["DIFS_OUTPUT_DIR"] = str(candidate)
        return candidate

    message = (
        f"still only {available:.1f} GiB free for {target} after cleanup, and no "
        f"fallback filesystem has {needed:g} GiB free. Set DIFS_OUTPUT_DIR (or "
        f"--output-dir) to a larger filesystem. "
        f"Tried: {', '.join(str(c) for c in _fallback_candidates(target))}"
    )
    _log(message)
    if export:
        os.environ["DIFS_OUTPUT_DIR"] = str(target)
    return target


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested", default=None, help="requested output directory")
    parser.add_argument("--min-free-gib", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args(argv)

    chosen = resolve_output_dir(
        args.requested,
        min_gib=args.min_free_gib,
        dry_run=args.dry_run,
        cleanup=not args.no_cleanup,
        export=not args.no_export,
    )
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a DeepResearch Bench module on Windows without editing pinned upstream code."""

from __future__ import annotations

import multiprocessing
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_deepresearch_module.py MODULE [ARGS ...]")
    module = sys.argv.pop(1)
    sys.path.insert(0, str(Path.cwd()))
    original = multiprocessing.set_start_method

    def compatible(method=None, force=False):
        # Upstream requests POSIX-only `fork` from inside its __main__ guard.
        # `spawn` is Python's supported Windows equivalent; all benchmark worker
        # functions live in importable utils modules and remain pickleable.
        if sys.platform == "win32" and method == "fork":
            method = "spawn"
        return original(method, force=force)

    multiprocessing.set_start_method = compatible
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()

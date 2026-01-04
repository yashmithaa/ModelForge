"""CLI entrypoint for the `modelforge` package.

This module delegates to `src.cli` so `python -m modelforge.cli` works while keeping
the main implementation in `src/cli.py`.
"""

import sys

from src.cli import main


def _normalize_argv_aliases():
    # Older scripts invoked `modelforge.cli train ...`. The new CLI uses `run`.
    # Normalize `train` -> `run` so both call patterns work.
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        sys.argv[1] = "run"


if __name__ == "__main__":
    _normalize_argv_aliases()
    main()

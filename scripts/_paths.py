"""Path helpers for cellscope scripts.

`setup_imports()` puts the project root on sys.path and chdirs to it,
so scripts can do `from core.X import Y` regardless of where they're run.

`benchmark_data_root()` returns the location of the sibling
`piezo1_analysis` project (with manual GT, training pairs, example
recordings). Defaults to a sibling directory next to cellscope/.
Override with the BENCHMARK_DATA_ROOT environment variable.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_imports():
    """Idempotently put PROJECT_ROOT on sys.path and chdir to it."""
    sp = str(PROJECT_ROOT)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    os.chdir(sp)


def benchmark_data_root():
    """Path to the sibling piezo1_analysis project (override via
    BENCHMARK_DATA_ROOT env var)."""
    env = os.environ.get("BENCHMARK_DATA_ROOT")
    if env:
        return Path(env)
    return PROJECT_ROOT.parent / "piezo1_analysis"


def require_benchmark_data():
    """Like benchmark_data_root() but raises a helpful error if missing."""
    p = benchmark_data_root()
    if not p.exists():
        raise SystemExit(
            f"Benchmark data not found at:\n  {p}\n\n"
            f"This script requires the sibling piezo1_analysis project "
            f"(manual GT + training data). Either clone/place it next to "
            f"cellscope/, or set BENCHMARK_DATA_ROOT=/path/to/piezo1_analysis"
        )
    return p

"""Shared, reproducible project configuration for the MNIST coursework."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Final

import numpy as np

RANDOM_SEED: Final[int] = 42
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_MNIST_DIR: Final[Path] = DATA_DIR / "raw" / "mnist"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
TABLES_DIR: Final[Path] = RESULTS_DIR / "tables"


def runtime_metadata() -> dict[str, str | int]:
    """Return non-secret information needed to reproduce a local run."""
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "execution_device": "CPU (no accelerator framework configured in tasks 1.2-2.3)",
        "numpy_version": np.__version__,
        "random_seed": RANDOM_SEED,
    }
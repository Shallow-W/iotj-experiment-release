"""Paths shared by the original training entry points."""

import os
from pathlib import Path
from tempfile import mkdtemp


PAPER_ROOT = Path(__file__).resolve().parents[2]


def input_prefix(scenario):
    return str(PAPER_ROOT / "inputs" / scenario) + os.sep


def output_directory(script):
    output_root = PAPER_ROOT / "reproduced" / "training"
    output_root.mkdir(parents=True, exist_ok=True)
    return mkdtemp(prefix=f"{Path(script).stem}-", dir=output_root)


def episode_count(default):
    count = int(os.environ.get("IOTJ_EPISODES", default))
    if count < 1:
        raise ValueError("IOTJ_EPISODES must be a positive integer")
    return count

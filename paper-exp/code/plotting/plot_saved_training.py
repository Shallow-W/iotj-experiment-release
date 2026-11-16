#!/usr/bin/env python3
"""Plot unscaled training records saved with the original manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "training"
OUTPUT = ROOT / "reproduced" / "figures"
FULL = ("MPD3QN", "d3qn_mask_pri")
BASELINES = (FULL, ("DQN", "dqn"), ("SAC", "sac"), ("GDDPG", "ddpg"))


def load_series(metric, variant):
    path = RESULTS / f"{metric}_data_{variant}.xlsx"
    frame = pd.read_excel(path, header=None)
    series = frame.apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    if series.empty or series.isna().any():
        raise ValueError(f"Missing numeric records in {path}")
    return series


def plot_series(axis, metric, methods, ylabel):
    for label, variant in methods:
        values = load_series(metric, variant)
        axis.plot(range(1, len(values) + 1), values, linewidth=1, label=label)
    axis.set(xlabel="Episode", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend()


def save(figure, name):
    path = OUTPUT / f"{name}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, alternative in (
        ("per", ("Without PER", "d3qn_mask_nopri")),
        ("masking", ("Without masking", "d3qn_nomask_pri")),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        plot_series(axes[0], "rewards", (FULL, alternative), "Episode return")
        plot_series(axes[1], "delays", (FULL, alternative), "Final delay (ms)")
        save(figure, f"saved_training_{name}")

    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    plot_series(axis, "rewards", BASELINES, "Episode return")
    save(figure, "saved_training_reward_comparison")


if __name__ == "__main__":
    main()

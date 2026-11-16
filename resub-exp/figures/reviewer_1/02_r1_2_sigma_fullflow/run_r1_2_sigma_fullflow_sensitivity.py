#!/usr/bin/env python3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EXP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXP_ROOT
FIGURES_DIR = EXP_ROOT / "reproduced"

DEFAULT_SIGMA_DELAY = 1.0
DEFAULT_SIGMA_RES = 1.0
DEFAULT_RED = "#D55E00"


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def find_row(rows, **kwargs):
    for row in rows:
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    raise KeyError(kwargs)


def sigma_values(rows, key):
    return sorted({float(row[key]) for row in rows})


def row_for(rows, sigma_delay, sigma_res):
    return find_row(rows, sigma_delay=sigma_delay, sigma_res=sigma_res)


def metric_grid(rows, metric, delay_values, res_values):
    data = np.zeros((len(res_values), len(delay_values)))
    for i, sigma_res in enumerate(res_values):
        for j, sigma_delay in enumerate(delay_values):
            data[i, j] = row_for(rows, sigma_delay, sigma_res)[metric]
    return data


def make_heatmap_figure(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    delay_values = sigma_values(rows, "sigma_delay")
    res_values = sigma_values(rows, "sigma_res")
    default_j = delay_values.index(DEFAULT_SIGMA_DELAY)
    default_i = res_values.index(DEFAULT_SIGMA_RES)

    metrics = [
        ("mean_fullflow_delay", "(a) Final delay", "{:.0f}", "viridis_r", None, None, "Delay"),
        ("cut_loss", "(b) Cut loss", "{:.3f}", "magma", None, None, "Cut loss"),
        ("psi", "(c) Cross-domain ratio", "{:.3f}", "YlOrRd", None, None, r"$\psi$"),
        ("resource_satisfaction_rate", "(d) Resource satisfaction", "{:.3f}", "Greens", 0.95, 1.0, "Satisfaction rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
    for ax, (metric, title, fmt, cmap, vmin, vmax, cbar_label) in zip(axes.ravel(), metrics):
        data = metric_grid(rows, metric, delay_values, res_values)
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$\sigma_{delay}$")
        ax.set_ylabel(r"$\sigma_{res}$")
        ax.set_xticks(range(len(delay_values)))
        ax.set_xticklabels([str(value) for value in delay_values])
        ax.set_yticks(range(len(res_values)))
        ax.set_yticklabels([str(value) for value in res_values])
        ax.add_patch(Rectangle((default_j - 0.5, default_i - 0.5), 1, 1, fill=False, edgecolor=DEFAULT_RED, linewidth=2.3))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                rgba = im.cmap(im.norm(data[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.45 else "black"
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8.5, color=text_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    default_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor=DEFAULT_RED, linewidth=2.3)
    fig.legend(
        handles=[default_handle],
        labels=[r"Default setting ($\sigma_{delay}=\sigma_{res}=1.0$)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        handlelength=1.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    png = FIGURES_DIR / "r1_2_sigma_fullflow_sensitivity_2x2.png"
    pdf = FIGURES_DIR / "r1_2_sigma_fullflow_sensitivity_2x2.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    rows = pd.read_csv(RESULTS_DIR / "r1_2_sigma_fullflow_sensitivity.csv").to_dict("records")
    for path in make_heatmap_figure(rows):
        print(path.relative_to(EXP_ROOT))


if __name__ == "__main__":
    main()

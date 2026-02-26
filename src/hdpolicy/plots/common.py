from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt


def new_fig(figsize = (7, 5)):
    fig, ax = plt.subplots(figsize = figsize)
    return fig, ax


def save_fig(fig, path: str | Path, dpi: int = 200):

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(path, dpi = dpi, bbox_inches = "tight")


def style_axes(ax):
    ax.set_xlabel("Number of features (p)")
    ax.grid(True, which = "both", axis = "y", linestyle = "--", alpha = 0.4)
    ax.legend()

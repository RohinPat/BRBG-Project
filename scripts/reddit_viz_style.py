"""Shared colors and matplotlib styling for paper figures."""
from __future__ import annotations

GROUP_COLORS = {
    "brainrot": "#D55E00",
    "general_slang": "#0072B2",
    "standard_control": "#009E73",
}

GROUP_LABELS = {
    "brainrot": "Brainrot",
    "general_slang": "General slang",
    "standard_control": "Standard control",
}

GROUP_ORDER = ("brainrot", "general_slang", "standard_control")

BIN_COLORS = {
    "early": "#332288",
    "late": "#CC6677",
}

YEAR_BUCKET_COLORS = {
    "early_window": "#332288",
    "late_window": "#CC6677",
    "gap_year": "#BBBBBB",
    "other": "#D0D0D0",
}

FIG_FACE = "#FAFAFA"


def apply_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Segoe UI",
                "DejaVu Sans",
                "Helvetica",
                "Arial",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": FIG_FACE,
            "axes.facecolor": FIG_FACE,
            "savefig.facecolor": FIG_FACE,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[GROUP_COLORS[g] for g in GROUP_ORDER]
    )

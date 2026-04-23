"""
Build summary figures from saved JSON artifacts (no model loading).

Writes outputs to `results/figures/`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from reddit_sample_utils import effective_calendar_year, row_lexicon_key
from reddit_viz_style import (
    FIG_FACE,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    YEAR_BUCKET_COLORS,
    apply_style,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
COV_PATH = ROOT / "results" / "corpus_coverage.json"
ROBERTA_PATH = ROOT / "results" / "reddit_roberta_apd.json"
TFIDF_PATH = ROOT / "results" / "reddit_tfidf_apd.json"
FASTTEXT_PATH = ROOT / "results" / "reddit_fasttext_apd.json"
JSONL_PATH = ROOT / "data" / "reddit_sample" / "reddit_sample_comments.jsonl"
TARGET_JSON = ROOT / "data" / "target_words.json"

GROUP_KEYS = ("brainrot", "general_slang", "standard_control")


def normalize(q: str) -> str:
    return (q or "").strip().lower()


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analysis_min_bin() -> int:
    if not ROBERTA_PATH.exists():
        return 20
    ro = load_json(ROBERTA_PATH)
    return int(ro.get("min_bin_for_group_stats", 20))


def per_word_balanced_flag(info: dict, min_bin: int) -> bool:
    key = f"both_bins_ge_{min_bin}"
    if key in info:
        return bool(info[key])
    return bool(info.get("both_bins_ge_50", False))


def load_word_to_group() -> dict[str, str]:
    with open(TARGET_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    w2g: dict[str, str] = {}
    for g in GROUP_KEYS:
        for w in cfg.get(g, []):
            w2g[normalize(w)] = g
    return w2g


def calendar_year_bucket(y: int) -> str:
    if y in (2019, 2020, 2021):
        return "early_window"
    if y in (2022, 2023, 2024):
        return "late_window"
    return "other"


def figure_group_means_three_encoders():
    ro = load_json(ROBERTA_PATH)
    tf = load_json(TFIDF_PATH)
    ft = load_json(FASTTEXT_PATH)

    ro_means = {g: ro["group_summary"][g]["mean_apd_balanced"] for g in GROUP_ORDER}
    ro_n = {g: ro["group_summary"][g]["n_words_balanced"] for g in GROUP_ORDER}
    tf_means = {g: tf["group_mean_apd_balanced"][g] for g in GROUP_ORDER}
    ft_means = {g: ft["group_mean_apd_balanced"][g] for g in GROUP_ORDER}

    titles = (
        "RoBERTa (contextual target span)",
        "TF–IDF (bag-of-ngrams)",
        "fastText (mean word vectors)",
    )
    series = (ro_means, tf_means, ft_means)
    fmtters = (
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.3f}",
        lambda v: f"{v:.2e}",
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.35))
    fig.patch.set_facecolor(FIG_FACE)
    x = np.arange(len(GROUP_ORDER))
    width = 0.72

    for ax, means, title, fmt in zip(axes, series, titles, fmtters, strict=True):
        ax.set_facecolor(FIG_FACE)
        colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
        heights = [means[g] for g in GROUP_ORDER]
        bars = ax.bar(x, heights, width, color=colors, edgecolor="#222", linewidth=0.6, zorder=3)
        ax.set_xticks(x, [GROUP_LABELS[g] for g in GROUP_ORDER], rotation=0)
        ax.set_title(title, pad=8)
        ax.set_ylabel("Mean APD (balanced words)")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        for i, g in enumerate(GROUP_ORDER):
            h = bars[i].get_height()
            ax.annotate(
                f"n={ro_n[g]}",
                xy=(bars[i].get_x() + bars[i].get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555",
            )
            ax.annotate(
                fmt(heights[i]),
                xy=(bars[i].get_x() + bars[i].get_width() / 2, h),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color="#111",
            )

    fig.suptitle(
        "Early vs late divergence by group (same balanced word set per encoder)",
        fontsize=14,
        y=1.03,
    )
    fig.text(
        0.5,
        0.02,
        "Note: absolute APD values are not comparable across encoders—compare group ordering within each panel only.",
        ha="center",
        fontsize=9,
        color="#444",
        style="italic",
    )
    fig.subplots_adjust(bottom=0.2, top=0.88, wspace=0.28)

    out = FIG_DIR / "apd_group_means_by_encoder.png"
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", out)


def figure_roberta_per_word():
    ro = load_json(ROBERTA_PATH)
    mb = int(ro.get("min_bin_for_group_stats", 30))
    per = ro["per_word"]
    rows = []
    for w, info in per.items():
        if info.get("apd") is None:
            continue
        g = info["group"]
        rows.append((w, float(info["apd"]), g))

    rows.sort(key=lambda t: (GROUP_ORDER.index(t[2]), -t[1]))

    fig_h = max(5.5, min(22.0, 0.32 * len(rows)))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    y = np.arange(len(rows))

    by_g: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, g) in enumerate(rows):
        by_g[g].append(i)

    for g in GROUP_ORDER:
        idx = by_g.get(g, [])
        if not idx:
            continue
        lo, hi = min(idx), max(idx)
        ax.axhspan(
            lo - 0.52,
            hi + 0.52,
            facecolor=GROUP_COLORS[g],
            alpha=0.07,
            zorder=0,
            lw=0,
        )

    colors = [GROUP_COLORS[g] for _, _, g in rows]
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="#222", linewidth=0.35, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RoBERTa APD (early vs late)")
    ax.set_title(f"Per-word APD — words with ≥{mb} comments per bin (RoBERTa run)")
    xmax = max(vals) * 1.12 if vals else 1.0
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    for yi, v in zip(y, vals, strict=True):
        ax.text(
            v + 0.002,
            yi,
            f"{v:.3f}",
            va="center",
            fontsize=7,
            color="#333",
            zorder=4,
        )

    from matplotlib.patches import Patch

    leg = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="#222", label=GROUP_LABELS[g])
        for g in GROUP_ORDER
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=9)

    out = FIG_DIR / "apd_roberta_per_word.png"
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", out)


def figure_corpus_early_vs_late():
    cov = load_json(COV_PATH)
    per = cov["per_word"]
    mb = analysis_min_bin()
    unbalanced = []
    balanced = []
    for w, info in per.items():
        ok_bal = per_word_balanced_flag(info, mb)
        t = (info["n_early"], info["n_late"], info["group"], w, ok_bal)
        if t[4]:
            balanced.append(t)
        else:
            unbalanced.append(t)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    ax.grid(False)

    def scatter_subset(subset, *, balanced_set: bool):
        for g in GROUP_ORDER:
            pts = [p for p in subset if p[2] == g]
            if not pts:
                continue
            ne = np.array([p[0] for p in pts], dtype=float)
            nl = np.array([p[1] for p in pts], dtype=float)
            ax.scatter(
                ne,
                nl,
                c=GROUP_COLORS[g],
                s=110 if balanced_set else 44,
                alpha=0.88 if balanced_set else 0.65,
                edgecolors="#222",
                linewidths=1.1 if balanced_set else 0.35,
                label=GROUP_LABELS[g] if not balanced_set else None,
            )

    scatter_subset(unbalanced, balanced_set=False)
    scatter_subset(balanced, balanced_set=True)

    h_ro = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888",
            markeredgecolor="#222",
            markersize=7,
            markeredgewidth=0.5,
            linestyle="",
            label=f"<{mb} in one or both bins",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888",
            markeredgecolor="#222",
            markersize=10,
            markeredgewidth=1.3,
            linestyle="",
            label=f"Both bins ≥{mb} (group-mean eligible)",
        ),
    ]
    leg1 = ax.legend(handles=h_ro, loc="upper left", fontsize=8, title="Marker size", title_fontsize=8)
    ax.add_artist(leg1)

    g_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GROUP_COLORS[g],
            markeredgecolor="#222",
            markersize=8,
            linestyle="",
            label=GROUP_LABELS[g],
        )
        for g in GROUP_ORDER
    ]
    ax.legend(handles=g_handles, loc="lower right", fontsize=8, title="Color", title_fontsize=8)

    ax.axvline(mb, color="#555", linestyle=":", linewidth=1, alpha=0.9, zorder=0)
    ax.axhline(mb, color="#555", linestyle=":", linewidth=1, alpha=0.9, zorder=0)
    ax.plot([1, 5000], [1, 5000], color="#999", linestyle="--", linewidth=0.85, alpha=0.75, zorder=0)

    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Comments in early bin (2019–2021)")
    ax.set_ylabel("Comments in late bin (2022–2024)")
    ax.set_title("Corpus coverage per target (symmetric log scale)")

    n20 = cov.get("words_with_both_bins_ge_20", "?")
    n30 = cov.get("words_with_both_bins_ge_30", "?")
    n50 = cov.get("words_with_both_bins_ge_50", "?")
    txt = (
        f"Total JSONL lines: {cov.get('jsonl_lines_total', '?')}\n"
        f"Both bins ≥20: {n20} / {cov.get('target_words_count', '?')}\n"
        f"Both bins ≥30: {n30} / {cov.get('target_words_count', '?')}\n"
        f"Both bins ≥50: {n50} / {cov.get('target_words_count', '?')}"
    )
    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#ccc", alpha=0.95),
    )

    top_bal = sorted(balanced, key=lambda p: p[0] + p[1], reverse=True)[:10]
    for ne, nl, _g, w, _ in top_bal:
        ax.annotate(
            w,
            (ne, nl),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#1a1a1a",
            alpha=0.92,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#ccc", alpha=0.88),
        )

    out = FIG_DIR / "corpus_early_vs_late_counts.png"
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", out)


def figure_comments_per_year():
    if not JSONL_PATH.exists():
        print("Skip comments_per_calendar_year: JSONL missing")
        return

    by_year: dict[int, int] = defaultdict(int)
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            y = effective_calendar_year(row)
            if y is not None:
                by_year[y] += 1

    years = sorted(by_year)
    counts = [by_year[y] for y in years]
    bar_colors = [YEAR_BUCKET_COLORS[calendar_year_bucket(y)] for y in years]

    fig, ax = plt.subplots(figsize=(8.2, 4.35))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    bars = ax.bar(years, counts, color=bar_colors, edgecolor="#2a3340", linewidth=0.45, zorder=3)
    ax.set_xticks(years)
    ax.set_xlabel("Calendar year (from comment timestamp)")
    ax.set_ylabel("Comment rows in sample")
    ax.set_title("Reddit sample volume by year (all matched queries)")
    ax.yaxis.grid(True, zorder=0, alpha=0.3)
    ax.set_axisbelow(True)

    for b, c in zip(bars, counts, strict=True):
        if c > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                str(c),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    from matplotlib.patches import Patch

    leg_items = [
        ("Early analysis window (2019–21)", YEAR_BUCKET_COLORS["early_window"]),
        ("Late analysis window (2022–24)", YEAR_BUCKET_COLORS["late_window"]),
        ("Outside design window", YEAR_BUCKET_COLORS["other"]),
    ]
    handles = [Patch(facecolor=c, edgecolor="#333", label=l) for l, c in leg_items]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    out = FIG_DIR / "comments_per_calendar_year.png"
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", out)


def figure_frequency_by_year_line():
    """Line + markers (same totals as bar chart); legacy-friendly filename."""
    if not JSONL_PATH.exists():
        print("Skip frequency_by_year: JSONL missing")
        return

    by_year: dict[int, int] = defaultdict(int)
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            y = effective_calendar_year(row)
            if y is not None:
                by_year[y] += 1

    years = sorted(by_year)
    counts = [by_year[y] for y in years]
    pt_colors = [YEAR_BUCKET_COLORS[calendar_year_bucket(y)] for y in years]

    fig, ax = plt.subplots(figsize=(8.2, 4.25))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)
    ax.plot(years, counts, color="#333333", linewidth=1.35, alpha=0.85, zorder=2)
    ax.scatter(
        years,
        counts,
        c=pt_colors,
        s=72,
        edgecolors="#1a1a1a",
        linewidths=0.45,
        zorder=4,
    )
    ax.set_xticks(years)
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Comment rows in sample")
    ax.set_title("Sample frequency over time (line + markers)")
    ax.yaxis.grid(True, zorder=0, alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch

    leg_items = [
        ("Early window (2019–21)", YEAR_BUCKET_COLORS["early_window"]),
        ("Late window (2022–24)", YEAR_BUCKET_COLORS["late_window"]),
        ("Other", YEAR_BUCKET_COLORS["other"]),
    ]
    handles = [Patch(facecolor=c, edgecolor="#333", label=l) for l, c in leg_items]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    out = FIG_DIR / "frequency_by_year.png"
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", out)


def figure_stacked_volume_by_group():
    """Stacked bars: per calendar year, JSONL rows attributed to lexicon group."""
    if not JSONL_PATH.exists():
        print("Skip sample_volume_stacked_by_group: JSONL missing")
        return

    w2g = load_word_to_group()
    by_year: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            y = effective_calendar_year(row)
            if y is None:
                continue
            mq = normalize(row_lexicon_key(row))
            if mq not in w2g:
                continue
            by_year[y][w2g[mq]] += 1

    years = sorted(by_year)
    if not years:
        print("Skip sample_volume_stacked_by_group: no matched rows")
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)

    x = np.arange(len(years), dtype=float)
    bottom = np.zeros(len(years))
    for g in GROUP_ORDER:
        hts = np.array([by_year[y].get(g, 0) for y in years], dtype=float)
        ax.bar(
            x,
            hts,
            bottom=bottom,
            label=GROUP_LABELS[g],
            color=GROUP_COLORS[g],
            edgecolor="#1a1a1a",
            linewidth=0.35,
            width=0.75,
        )
        bottom = bottom + hts

    ax.set_xticks(x, [str(y) for y in years])
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Comments (target lexicon only)")
    ax.set_title("Sample volume by year and lexicon group")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    out = FIG_DIR / "sample_volume_stacked_by_group.png"
    fig.savefig(out, bbox_inches="tight", facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", out)


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    missing = [p for p in (COV_PATH, ROBERTA_PATH, TFIDF_PATH, FASTTEXT_PATH) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing inputs: {missing}. Run APD scripts and report_corpus_coverage first.")

    figure_group_means_three_encoders()
    figure_roberta_per_word()
    figure_corpus_early_vs_late()
    figure_comments_per_year()
    figure_frequency_by_year_line()
    figure_stacked_volume_by_group()


if __name__ == "__main__":
    main()

"""
Generate per-word UMAP plots for balanced target words.

Writes PNGs to `results/figures/umap_all_words/`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from reddit_sample_utils import effective_calendar_year, row_lexicon_key, time_bin_for_year
from reddit_viz_style import BIN_COLORS, FIG_FACE, apply_style

ROOT = Path(__file__).resolve().parents[1]
DATA_JSONL = ROOT / "data" / "reddit_sample" / "reddit_sample_comments.jsonl"
OUT_DIR_DEFAULT = ROOT / "results" / "figures" / "umap_all_words"


def _load_module_by_path(name: str, path: Path):
    """Load a sibling script as a module (avoids packaging the scripts/ dir)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_rapd():
    """Dynamically import reddit_roberta_apd.py to reuse its tokenizer/embedding helpers."""
    return _load_module_by_path(
        "reddit_roberta_apd",
        Path(__file__).resolve().parent / "reddit_roberta_apd.py",
    )


def install_umap():
    """Install umap-learn on demand so `pip install -r requirements.txt` is still enough."""
    try:
        import umap  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "umap-learn", "matplotlib", "-q"]
        )


def collect_span_matched_rows(rapd) -> dict[str, list[dict]]:
    """Stream the merged JSONL once, keeping only rows whose body contains a
    boundary-matched span for their canonical target. Returns {word: [rows...]}."""
    by_word: dict[str, list[dict]] = {}
    with open(DATA_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = rapd.normalize_query(row_lexicon_key(row))
            if not key:
                continue
            y = effective_calendar_year(row)
            if y is None:
                continue
            bin_name = time_bin_for_year(y)
            if bin_name is None:
                continue
            body = row.get("body") or ""
            pat = rapd.word_pattern(row.get("matched_query") or row_lexicon_key(row))
            if rapd.find_first_span(body, pat) is None:
                continue
            by_word.setdefault(key, []).append({"body": body, "bin": bin_name})
    return by_word


def balanced_words(
    by_word: dict[str, list[dict]],
    w2g: dict[str, str],
    min_bin: int,
    groups_filter: set[str] | None,
) -> list[tuple[str, str]]:
    """Return sorted [(word, group)] whose early AND late bin each clear min_bin."""
    out: list[tuple[str, str]] = []
    for w, rows in by_word.items():
        g = w2g.get(w)
        if g is None:
            continue
        if groups_filter is not None and g not in groups_filter:
            continue
        n_e = sum(1 for r in rows if r["bin"] == "early")
        n_l = sum(1 for r in rows if r["bin"] == "late")
        if n_e >= min_bin and n_l >= min_bin:
            out.append((w, g))
    out.sort()
    return out


def _add_kde_contours(ax, z_sub: np.ndarray, color: str, *, zorder: float = 1) -> None:
    """Overlay a Gaussian KDE contour set for the subset `z_sub` (skipped if too sparse)."""
    if z_sub.shape[0] < 12:
        return
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return
    try:
        kde = gaussian_kde(z_sub.T, bw_method=0.2)
    except np.linalg.LinAlgError:
        return
    xmi, xma = float(z_sub[:, 0].min()), float(z_sub[:, 0].max())
    ymi, yma = float(z_sub[:, 1].min()), float(z_sub[:, 1].max())
    pad_x = max((xma - xmi) * 0.2, 0.15)
    pad_y = max((yma - ymi) * 0.2, 0.15)
    gx = np.linspace(xmi - pad_x, xma + pad_x, 90)
    gy = np.linspace(ymi - pad_y, yma + pad_y, 90)
    X, Y = np.meshgrid(gx, gy)
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(pos).reshape(X.shape)
    ax.contour(X, Y, Z, levels=5, colors=[color], linewidths=1.05, alpha=0.45, zorder=zorder)


def _draw_umap_on_ax(
    ax,
    z: np.ndarray,
    labels: np.ndarray,
    word: str,
    n_early: int,
    n_late: int,
    max_per_bin: int,
    *,
    kde: bool,
    show_legend: bool,
) -> None:
    """Render scatter + optional KDE contours for one word's UMAP projection."""
    from matplotlib.lines import Line2D

    ze = z[labels == 0]
    zl = z[labels == 1]
    if kde:
        _add_kde_contours(ax, ze, BIN_COLORS["early"], zorder=1)
        _add_kde_contours(ax, zl, BIN_COLORS["late"], zorder=2)

    ax.scatter(
        ze[:, 0], ze[:, 1],
        alpha=0.55, s=22, c=BIN_COLORS["early"],
        edgecolors="#1a1a2e", linewidths=0.25, zorder=4,
    )
    ax.scatter(
        zl[:, 0], zl[:, 1],
        alpha=0.55, s=22, c=BIN_COLORS["late"],
        edgecolors="#1a1a2e", linewidths=0.25, zorder=4,
    )
    ax.set_title(
        f'"{word}"\nn_early={n_early}, n_late={n_late} (cap {max_per_bin}/bin)',
        fontsize=11,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(False)
    if show_legend:
        leg = [
            Line2D([0], [0], marker="o", color="w", label="Early (2019-2021)",
                   markerfacecolor=BIN_COLORS["early"], markeredgecolor="#1a1a2e", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Late (2022-2024)",
                   markerfacecolor=BIN_COLORS["late"], markeredgecolor="#1a1a2e", markersize=8),
        ]
        ax.legend(handles=leg, loc="best", fontsize=9)


def _compute_umap_for_rows(
    word: str,
    rows: list[dict],
    rapd,
    tokenizer,
    model,
    device,
    max_per_bin: int,
):
    """Embed early+late rows with RoBERTa and fit a 2D UMAP. Returns (z, labels, n_early, n_late) or None."""
    early = [r for r in rows if r["bin"] == "early"]
    late = [r for r in rows if r["bin"] == "late"]
    early = rapd.subsample(early, max_per_bin)
    late = rapd.subsample(late, max_per_bin)
    if not early or not late:
        print(f"Skip UMAP for {word!r}: empty bin after subsample "
              f"(early={len(early)}, late={len(late)}).")
        return None

    import umap

    texts = [r["body"] for r in early + late]
    queries = [word] * len(texts)
    labels = np.array([0] * len(early) + [1] * len(late))
    embs = rapd.extract_embeddings_batched(texts, queries, tokenizer, model, device)
    reducer = umap.UMAP(
        n_neighbors=min(15, len(embs) - 1),
        min_dist=0.15,
        metric="cosine",
        random_state=rapd.SEED,
    )
    z = reducer.fit_transform(embs)
    return z, labels, len(early), len(late)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--min-bin", type=int, default=20,
                    help="Minimum span-matched comments per bin to include a target.")
    ap.add_argument("--max-per-bin", type=int, default=120,
                    help="Cap comments per bin before UMAP (matches reddit_roberta_apd.py default).")
    ap.add_argument("--model-name", default="roberta-base",
                    help="HF model id or local path.")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=["brainrot", "general_slang", "standard_control"],
        help="Restrict to one or more lexical groups (default: all three).",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only plot the first N balanced words (smoke test).")
    ap.add_argument("--no-kde", action="store_true",
                    help="Disable Gaussian KDE contour underlays (faster, plainer plots).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip words whose PNG is already present in the output directory.")
    args = ap.parse_args()

    rapd = load_rapd()
    rapd.install_deps()
    install_umap()

    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoModel, AutoTokenizer

    apply_style()

    targets = rapd.load_targets()
    w2g = rapd.word_to_group(targets)

    print("Scanning span-matched rows ...")
    by_word = collect_span_matched_rows(rapd)
    groups_filter = set(args.groups) if args.groups else None
    words = balanced_words(by_word, w2g, args.min_bin, groups_filter)
    if args.limit > 0:
        words = words[: args.limit]

    if not words:
        raise SystemExit(
            f"No words meet >= {args.min_bin} span-matched in both bins"
            + (f" for groups {sorted(groups_filter)}" if groups_filter else "")
        )

    print(f"{len(words)} target words to plot.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    rapd.MAX_PER_BIN = args.max_per_bin
    np.random.seed(rapd.SEED)

    print(f"Loading {args.model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    written: list[Path] = []
    skipped: list[tuple[str, str]] = []

    for i, (w, group) in enumerate(words, 1):
        slug = rapd.normalize_query(w).replace(" ", "_")
        png_path = out_dir / f"umap_{group}_{slug}.png"
        if args.skip_existing and png_path.exists():
            print(f"[{i}/{len(words)}] skip (exists): {png_path.name}")
            continue

        try:
            out = _compute_umap_for_rows(
                w, by_word[w], rapd, tokenizer, model, device, args.max_per_bin
            )
        except Exception as exc:
            print(f"[{i}/{len(words)}] {w!r} failed: {exc}")
            skipped.append((w, str(exc)))
            continue

        if out is None:
            print(f"[{i}/{len(words)}] {w!r} skipped (insufficient bin coverage).")
            skipped.append((w, "insufficient bin coverage"))
            continue

        z, labels, n_early, n_late = out
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        fig.patch.set_facecolor(FIG_FACE)
        ax.set_facecolor(FIG_FACE)
        _draw_umap_on_ax(
            ax, z, labels, w, n_early, n_late, args.max_per_bin,
            kde=not args.no_kde, show_legend=True,
        )
        fig.text(0.02, 0.97, f"group: {group}", ha="left", va="top",
                 fontsize=9, color="#333", fontweight="bold")
        fig.text(0.99, 0.01, args.model_name, ha="right", va="bottom",
                 fontsize=8, color="#555")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, facecolor=FIG_FACE)
        plt.close(fig)
        written.append(png_path)
        print(f"[{i}/{len(words)}] wrote {png_path.name}")

    print(f"\nWrote {len(written)} PNGs to {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} words:")
        for w, reason in skipped:
            print(f"  - {w}: {reason}")


if __name__ == "__main__":
    main()

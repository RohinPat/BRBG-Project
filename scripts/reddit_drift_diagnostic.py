"""
Compare cross-bin APD against within-bin APD for each balanced target word.

Writes:
  - results/reddit_drift_diagnostic.json
  - results/reddit_drift_diagnostic.md
  - results/figures/drift_diagnostic_scatter.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reddit_sample_utils import effective_calendar_year, row_lexicon_key, time_bin_for_year
from reddit_viz_style import FIG_FACE, GROUP_COLORS, GROUP_LABELS, GROUP_ORDER, apply_style

ROOT = Path(__file__).resolve().parents[1]
DATA_JSONL = ROOT / "data" / "reddit_sample" / "reddit_sample_comments.jsonl"
OUT_JSON = ROOT / "results" / "reddit_drift_diagnostic.json"
OUT_MD = ROOT / "results" / "reddit_drift_diagnostic.md"
OUT_FIG_DIR = ROOT / "results" / "figures"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_rapd():
    return _load_by_path(
        "reddit_roberta_apd", Path(__file__).resolve().parent / "reddit_roberta_apd.py"
    )


def collect_span_matched(rapd) -> dict[str, list[dict]]:
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


def apd(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine distance between two sets of L2-normalized row vectors."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    sims = a @ b.T
    return float((1.0 - sims).mean())


def within_bin_apd(embs: np.ndarray, n_splits: int, rng: np.random.Generator) -> float:
    """Average APD between two random halves of `embs` over n_splits draws."""
    n = embs.shape[0]
    if n < 4:
        return float("nan")
    half = n // 2
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a = embs[perm[:half]]
        b = embs[perm[half : 2 * half]]
        vals.append(apd(a, b))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--min-bin", type=int, default=20)
    ap.add_argument("--max-per-bin", type=int, default=120)
    ap.add_argument("--n-splits", type=int, default=5, help="Random halving repeats for within-bin APD.")
    ap.add_argument("--model-name", default="roberta-base")
    args = ap.parse_args()

    rapd = load_rapd()
    rapd.install_deps()

    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoModel, AutoTokenizer

    apply_style()

    targets = rapd.load_targets()
    w2g = rapd.word_to_group(targets)

    print("Scanning span-matched rows ...")
    by_word = collect_span_matched(rapd)

    words: list[tuple[str, str]] = []
    for w, rows in by_word.items():
        g = w2g.get(w)
        if g is None:
            continue
        n_e = sum(1 for r in rows if r["bin"] == "early")
        n_l = sum(1 for r in rows if r["bin"] == "late")
        if n_e >= args.min_bin and n_l >= args.min_bin:
            words.append((w, g))
    words.sort()
    print(f"{len(words)} balanced words (>= {args.min_bin} per bin).")
    if not words:
        raise SystemExit("No balanced words.")

    rapd.MAX_PER_BIN = args.max_per_bin
    rng = np.random.default_rng(rapd.SEED)
    random.seed(rapd.SEED)
    np.random.seed(rapd.SEED)

    print(f"Loading {args.model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    per_word: dict[str, dict] = {}
    for i, (w, g) in enumerate(words, 1):
        rows = by_word[w]
        early = rapd.subsample([r for r in rows if r["bin"] == "early"], args.max_per_bin)
        late = rapd.subsample([r for r in rows if r["bin"] == "late"], args.max_per_bin)
        texts = [r["body"] for r in early + late]
        queries = [w] * len(texts)
        embs = rapd.extract_embeddings_batched(texts, queries, tokenizer, model, device)
        n_e = len(early)
        early_e = embs[:n_e]
        late_e = embs[n_e:]

        a_cross = apd(early_e, late_e)
        a_within_e = within_bin_apd(early_e, args.n_splits, rng)
        a_within_l = within_bin_apd(late_e, args.n_splits, rng)
        a_within = float(np.nanmean([a_within_e, a_within_l]))
        drift = a_cross - a_within

        per_word[w] = {
            "group": g,
            "n_early": n_e,
            "n_late": len(late),
            "apd_cross": a_cross,
            "apd_within_early": a_within_e,
            "apd_within_late": a_within_l,
            "apd_within": a_within,
            "drift_score": drift,
        }
        print(
            f"[{i}/{len(words)}] {w}: cross={a_cross:.4f} "
            f"within={a_within:.4f} drift={drift:+.4f}"
        )

    group_agg = {}
    for g in GROUP_ORDER:
        vals = [p for p in per_word.values() if p["group"] == g]
        cross = np.array([p["apd_cross"] for p in vals], dtype=float)
        within = np.array([p["apd_within"] for p in vals], dtype=float)
        drift = np.array([p["drift_score"] for p in vals], dtype=float)
        group_agg[g] = {
            "n_words": int(len(vals)),
            "mean_apd_cross": float(cross.mean()) if len(cross) else None,
            "mean_apd_within": float(within.mean()) if len(within) else None,
            "mean_drift_score": float(drift.mean()) if len(drift) else None,
            "frac_with_positive_drift": float((drift > 0).mean()) if len(drift) else None,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name,
        "min_bin": args.min_bin,
        "max_per_bin": args.max_per_bin,
        "n_splits": args.n_splits,
        "per_word": per_word,
        "group_summary": group_agg,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Drift diagnostic: cross-bin vs within-bin APD",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"Model: `{args.model_name}`; min-bin={args.min_bin}; "
        f"max-per-bin={args.max_per_bin}; n-splits={args.n_splits}.",
        "",
        "drift_score = apd_cross - mean(apd_within_early, apd_within_late). "
        "Values near 0 mean the cross-bin distance is no larger than the distance "
        "between two random halves of the same bin -> no evidence of drift "
        "beyond within-word variance / polysemy.",
        "",
        "## Group summary",
        "",
        "| Group | n | mean cross APD | mean within APD | mean drift | frac(drift>0) |",
        "|-------|---:|---------------:|----------------:|-----------:|--------------:|",
    ]
    for g in GROUP_ORDER:
        gs = group_agg[g]
        if gs["n_words"] == 0:
            continue
        lines.append(
            f"| {GROUP_LABELS[g]} | {gs['n_words']} | {gs['mean_apd_cross']:.4f} | "
            f"{gs['mean_apd_within']:.4f} | {gs['mean_drift_score']:+.4f} | "
            f"{gs['frac_with_positive_drift']:.2f} |"
        )

    lines.extend(["", "## Per-word", "",
                  "| Word | Group | cross APD | within APD | drift |",
                  "|------|-------|----------:|-----------:|------:|"])
    for w in sorted(per_word.keys(), key=lambda x: -per_word[x]["drift_score"]):
        p = per_word[w]
        lines.append(
            f"| {w} | {p['group']} | {p['apd_cross']:.4f} | "
            f"{p['apd_within']:.4f} | {p['drift_score']:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT_JSON)
    print("Wrote", OUT_MD)

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(FIG_FACE)

    all_x: list[float] = []
    all_y: list[float] = []
    for g in GROUP_ORDER:
        gs = [p for w, p in per_word.items() if p["group"] == g]
        if not gs:
            continue
        x = np.array([p["apd_within"] for p in gs])
        y = np.array([p["apd_cross"] for p in gs])
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        ax.scatter(
            x, y, s=48, c=GROUP_COLORS[g], edgecolors="#1a1a2e", linewidths=0.4,
            alpha=0.85, label=f"{GROUP_LABELS[g]} (n={len(gs)})",
        )

    lo = float(min(min(all_x), min(all_y)))
    hi = float(max(max(all_x), max(all_y)))
    pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#555", linestyle="--", linewidth=1.0, zorder=1, label="y = x (no drift)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    top_drift = sorted(per_word.items(), key=lambda kv: -kv[1]["drift_score"])[:8]
    for w, p in top_drift:
        ax.annotate(
            w,
            xy=(p["apd_within"], p["apd_cross"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#1a1a2e",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#888", "alpha": 0.85, "linewidth": 0.6},
            zorder=5,
        )

    ax.set_xlabel("Within-bin APD (mean of early/early and late/late halves)")
    ax.set_ylabel("Cross-bin APD (early vs late)")
    ax.set_title(f"Drift diagnostic: per-word cross vs within-bin APD ({args.model_name})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    scatter_path = OUT_FIG_DIR / "drift_diagnostic_scatter.png"
    fig.savefig(scatter_path, dpi=180, facecolor=FIG_FACE)
    plt.close(fig)
    print("Wrote", scatter_path)


if __name__ == "__main__":
    main()

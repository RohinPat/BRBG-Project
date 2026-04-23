"""
Compute RoBERTa APD metrics for early vs late Reddit usage by target word.

Reads the merged JSONL and writes summary outputs under `results/`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np

from reddit_sample_utils import (
    EARLY_YEARS,
    LATE_YEARS,
    effective_calendar_year,
    row_lexicon_key,
    time_bin_for_year,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_JSONL = ROOT / "data" / "reddit_sample" / "reddit_sample_comments.jsonl"
TARGET_JSON = ROOT / "data" / "target_words.json"
OUT_DIR = ROOT / "results"
OUT_JSON = OUT_DIR / "reddit_roberta_apd.json"
OUT_MD = OUT_DIR / "reddit_roberta_apd.md"

MODEL_NAME = "roberta-base"
MAX_LEN = 256
MAX_PER_BIN = 120
MIN_BIN_FOR_GROUP_STATS = 20
BATCH_SIZE = 16
BASE_SEED = 4120
SEED = BASE_SEED

GROUP_KEYS = ("brainrot", "general_slang", "standard_control")


def install_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch", "transformers", "scipy", "-q"]
        )


def load_targets() -> dict[str, list[str]]:
    with open(TARGET_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    out: dict[str, list[str]] = {}
    for k in GROUP_KEYS:
        out[k] = list(cfg.get(k, []))
    return out


def word_to_group(targets: dict[str, list[str]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for g, words in targets.items():
        for w in words:
            m[normalize_query(w)] = g
    return m


def normalize_query(q: str) -> str:
    return (q or "").strip().lower()


def word_pattern(query: str):
    q = query.strip()
    if " " in q:
        return re.compile(re.escape(q), re.IGNORECASE)
    return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)


def find_first_span(text: str, pat: re.Pattern) -> tuple[int, int] | None:
    m = pat.search(text)
    if not m:
        return None
    return m.start(), m.end()


def load_comments_by_word(targets: dict[str, list[str]]) -> dict[str, list[dict]]:
    allowed = set(word_to_group(targets).keys())
    by_word: dict[str, list[dict]] = defaultdict(list)
    with open(DATA_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mq = normalize_query(row_lexicon_key(row))
            if mq not in allowed:
                continue
            year = effective_calendar_year(row)
            if year is None:
                continue
            bin_name = time_bin_for_year(year)
            if bin_name is None:
                continue
            body = row.get("body") or ""
            pat = word_pattern(row.get("matched_query") or row_lexicon_key(row))
            if find_first_span(body, pat) is None:
                continue
            by_word[mq].append({"body": body, "year": year, "bin": bin_name})
    return dict(by_word)


def subsample(rows: list[dict], k: int) -> list[dict]:
    if len(rows) <= k:
        return rows
    idx = np.random.choice(len(rows), size=k, replace=False)
    return [rows[i] for i in idx]


def apd_cosine(early: np.ndarray, late: np.ndarray) -> float:
    if early.shape[0] == 0 or late.shape[0] == 0:
        return float("nan")
    sim = early @ late.T
    dist = 1.0 - sim
    return float(dist.mean())


def mann_whitney_u(x: np.ndarray, y: np.ndarray):
    from scipy.stats import mannwhitneyu

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 1 or len(y) < 1:
        return float("nan"), float("nan")
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(stat), float(p)


def extract_embeddings_batched(
    texts: list[str],
    queries: list[str],
    tokenizer,
    model,
    device,
) -> np.ndarray:
    import torch

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_t = texts[i : i + BATCH_SIZE]
        batch_q = queries[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch_t,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = enc.pop("offset_mapping").to(device)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state

        for b in range(hidden.size(0)):
            text = batch_t[b]
            q = batch_q[b]
            pat = word_pattern(q)
            span = find_first_span(text, pat)
            if span is None:
                all_embs.append(np.zeros(hidden.size(-1), dtype=np.float32))
                continue
            start_c, end_c = span
            offs = offset_mapping[b]
            token_indices = []
            for ti, (s, e) in enumerate(offs.tolist()):
                if s == e == 0:
                    continue
                if e <= start_c or s >= end_c:
                    continue
                if s < end_c and e > start_c:
                    token_indices.append(ti)
            if not token_indices:
                all_embs.append(np.zeros(hidden.size(-1), dtype=np.float32))
                continue
            vec = hidden[b, token_indices].mean(dim=0).cpu().numpy().astype(np.float32)
            nrm = np.linalg.norm(vec)
            if nrm > 1e-8:
                vec = vec / nrm
            all_embs.append(vec)
    return np.stack(all_embs, axis=0)


def qualifies(p: dict, min_bin: int) -> bool:
    a = p.get("apd")
    if a is None:
        return False
    if isinstance(a, float) and a != a:
        return False
    return p["n_early"] >= min_bin and p["n_late"] >= min_bin


def run_one_replicate(
    targets: dict[str, list[str]],
    by_word: dict[str, list[dict]],
    tokenizer,
    model,
    device,
    min_bin: int,
) -> dict[str, dict]:
    w2g = word_to_group(targets)
    per_word: dict[str, dict] = {}
    for group_name in GROUP_KEYS:
        for w in targets[group_name]:
            key = normalize_query(w)
            rows = by_word.get(key, [])
            early_rows = [r for r in rows if r["bin"] == "early"]
            late_rows = [r for r in rows if r["bin"] == "late"]
            early_rows = subsample(early_rows, MAX_PER_BIN)
            late_rows = subsample(late_rows, MAX_PER_BIN)
            texts = [r["body"] for r in early_rows + late_rows]
            queries = [w] * len(texts)
            if not early_rows or not late_rows:
                per_word[key] = {
                    "group": group_name,
                    "n_early": len(early_rows),
                    "n_late": len(late_rows),
                    "apd": None,
                    "skip": "missing bin",
                }
                continue
            embs = extract_embeddings_batched(texts, queries, tokenizer, model, device)
            n_e = len(early_rows)
            early_e = embs[:n_e]
            late_e = embs[n_e:]
            apd = apd_cosine(early_e, late_e)
            per_word[key] = {
                "group": group_name,
                "n_early": len(early_rows),
                "n_late": len(late_rows),
                "apd": apd,
                "skip": None,
            }
    return per_word


def group_apd_array(per_word: dict[str, dict], group: str, min_bin: int) -> np.ndarray:
    vals = [
        p["apd"]
        for p in per_word.values()
        if p["group"] == group and qualifies(p, min_bin)
    ]
    return np.array(vals, dtype=float)


def pairwise_mwu(per_word: dict[str, dict], min_bin: int) -> dict[str, dict]:
    out = {}
    groups = list(GROUP_KEYS)
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            xa = group_apd_array(per_word, a, min_bin)
            xb = group_apd_array(per_word, b, min_bin)
            key = f"{a}_vs_{b}"
            if len(xa) >= 2 and len(xb) >= 2:
                stat, p = mann_whitney_u(xa, xb)
            else:
                stat, p = None, None
            out[key] = {"u_statistic": stat, "p_two_sided": p, "n_a": len(xa), "n_b": len(xb)}
    return out


def parse_args():
    p = argparse.ArgumentParser(description="RoBERTa early/late APD on Reddit JSONL")
    p.add_argument(
        "--max-per-bin",
        type=int,
        default=MAX_PER_BIN,
        help="Max comments per time bin per word",
    )
    p.add_argument(
        "--min-bin",
        type=int,
        default=MIN_BIN_FOR_GROUP_STATS,
        help="Min early and late n required for group stats",
    )
    p.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Transformers model id or local path (e.g. UD fine-tuned checkpoint)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    global MAX_PER_BIN
    MAX_PER_BIN = args.max_per_bin
    min_bin = args.min_bin
    model_name = args.model_name

    install_deps()
    import torch
    from transformers import AutoModel, AutoTokenizer

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    targets = load_targets()
    by_word = load_comments_by_word(targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    final_per_word = run_one_replicate(
        targets, by_word, tokenizer, model, device, min_bin
    )

    group_agg = {}
    for g in GROUP_KEYS:
        arr = group_apd_array(final_per_word, g, min_bin)
        group_agg[g] = {
            "mean_apd_balanced": float(arr.mean()) if len(arr) else None,
            "n_words_balanced": int(len(arr)),
        }

    last_mw = pairwise_mwu(final_per_word, min_bin)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "data": str(DATA_JSONL),
        "max_per_bin": MAX_PER_BIN,
        "early_years": list(EARLY_YEARS),
        "late_years": list(LATE_YEARS),
        "min_bin_for_group_stats": min_bin,
        "per_word": final_per_word,
        "group_summary": group_agg,
        "pairwise_mannwhitney": last_mw,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Reddit: RoBERTa APD (early vs late)",
        "",
        f"*Generated: {gen_ts}*",
        "",
        f"Model: `{model_name}`",
        f"Source: `{DATA_JSONL}`",
        f"Early years: {list(EARLY_YEARS)}; late years: {list(LATE_YEARS)}",
        f"Max comments per bin: {MAX_PER_BIN}; min per bin for group stats: {min_bin}",
        "",
        "## Per-word APD",
        "",
        "| Word | Group | n_early | n_late | APD |",
        "|------|-------|---------|--------|-----|",
    ]
    for w in sorted(final_per_word.keys()):
        p = final_per_word[w]
        apd_s = f"{p['apd']:.4f}" if p.get("apd") is not None and p["apd"] == p["apd"] else "—"
        lines.append(
            f"| {w} | {p['group']} | {p['n_early']} | {p['n_late']} | {apd_s} |"
        )
    lines.extend(["", "## Group means (balanced words)"])
    for g in GROUP_KEYS:
        gs = group_agg[g]
        lines.append(
            f"- **{g}**: n={gs['n_words_balanced']}, mean APD={gs['mean_apd_balanced']}"
        )
    lines.extend(["", "## Pairwise Mann–Whitney U", ""])
    for k, v in last_mw.items():
        lines.append(
            f"- `{k}`: n_a={v['n_a']}, n_b={v['n_b']}, "
            f"stat={v['u_statistic']}, p={v['p_two_sided']}"
        )
    lines.append("\nNote: exploratory p-values; add more data for stable inference.")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT_JSON)
    print("Wrote", OUT_MD)


if __name__ == "__main__":
    main()

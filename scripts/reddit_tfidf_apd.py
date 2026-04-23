"""
Static baseline: TF-IDF vectors + same APD metric as RoBERTa (early vs late bins).
Three groups: brainrot, general_slang, standard_control.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
OUT_JSON = ROOT / "results" / "reddit_tfidf_apd.json"

MAX_PER_BIN = 120
MIN_BIN = 20
SEED = 4120

GROUP_KEYS = ("brainrot", "general_slang", "standard_control")


def install():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])


def normalize_query(q: str) -> str:
    return (q or "").strip().lower()


def word_pattern(query: str):
    q = query.strip()
    if " " in q:
        return re.compile(re.escape(q), re.IGNORECASE)
    return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)


def find_first_span(text: str, pat: re.Pattern):
    m = pat.search(text)
    return (m.start(), m.end()) if m else None


def load_targets() -> dict[str, list[str]]:
    with open(TARGET_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: list(cfg.get(k, [])) for k in GROUP_KEYS}


def word_to_group(targets: dict[str, list[str]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for g, words in targets.items():
        for w in words:
            m[normalize_query(w)] = g
    return m


def load_comments(targets: dict[str, list[str]]):
    allowed = set(word_to_group(targets).keys())
    bw = defaultdict(list)
    with open(DATA_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mq = normalize_query(row_lexicon_key(row))
            if mq not in allowed:
                continue
            y = effective_calendar_year(row)
            if y is None:
                continue
            bin_name = time_bin_for_year(y)
            if bin_name is None:
                continue
            body = row.get("body") or ""
            if find_first_span(
                body, word_pattern(row.get("matched_query") or row_lexicon_key(row))
            ) is None:
                continue
            bw[mq].append({"body": body, "bin": bin_name})
    return dict(bw)


def subsample(rows, k, rng):
    if len(rows) <= k:
        return rows
    idx = rng.choice(len(rows), size=k, replace=False)
    return [rows[i] for i in idx]


def apd_cosine(early, late):
    if early.shape[0] == 0 or late.shape[0] == 0:
        return float("nan")
    early = early / (np.linalg.norm(early, axis=1, keepdims=True) + 1e-9)
    late = late / (np.linalg.norm(late, axis=1, keepdims=True) + 1e-9)
    dist = 1.0 - (early @ late.T)
    return float(dist.mean())


def main():
    install()
    from scipy.stats import mannwhitneyu
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = np.random.default_rng(SEED)
    targets = load_targets()
    by_word = load_comments(targets)

    all_texts: list[str] = []
    spans: list[tuple[str, str, int, int, int, int]] = []
    per_word: dict = {}

    for group_name in GROUP_KEYS:
        for w in targets[group_name]:
            key = normalize_query(w)
            rows = by_word.get(key, [])
            early = subsample([r for r in rows if r["bin"] == "early"], MAX_PER_BIN, rng)
            late = subsample([r for r in rows if r["bin"] == "late"], MAX_PER_BIN, rng)
            if not early or not late:
                per_word[key] = {
                    "group": group_name,
                    "n_early": len(early),
                    "n_late": len(late),
                    "apd": None,
                }
                continue
            i0 = len(all_texts)
            all_texts.extend(r["body"] for r in early)
            i1 = len(all_texts)
            all_texts.extend(r["body"] for r in late)
            i2 = len(all_texts)
            spans.append((key, group_name, i0, i1, i1, i2))

    vec = TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1, 2))
    X = vec.fit_transform(all_texts).toarray().astype(np.float32)

    for key, group_name, e0, e1, l0, l1 in spans:
        apd = apd_cosine(X[e0:e1], X[l0:l1])
        per_word[key] = {
            "group": group_name,
            "n_early": e1 - e0,
            "n_late": l1 - l0,
            "apd": apd,
        }

    def ok(p):
        a = p.get("apd")
        if a is None or (isinstance(a, float) and a != a):
            return False
        return p["n_early"] >= MIN_BIN and p["n_late"] >= MIN_BIN

    def grp_arr(g):
        return np.array(
            [p["apd"] for p in per_word.values() if p["group"] == g and ok(p)],
            dtype=float,
        )

    group_means = {}
    for g in GROUP_KEYS:
        arr = grp_arr(g)
        group_means[g] = float(arr.mean()) if len(arr) else None

    pairwise = {}
    groups = list(GROUP_KEYS)
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            xa, xb = grp_arr(a), grp_arr(b)
            key = f"{a}_vs_{b}"
            if len(xa) >= 2 and len(xb) >= 2:
                stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
                pairwise[key] = {
                    "u_statistic": float(stat),
                    "p_two_sided": float(p),
                    "n_a": len(xa),
                    "n_b": len(xb),
                }
            else:
                pairwise[key] = {
                    "u_statistic": None,
                    "p_two_sided": None,
                    "n_a": len(xa),
                    "n_b": len(xb),
                }

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "tfidf_sentence_vector",
        "data": str(DATA_JSONL),
        "max_per_bin": MAX_PER_BIN,
        "min_bin_for_group_stats": MIN_BIN,
        "max_features": 8000,
        "ngram_range": [1, 2],
        "early_years": list(EARLY_YEARS),
        "late_years": list(LATE_YEARS),
        "per_word": per_word,
        "group_mean_apd_balanced": group_means,
        "pairwise_mannwhitney": pairwise,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()

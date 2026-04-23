"""
fastText-style baseline (gensim): train FastText on subsampled comments per word,
encode each comment as mean of word vectors, same APD as RoBERTa/TF-IDF.
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
OUT_JSON = ROOT / "results" / "reddit_fasttext_apd.json"

MAX_PER_BIN = 120
MIN_BIN = 20
SEED = 4120
GROUP_KEYS = ("brainrot", "general_slang", "standard_control")


def install():
    try:
        import gensim  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "gensim", "-q"])


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


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (s or "").lower())


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
            b = time_bin_for_year(y)
            if b is None:
                continue
            body = row.get("body") or ""
            if find_first_span(
                body, word_pattern(row.get("matched_query") or row_lexicon_key(row))
            ) is None:
                continue
            bw[mq].append({"body": body, "bin": b})
    return dict(bw)


def subsample(rows, k, rng):
    if len(rows) <= k:
        return rows
    idx = rng.choice(len(rows), size=k, replace=False)
    return [rows[i] for i in idx]


def apd_cosine(early: np.ndarray, late: np.ndarray) -> float:
    if early.shape[0] == 0 or late.shape[0] == 0:
        return float("nan")
    early = early / (np.linalg.norm(early, axis=1, keepdims=True) + 1e-9)
    late = late / (np.linalg.norm(late, axis=1, keepdims=True) + 1e-9)
    return float((1.0 - (early @ late.T)).mean())


def sentence_vec(model, text: str) -> np.ndarray:
    toks = tokenize(text)
    if not toks:
        return np.zeros(model.vector_size, dtype=np.float32)
    vecs = []
    for t in toks:
        try:
            vecs.append(model.wv.get_vector(t, norm=False))
        except KeyError:
            continue
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def main():
    install()
    from gensim.models import FastText
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(SEED)
    targets = load_targets()
    by_word = load_comments(targets)
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
            sents = [tokenize(r["body"]) for r in early + late]
            if sum(len(s) for s in sents) < 10:
                per_word[key] = {
                    "group": group_name,
                    "n_early": len(early),
                    "n_late": len(late),
                    "apd": None,
                }
                continue
            model = FastText(
                vector_size=100,
                window=5,
                min_count=1,
                workers=1,
                seed=SEED,
                epochs=12,
            )
            model.build_vocab(corpus_iterable=sents)
            model.train(
                corpus_iterable=sents,
                total_examples=len(sents),
                epochs=model.epochs,
            )
            texts = [r["body"] for r in early + late]
            embs = np.stack([sentence_vec(model, t) for t in texts], axis=0)
            ne = len(early)
            apd = apd_cosine(embs[:ne], embs[ne:])
            per_word[key] = {
                "group": group_name,
                "n_early": ne,
                "n_late": len(late),
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

    group_means = {g: float(np.mean(grp_arr(g))) if len(grp_arr(g)) else None for g in GROUP_KEYS}
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
        "method": "gensim_fasttext_mean_word_vectors",
        "data": str(DATA_JSONL),
        "max_per_bin": MAX_PER_BIN,
        "min_bin_for_group_stats": MIN_BIN,
        "vector_size": 100,
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

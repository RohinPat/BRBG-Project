"""Write results/corpus_coverage.json: per-target early/late counts from JSONL + target_words.json."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from reddit_sample_utils import effective_calendar_year, row_lexicon_key, time_bin_for_year

ROOT = Path(__file__).resolve().parents[1]
DATA_JSONL = ROOT / "data" / "reddit_sample" / "reddit_sample_comments.jsonl"
TARGET_JSON = ROOT / "data" / "target_words.json"
OUT_JSON = ROOT / "results" / "corpus_coverage.json"
GROUP_KEYS = ("brainrot", "general_slang", "standard_control")


def normalize(q: str) -> str:
    return (q or "").strip().lower()


def normalize_sub(s: str) -> str:
    t = (s or "").strip()
    return t.lower() if t else "(unknown)"


def load_targets():
    with open(TARGET_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    w2g = {}
    for g in GROUP_KEYS:
        for w in cfg.get(g, []):
            w2g[normalize(w)] = g
    return w2g


def main():
    w2g = load_targets()
    early: dict[str, int] = defaultdict(int)
    late: dict[str, int] = defaultdict(int)
    total_lines = 0
    sub_early: dict[str, int] = defaultdict(int)
    sub_late: dict[str, int] = defaultdict(int)
    word_sub_early: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    word_sub_late: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if DATA_JSONL.exists():
        with open(DATA_JSONL, encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                row = json.loads(line)
                mq = normalize(row_lexicon_key(row))
                if mq not in w2g:
                    continue
                y = effective_calendar_year(row)
                if y is None:
                    continue
                b = time_bin_for_year(y)
                sub = normalize_sub(row.get("subreddit"))
                if b == "early":
                    early[mq] += 1
                    sub_early[sub] += 1
                    word_sub_early[mq][sub] += 1
                elif b == "late":
                    late[mq] += 1
                    sub_late[sub] += 1
                    word_sub_late[mq][sub] += 1

    per_word = {}
    for mq, g in sorted(w2g.items()):
        ne, nl = early[mq], late[mq]
        se = {k: int(v) for k, v in sorted(word_sub_early[mq].items()) if v > 0}
        sl = {k: int(v) for k, v in sorted(word_sub_late[mq].items()) if v > 0}
        per_word[mq] = {
            "group": g,
            "n_early": ne,
            "n_late": nl,
            "both_bins_ge_20": ne >= 20 and nl >= 20,
            "both_bins_ge_30": ne >= 30 and nl >= 30,
            "both_bins_ge_50": ne >= 50 and nl >= 50,
            "both_bins_ge_120": ne >= 120 and nl >= 120,
            "by_subreddit_early": se,
            "by_subreddit_late": sl,
        }

    n20 = sum(1 for p in per_word.values() if p["both_bins_ge_20"])
    n30 = sum(1 for p in per_word.values() if p["both_bins_ge_30"])
    n100 = sum(1 for p in per_word.values() if p["both_bins_ge_50"])
    n_any_early = sum(1 for p in per_word.values() if p["n_early"] > 0)
    n_any_late = sum(1 for p in per_word.values() if p["n_late"] > 0)

    agg_sub = {}
    for sub in sorted(set(sub_early) | set(sub_late)):
        agg_sub[sub] = {
            "early": int(sub_early[sub]),
            "late": int(sub_late[sub]),
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_path": str(DATA_JSONL),
        "jsonl_lines_total": total_lines,
        "target_words_count": len(w2g),
        "words_with_early_gt_0": n_any_early,
        "words_with_late_gt_0": n_any_late,
        "words_with_both_bins_ge_20": n20,
        "words_with_both_bins_ge_30": n30,
        "words_with_both_bins_ge_50": n100,
        "aggregate_by_subreddit_bin": agg_sub,
        "per_word": per_word,
        "note": "Counts use created_utc calendar year and early/late bins; lexicon key prefers canonical_query when set (query variants). per_word.by_subreddit_* stratifies matched rows.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()

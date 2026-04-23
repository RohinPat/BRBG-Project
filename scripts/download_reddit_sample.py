"""
Download Reddit comments from Pullpush and write deduped JSONL output.

Config comes from `data/pullpush_config.json`, `data/target_words.json`,
and optional `data/query_variants.json`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from reddit_sample_utils import effective_calendar_year, row_lexicon_key

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "reddit_sample"
OUT_FILE = OUT_DIR / "reddit_sample_comments.jsonl"
SUMMARY_FILE = OUT_DIR / "reddit_sample_summary.json"
TARGET_JSON = ROOT / "data" / "target_words.json"
PULLPUSH_CONFIG = ROOT / "data" / "pullpush_config.json"
QUERY_VARIANTS_JSON = ROOT / "data" / "query_variants.json"

BASE_URL = "https://api.pullpush.io/reddit/search/comment/"
DEFAULT_SUBREDDITS = ["GenZ", "teenagers", "memes"]
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
CAP_PER_WORD_YEAR = 1000
DEFAULT_SIZE = 100
DEFAULT_SLEEP_OK = 1.15
FAST_SLEEP_OK = 0.25
DEFAULT_COOLDOWN_AFTER_429 = 900.0
MAX_RETRIES = 8
BASE_BACKOFF_SEC = 3.0
MAX_BACKOFF_SEC = 180.0
BACKOFF_MULTIPLIER = 2.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _line_buffer_stdout() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def normalize_word_key(w: str) -> str:
    return (w or "").strip().lower()


def load_subreddits_arg(arg: str | None) -> list[str]:
    """CLI --subreddits comma list overrides data/pullpush_config.json and defaults."""
    if arg:
        return [s.strip() for s in arg.split(",") if s.strip()]
    if PULLPUSH_CONFIG.exists():
        with open(PULLPUSH_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        subs = cfg.get("subreddits")
        if isinstance(subs, list) and subs:
            return [str(s).strip() for s in subs if str(s).strip()]
    return list(DEFAULT_SUBREDDITS)


def load_variant_map() -> dict[str, list[str]]:
    """Lowercase canonical key -> non-empty Pullpush q strings."""
    if not QUERY_VARIANTS_JSON.exists():
        return {}
    with open(QUERY_VARIANTS_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg.get("by_canonical") or {}
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        kk = normalize_word_key(str(k))
        if not kk or not isinstance(v, list):
            continue
        qs = [str(x).strip() for x in v if str(x).strip()]
        if qs:
            out[kk] = qs
    return out


def queries_for_target_word(word: str, variant_map: dict[str, list[str]]) -> list[str]:
    """Ordered Pullpush q strings for one lexicon row (deduped)."""
    wk = normalize_word_key(word)
    base = (word or "").strip()
    qs = list(variant_map.get(wk, []))
    seen: set[str] = set()
    out: list[str] = []
    if base:
        bl = normalize_word_key(base)
        seen.add(bl)
        out.append(base)
    for q in qs:
        ql = normalize_word_key(q)
        if ql in seen:
            continue
        seen.add(ql)
        out.append(q.strip())
    return out if out else ([base] if base else [])


def count_total_fetch_jobs(
    year_months: list[tuple[int, int]],
    subreddits: list[str],
    target_words: list[str],
    variant_map: dict[str, list[str]],
) -> int:
    n = 0
    for _y, _m in year_months:
        for _s in subreddits:
            for w in target_words:
                n += len(queries_for_target_word(w, variant_map))
    return max(n, 1)


def load_target_words():
    with open(TARGET_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    out = []
    for key in ("brainrot", "general_slang", "standard_control"):
        out.extend(cfg.get(key, []))
    seen = set()
    deduped = []
    for w in out:
        s = (w or "").strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        deduped.append(s)
    return deduped


def to_epoch(year, month, day=1):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def month_window(year, month):
    start = to_epoch(year, month, 1)
    if month == 12:
        end = to_epoch(year + 1, 1, 1)
    else:
        end = to_epoch(year, month + 1, 1)
    return start, end


def utc_ym(ts) -> tuple[int, int]:
    if ts is None:
        return 0, 0
    t = float(ts)
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    return dt.year, dt.month


def _retry_after_seconds(headers) -> float | None:
    """Parse Retry-After (seconds only). Returns None if missing or not an integer."""
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None or raw == "":
        return None
    try:
        sec = int(str(raw).strip())
    except ValueError:
        return None
    return float(max(0, min(sec, int(MAX_BACKOFF_SEC))))


def _sleep_backoff(
    attempt: int,
    *,
    status: int | None = None,
    retry_after: float | None = None,
) -> None:
    raw = min(
        MAX_BACKOFF_SEC,
        BASE_BACKOFF_SEC * (BACKOFF_MULTIPLIER**attempt),
    )
    jitter = random.uniform(0, min(2.0, raw * 0.15))
    delay = raw + jitter
    if retry_after is not None:
        delay = max(delay, retry_after)
    delay = min(delay, MAX_BACKOFF_SEC)
    tag = f" HTTP {status}" if status is not None else ""
    ra = f", Retry-After={retry_after:.0f}s" if retry_after is not None else ""
    print(
        f"  Backing off {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES}){tag}{ra}...",
        flush=True,
    )
    time.sleep(delay)


def fetch_comments(subreddit, after_ts, before_ts, q, *, size: int):
    params = {
        "subreddit": subreddit,
        "after": after_ts,
        "before": before_ts,
        "size": size,
        "sort": "desc",
        "sort_type": "created_utc",
        "q": q,
    }
    url = BASE_URL + "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "cs4120-checkin-bot/1.0"})

    last_error = None
    for attempt in range(MAX_RETRIES):
        print(
            f"  HTTP try {attempt + 1}/{MAX_RETRIES} (timeout 45s)…",
            flush=True,
        )
        try:
            with urlopen(req, timeout=45) as resp:
                body = resp.read().decode("utf-8")
                payload = json.loads(body)
            data = payload.get("data", [])
            print(f"  OK — {len(data)} comments returned", flush=True)
            return data
        except HTTPError as e:
            last_error = e
            code = e.code
            try:
                e.read()
            except Exception:
                pass
            if code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                ra = _retry_after_seconds(e.headers) if code == 429 else None
                _sleep_backoff(attempt, status=code, retry_after=ra)
                continue
            raise
        except URLError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                _sleep_backoff(attempt)
                continue
            raise
        except (TimeoutError, OSError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                _sleep_backoff(attempt)
                continue
            raise

    if last_error:
        raise last_error
    return []


def rebuild_state(rows: list[dict]):
    counts: dict[tuple[str, int], int] = defaultdict(int)
    seen_ids: set = set()
    for r in rows:
        rid = r.get("id")
        if rid:
            seen_ids.add(rid)
        wk = normalize_word_key(row_lexicon_key(r))
        y = effective_calendar_year(r)
        if wk and y is not None:
            counts[(wk, y)] += 1
    return counts, seen_ids


def load_existing_rows():
    if not OUT_FILE.exists():
        return []
    out = []
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_dedup_jsonl(rows: list[dict], path: Path, *, verbose: bool = True) -> None:
    dedup = {}
    for r in rows:
        rid = r.get("id")
        if rid and rid not in dedup:
            dedup[rid] = r
    final_rows = list(dedup.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    if verbose:
        print(f"Checkpoint: {len(final_rows)} rows -> {path}", flush=True)


def _ingest_comment_batch(
    comments: list,
    search_query: str,
    canonical_word: str,
    wkey: str,
    year: int,
    month: int,
    rows: list,
    counts: dict,
    seen_ids: set,
    counts_increment_total: Counter,
    by_year: Counter,
    by_subreddit: Counter,
    by_word: Counter,
) -> None:
    for c in comments:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        rid = c.get("id")
        if rid and rid in seen_ids:
            continue
        cy, cm = utc_ym(c.get("created_utc"))
        if cy <= 0:
            cy, cm = year, month
        if counts.get((wkey, cy), 0) >= CAP_PER_WORD_YEAR:
            continue
        row = {
            "id": rid,
            "subreddit": c.get("subreddit"),
            "author": c.get("author"),
            "created_utc": c.get("created_utc"),
            "body": body,
            "matched_query": search_query,
            "canonical_query": canonical_word,
            "year": cy,
            "month": cm,
        }
        rows.append(row)
        if rid:
            seen_ids.add(rid)
        counts[(wkey, cy)] = counts.get((wkey, cy), 0) + 1
        counts_increment_total["added"] += 1
        by_year[str(cy)] += 1
        by_subreddit[row["subreddit"] or ""] += 1
        by_word[canonical_word] += 1


def main():
    p = argparse.ArgumentParser(description="Pullpush Reddit comment crawl for CS4120 lexicon.")
    p.add_argument(
        "--sleep-between",
        type=float,
        default=None,
        metavar="SEC",
        help=f"Pause after each OK request. Default {DEFAULT_SLEEP_OK}, or {FAST_SLEEP_OK} with --fast.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=f"Shortcut for --sleep-between {FAST_SLEEP_OK} (more 429/backoff if API is strict).",
    )
    p.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Pullpush size= (max comments per request). Try 100–250 if the API accepts it.",
    )
    p.add_argument(
        "--start-pause",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Sleep this many seconds before the first HTTP request (use after heavy 429s, e.g. 300).",
    )
    p.add_argument(
        "--cooldown-on-429",
        type=float,
        default=DEFAULT_COOLDOWN_AFTER_429,
        metavar="SEC",
        help=f"After a query exhausts retries on HTTP 429, sleep this long before the next query "
        f"(default {DEFAULT_COOLDOWN_AFTER_429:.0f}s). Use 0 to disable.",
    )
    p.add_argument(
        "--subreddits",
        default=None,
        metavar="CSV",
        help="Override subreddit list (comma-separated). Default: data/pullpush_config.json or "
        "GenZ,teenagers,memes.",
    )
    args = p.parse_args()
    _line_buffer_stdout()
    sleep_between = args.sleep_between
    if sleep_between is None:
        sleep_between = FAST_SLEEP_OK if args.fast else DEFAULT_SLEEP_OK
    size = max(1, min(args.size, 500))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_words = load_target_words()
    if not target_words:
        raise SystemExit(f"No words in {TARGET_JSON}")

    subreddits = load_subreddits_arg(args.subreddits)
    variant_map = load_variant_map()
    print(
        "Subreddits:",
        ", ".join(subreddits),
        f"| variant map: {len(variant_map)} canonicals" if variant_map else "| variant map: (none)",
        flush=True,
    )

    months = list(range(1, 13))
    rows = load_existing_rows()
    counts, seen_ids = rebuild_state(rows)

    counts_increment_total = Counter()
    by_year = Counter()
    by_subreddit = Counter()
    by_word = Counter()

    if args.start_pause > 0:
        print(f"Start pause {args.start_pause:.0f}s before first request...", flush=True)
        time.sleep(args.start_pause)

    year_months = [(y, m) for y in YEARS for m in months]
    total_queries = count_total_fetch_jobs(year_months, subreddits, target_words, variant_map)
    fetch_idx = 0

    for year, month in year_months:
        after_ts, before_ts = month_window(year, month)
        for subreddit in subreddits:
            for word in target_words:
                wkey = normalize_word_key(word)
                if counts.get((wkey, year), 0) >= CAP_PER_WORD_YEAR:
                    continue
                for q in queries_for_target_word(word, variant_map):
                    if counts.get((wkey, year), 0) >= CAP_PER_WORD_YEAR:
                        break
                    fetch_idx += 1
                    print(
                        f"[{fetch_idx}/{total_queries}] {subreddit} {year}-{month:02d} "
                        f"canonical={word!r} q={q!r}",
                        flush=True,
                    )
                    had_429_cooldown = False
                    try:
                        comments = fetch_comments(
                            subreddit, after_ts, before_ts, q, size=size
                        )
                    except Exception as exc:
                        print(
                            f"Fetch failed after retries for {subreddit} "
                            f"{year}-{month:02d} {q!r}: {exc}",
                            flush=True,
                        )
                        comments = []
                        if (
                            isinstance(exc, HTTPError)
                            and exc.code == 429
                            and args.cooldown_on_429 > 0
                        ):
                            print(
                                f"  Cooldown {args.cooldown_on_429:.0f}s before next query "
                                f"(only one crawl; IP may need longer — try again tomorrow).",
                                flush=True,
                            )
                            time.sleep(args.cooldown_on_429)
                            had_429_cooldown = True
                    _ingest_comment_batch(
                        comments,
                        q,
                        word,
                        wkey,
                        year,
                        month,
                        rows,
                        counts,
                        seen_ids,
                        counts_increment_total,
                        by_year,
                        by_subreddit,
                        by_word,
                    )
                    if not had_429_cooldown:
                        time.sleep(sleep_between)
        _write_dedup_jsonl(rows, OUT_FILE, verbose=False)

        _write_dedup_jsonl(rows, OUT_FILE)

    dedup = {}
    for r in rows:
        rid = r.get("id")
        if rid and rid not in dedup:
            dedup[rid] = r
    final_rows = list(dedup.values())

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    summary = {
        "source": "pullpush_api_comment_search",
        "subreddits": subreddits,
        "pullpush_config_path": str(PULLPUSH_CONFIG) if PULLPUSH_CONFIG.exists() else None,
        "query_variants_path": str(QUERY_VARIANTS_JSON) if QUERY_VARIANTS_JSON.exists() else None,
        "cap_per_word_per_calendar_year": CAP_PER_WORD_YEAR,
        "years": YEARS,
        "months_per_year": months,
        "target_words_source": str(TARGET_JSON),
        "target_words_count": len(target_words),
        "query_size": size,
        "rows_after_dedup": len(final_rows),
        "counts_by_year": dict(by_year),
        "counts_by_subreddit": dict(by_subreddit),
        "counts_by_query_word": dict(by_word),
        "rows_added_this_run_approx": int(counts_increment_total["added"]),
        "output_jsonl": str(OUT_FILE),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", OUT_FILE, flush=True)
    print("Wrote:", SUMMARY_FILE, flush=True)
    print("Rows after dedup:", len(final_rows), flush=True)


if __name__ == "__main__":
    main()

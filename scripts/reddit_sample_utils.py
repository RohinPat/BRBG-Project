"""Shared helpers for Reddit JSONL rows (proposal-aligned time bins)."""
from __future__ import annotations

from datetime import datetime, timezone

EARLY_YEARS = (2019, 2020, 2021)
LATE_YEARS = (2022, 2023, 2024)


def effective_calendar_year(row: dict) -> int | None:
    """Prefer UTC year from created_utc (Pushshift-style truth); fall back to row['year']."""
    ts = row.get("created_utc")
    if ts is not None:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).year
        except (ValueError, TypeError, OSError):
            pass
    y = row.get("year")
    if y is None:
        return None
    try:
        return int(y)
    except (TypeError, ValueError):
        return None


def time_bin_for_year(y: int | None) -> str | None:
    if y is None:
        return None
    if y in EARLY_YEARS:
        return "early"
    if y in LATE_YEARS:
        return "late"
    return None


def normalize_lexicon_key(q: str) -> str:
    """Lowercase strip for grouping JSONL rows to target_words entries."""
    return (q or "").strip().lower()


def row_lexicon_key(row: dict) -> str:
    """
    Key for lexicon group + per-word stats. Prefer canonical_query (set when Pullpush
    used a query variant); fall back to matched_query for older JSONL rows.
    """
    c = row.get("canonical_query")
    if isinstance(c, str) and c.strip():
        return normalize_lexicon_key(c)
    return normalize_lexicon_key(row.get("matched_query", ""))

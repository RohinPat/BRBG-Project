# Brainrot or Brain Growth? Tracking Semantic Drift of Internet Slang

CS4120 Final Project — Rohin Patel, Pranav Rana, Qihao Lin (Spring 2026).

This repo contains everything needed to reproduce the numbers, tables, and figures in
our CS4120 final paper and presentation. We measure **average pairwise distance
(APD)** between **early (2019–2021)** and **late (2022–2024)** Reddit usage of
100 target strings (50 brainrot / 25 general slang / 25 standard controls)
under three encoders — **RoBERTa**, **TF–IDF**, and **fastText** — and report
a **within-bin null** diagnostic that separates true drift from within-word
variance.

## 1. Setup

Python 3.10+ is recommended. A GPU accelerates the RoBERTa step but is not required.

```bash
git clone https://github.com/RohinPat/NLP-Project.git
cd NLP-Project
pip install -r requirements.txt
```

The merged Reddit corpus (`data/reddit_sample/reddit_sample_comments.jsonl`,
~208 MB, 500,052 comments) is tracked with **Git LFS** because it exceeds
GitHub's 100 MB blob limit. Install LFS once (`git lfs install`) before cloning
so the file is pulled automatically; otherwise the analysis scripts will not
find any input. To rebuild the corpus from scratch instead, see §3.

## 2. Reproduce every paper artifact

Once the corpus JSONL is in place, one command regenerates every result file
and figure cited in the paper and presentation:

```bash
python scripts/refresh_paper_artifacts.py
```

This runs, in order:

1. `report_corpus_coverage.py` &rarr; `results/corpus_coverage.json`
2. `reddit_roberta_apd.py --min-bin 20` &rarr; `results/reddit_roberta_apd.{json,md}`
3. `reddit_tfidf_apd.py` &rarr; `results/reddit_tfidf_apd.json`
4. `reddit_fasttext_apd.py` &rarr; `results/reddit_fasttext_apd.json`
5. `reddit_drift_diagnostic.py` &rarr; `results/reddit_drift_diagnostic.{json,md}` + `results/figures/drift_diagnostic_scatter.png` (**paper Figure 2**)
6. `reddit_umap_all.py` &rarr; `results/figures/umap_all_words/*.png` (one per-word UMAP for every target that passes the inclusion rule; presentation slide 8 draws from this folder)
7. `reddit_results_figures.py` &rarr; **paper Figure 1** (`apd_group_means_by_encoder.png`) plus coverage / frequency / per-word APD plots used in the presentation deck

To run any step in isolation, invoke the corresponding script directly — every
script is idempotent and reads the same JSONL.

## 3. (Optional) Rebuild the Reddit corpus from Pullpush

`scripts/download_reddit_sample.py` queries the public Pullpush Reddit search
API one request at a time (with ~1.15 s pause after each OK response and
`Retry-After`-aware back-off on 429s). Subreddits come from
`data/pullpush_config.json`, targets from `data/target_words.json`, and
optional expansions from `data/query_variants.json`. JSONL is checkpointed at
every calendar-month boundary so long runs resume cleanly.

```bash
python scripts/download_reddit_sample.py                            # default full crawl
python scripts/download_reddit_sample.py --subreddits GenZ,teenagers,memes
python scripts/download_reddit_sample.py --start-pause 300 --cooldown-on-429 1800
```

The merged JSONL used for this submission was generated **2026-04-19 UTC**.

## 4. Repository layout

```
.
|-- README.md                          This file
|-- requirements.txt                   Python dependencies
|-- data/
|   |-- target_words.json              100-term lexicon (brainrot / slang / control)
|   |-- pullpush_config.json           Default subreddit list
|   |-- query_variants.json            Optional query-string expansions per target
|   \-- reddit_sample/
|       |-- reddit_sample_comments.jsonl  Merged deduped corpus (Git LFS)
|       \-- reddit_sample_summary.json    Crawl summary metadata
|-- scripts/
|   |-- download_reddit_sample.py      Pullpush crawl -> JSONL
|   |-- reddit_sample_utils.py         Time-bin / lexicon-key helpers
|   |-- reddit_viz_style.py            Shared matplotlib styling
|   |-- report_corpus_coverage.py      Per-target coverage + subreddit stratification
|   |-- reddit_roberta_apd.py          RoBERTa contextual APD + Mann-Whitney U
|   |-- reddit_tfidf_apd.py            TF-IDF sentence-vector APD baseline
|   |-- reddit_fasttext_apd.py         fastText mean-word-vector APD baseline
|   |-- reddit_drift_diagnostic.py     Within-bin null baseline (cross vs within APD)
|   |-- reddit_umap_all.py             Per-target UMAP PNGs (presentation slide 8)
|   |-- reddit_results_figures.py      Static figures from JSON artifacts
|   \-- refresh_paper_artifacts.py     One-shot reproducer (calls all of the above)
\-- results/
    |-- corpus_coverage.json
    |-- reddit_roberta_apd.{json,md}
    |-- reddit_tfidf_apd.json
    |-- reddit_fasttext_apd.json
    |-- reddit_drift_diagnostic.{json,md}
    \-- figures/
        |-- apd_group_means_by_encoder.png      Paper Figure 1
        |-- drift_diagnostic_scatter.png        Paper Figure 2
        |-- apd_roberta_per_word.png            Presentation figure
        |-- corpus_early_vs_late_counts.png     Presentation figure (coverage)
        |-- comments_per_calendar_year.png      Presentation figure
        |-- frequency_by_year.png               Presentation figure
        |-- sample_volume_stacked_by_group.png  Presentation figure
        |-- umap_roberta_mid.png                Pre-generated UMAP (retained for reference)
        |-- umap_roberta_based.png              Pre-generated UMAP (retained for reference)
        |-- umap_roberta_book.png               Pre-generated UMAP (retained for reference)
        |-- umap_roberta_grid_mid_based_book.png   Pre-generated UMAP panel (retained for reference)
        |-- umap_roberta_rizz.png               Pre-generated UMAP (retained for reference)
        |-- umap_roberta_slay.png               Pre-generated UMAP (retained for reference)
        \-- umap_all_words/                     Per-target UMAPs used in presentation slide 8
```

## 5. Models and datasets

**Models (three encoders compared on the same balanced word set):**

- `roberta-base` (Hugging Face Transformers). Frozen, last-layer token hidden
  states mean-pooled over the target span, L2-normalized before APD.
- **TF-IDF** sentence vectors (`scikit-learn`, 8k features, unigrams + bigrams).
- **fastText** mean-of-word-vectors (gensim, 100-d).

**Dataset:** 500,052 deduplicated Reddit comments crawled via the public
[Pullpush](https://pullpush.io/) API across r/GenZ, r/teenagers, r/memes, and
r/AskReddit for all 72 months of 2019-2024. The file is distributed in this
repo via Git LFS (`data/reddit_sample/reddit_sample_comments.jsonl`). If Git
LFS is unavailable, run `scripts/download_reddit_sample.py` to regenerate it
from the Pullpush API (§3).

## 6. Key result summary

| Group (balanced n) | RoBERTa APD | TF-IDF APD | fastText APD |
|--------------------|-------------|------------|--------------|
| Brainrot (n = 43)        | 0.146 | 0.838 | 7.81e-04 |
| General slang (n = 25)   | 0.188 | 0.859 | 1.85e-04 |
| Standard control (n = 25)| 0.114 | 0.899 | 8.69e-05 |

Mann-Whitney U (two-sided, RoBERTa): general slang vs standard
p ~= 9.1e-04; brainrot vs general slang p ~= 0.018; brainrot vs standard
p ~= 0.80. Per-word cross-bin APD is essentially identical to within-bin APD
(Figure 2), i.e. the observed APD reflects within-word variance more than
genuine diachronic drift. See the final paper for the full discussion.

## 7. Dependencies

Pinned in `requirements.txt`:

- torch >= 2.0
- transformers >= 4.36
- numpy >= 1.24
- scipy >= 1.11
- scikit-learn >= 1.3
- umap-learn >= 0.5
- matplotlib >= 3.8
- gensim >= 4.3

No paid APIs or private credentials are required. The Pullpush endpoint is
public and rate-limited.

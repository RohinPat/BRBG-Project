"""
Run the full artifact pipeline from the merged Reddit JSONL.

Each step can still be run directly via its own script.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str]) -> None:
    print(">>", " ".join(args))
    subprocess.run(args, check=True, cwd=str(ROOT))


def main() -> None:
    scripts = ROOT / "scripts"
    run([PY, str(scripts / "report_corpus_coverage.py")])
    run([PY, str(scripts / "reddit_roberta_apd.py"), "--min-bin", "20"])
    run([PY, str(scripts / "reddit_tfidf_apd.py")])
    run([PY, str(scripts / "reddit_fasttext_apd.py")])
    run([PY, str(scripts / "reddit_drift_diagnostic.py")])
    run([PY, str(scripts / "reddit_umap_all.py")])
    run([PY, str(scripts / "reddit_results_figures.py")])
    print("Done. Outputs in results/ and results/figures/.")


if __name__ == "__main__":
    main()

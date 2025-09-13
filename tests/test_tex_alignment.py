"""
Checks that: if the paper mentions CFR, at least one CFR implementation file exists.
This is a smoke alignment test; see tools/tex_alignment_checker.py for the full report.
"""
import io
import os
import re
import glob

ROOT = os.path.dirname(os.path.dirname(__file__))
PAPER = os.path.join(ROOT, "paper.tex")


def _paper_mentions(term: str) -> bool:
    if not os.path.exists(PAPER):
        # allow pass if paper not present in checkout
        return False
    with io.open(PAPER, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().lower()
    return term.lower() in txt


def test_cfr_exists_if_paper_mentions_it():
    if not _paper_mentions("cfr"):
        return
    # search broadly across the repo
    paths = glob.glob(os.path.join(ROOT, "**", "*cfr*.py"), recursive=True)
    assert len(paths) >= 1, "Paper mentions CFR but no *cfr*.py found"


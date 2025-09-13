#!/usr/bin/env python3
"""
Paper-to-code alignment checker.
Parses `paper.tex` for algorithm keywords and scans the repo for likely implementations.
Writes `paper_alignment_report.json` at the repo root.
"""
import io
import json
import os
import re
import glob
from typing import Dict, List

KEYWORDS = {
    "CFR": [r"\\bCFR\\b", r"\\bCFR\\+\\b", r"\\bDeep\\s+CFR\\b"],
    "CFR+": [r"\\bCFR\\+\\b"],
    "Deep CFR": [r"\\bDeep\\s+CFR\\b"],
    "Transformer": [r"\\bTransformer\\b", r"\\battention\\b"],
}

FILE_PATTERNS = {
    "CFR": ["**/*cfr*.py", "cfr_algorithm/**/*.py"],
    "CFR+": ["**/*cfr*plus*.py", "**/*cfr_plus*.py"],
    "Deep CFR": ["**/*deep*cfr*.py"],
    "Transformer": ["**/*transformer*.py", "ai_models/**/*.py", "models/**/*.py"],
}


def _read_text(path: str) -> str:
    try:
        with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _paper_keywords(paper_path: str) -> Dict[str, bool]:
    txt = _read_text(paper_path).lower()
    res = {}
    for k, pats in KEYWORDS.items():
        res[k] = any(re.search(pat, txt, flags=re.IGNORECASE) for pat in pats)
    return res


def _find_impls(root: str, patterns: List[str]) -> List[str]:
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(root, pat), recursive=True))
    # deduplicate and keep relative paths
    root = os.path.abspath(root)
    uniq = sorted(set([os.path.relpath(os.path.abspath(p), root) for p in out if os.path.isfile(p)]))
    return uniq


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paper = os.path.join(root, "paper.tex")
    report = {
        "paper_present": os.path.exists(paper),
        "keywords_in_paper": {},
        "implementations_found": {},
    }
    if os.path.exists(paper):
        report["keywords_in_paper"] = _paper_keywords(paper)
    for k, pats in FILE_PATTERNS.items():
        report["implementations_found"][k] = _find_impls(root, pats)

    out_path = os.path.join(root, "paper_alignment_report.json")
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Wrote {out_path}")
    # Basic console hints
    for k in KEYWORDS.keys():
        in_paper = report["keywords_in_paper"].get(k, False)
        found = len(report["implementations_found"].get(k, [])) > 0
        if in_paper and not found:
            print(f"[WARN] Paper mentions '{k}' but no matching implementation files were found.")

if __name__ == "__main__":
    main()


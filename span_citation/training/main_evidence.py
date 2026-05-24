"""Compatibility wrapper for the standalone evaluation script.

This keeps older commands using `main_evidence.py` working while the
actual implementation lives in `llama_evidence.py`.
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("span_citation.training.llama_evidence", run_name="__main__")

#!/usr/bin/env python3
"""
CRAFT Pipeline for OTT-QA dataset.

Usage:
    python run_ottqa.py --stage [1|2|3|all]
    python run_ottqa.py --stage all --api-key $GEMINI_API_KEY --use-gemini
    python run_ottqa.py --stage 2 --stage1-path results/stage1/ottqa_stage1_splade_corpus.jsonl
    python run_ottqa.py --evaluate results/stage3/ottqa_stage3_results.pkl

Notebook alternatives (for interactive development):
    Stage 1: scripts/stage1_splade_retrieval.ipynb
    Stage 2: scripts/stage2_dense_reranking.ipynb  (set DATASET="ottqa")
    Stage 3: scripts/stage3_neural_reranking.ipynb
"""

import sys
from pathlib import Path

# Ensure scripts/ is importable without package installation
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from craft_pipeline import main

if __name__ == "__main__":
    # Inject the dataset default so the shared main() targets OTT-QA
    if "--dataset" not in sys.argv:
        sys.argv.insert(1, "ottqa")
        sys.argv.insert(1, "--dataset")
    # Default to Gemini for OTT-QA (Stage 3)
    if "--use-gemini" not in sys.argv:
        sys.argv.append("--use-gemini")
    main()
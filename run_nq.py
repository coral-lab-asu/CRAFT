#!/usr/bin/env python3
"""
CRAFT Pipeline for Natural Questions (NQ-Tables) dataset.

Usage:
    python run_nq.py --stage [1|2|3|all]
    python run_nq.py --stage all --api-key $OPENAI_API_KEY
    python run_nq.py --stage 2 --stage1-path results/stage1/nq_stage1_splade_corpus.jsonl
    python run_nq.py --evaluate results/stage3/nq_stage3_results.pkl

Notebook alternatives (for interactive development):
    Stage 1: scripts/stage1_splade_retrieval.ipynb
    Stage 2: scripts/stage2_dense_reranking.ipynb  (set DATASET="nq")
    Stage 3: scripts/stage3_neural_reranking.ipynb
"""

import sys
from pathlib import Path

# Ensure scripts/ is importable without package installation
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from craft_pipeline import main

if __name__ == "__main__":
    # Inject the dataset default so the shared main() targets NQ
    import sys
    if "--dataset" not in sys.argv:
        sys.argv.insert(1, "nq")
        sys.argv.insert(1, "--dataset")
    main()
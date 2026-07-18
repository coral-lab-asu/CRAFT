"""Build and cache the SPLADE inverted index (Stage 1's input).

Run once per corpus. If the index file already exists it is loaded from cache,
so re-running preprocessing is cheap.
"""

import gc
import time
from pathlib import Path
from typing import List

import torch

from craft_tabqa.core.files import load_pickle, save_pickle
from craft_tabqa.core.sparse import build_inverted_index, load_splade_model


def build_splade_index(
    corpus_texts: List[str],
    output_path: str,
    model_id: str = "naver/splade_v2_distil",
    batch_size: int = 128,
    sparse_top_k: int = 5_000,
    logger=None,
) -> dict:
    """Encode ``corpus_texts`` with SPLADE and save the inverted index.

    Returns the index dict. The SPLADE model is freed from GPU memory before
    returning so Stage 2's model can load without contention.
    """
    log = logger.info if logger else print
    output_path = Path(output_path)

    if output_path.exists():
        log(f"[splade] Cached index found at {output_path}")
        return load_pickle(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[splade] Loading model {model_id} on {device}")
    tokenizer, model, device = load_splade_model(model_id, str(device))

    log(f"[splade] Indexing {len(corpus_texts):,} tables ...")
    start = time.time()
    index = build_inverted_index(
        corpus_texts=corpus_texts,
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        batch_size=batch_size,
        top_k=sparse_top_k,
        device=device,
        logger=logger,
    )
    log(f"[splade] Indexed in {time.time() - start:.1f}s ({len(index):,} terms)")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    save_pickle(output_path, index)
    log(f"[splade] Index saved to {output_path}")
    return index

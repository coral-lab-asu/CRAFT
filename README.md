# CRAFT: Training-Free Cascaded Retrieval for Tabular QA

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-red)](https://arxiv.org/abs/2505.14984)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)

CRAFT is a **training-free**, three-stage cascaded retriever for open-domain table
question answering. It reaches state-of-the-art retrieval on NQ-Tables and strong
zero-shot generalisation on OTT-QA with **no dataset-specific fine-tuning**.

| Stage | Model | Input → Output |
|-------|-------|----------------|
| **Stage 1** | SPLADE (sparse) | full corpus → top 5,000 |
| **Stage 2** | Sentence Transformer (dense, per-row) | 5,000 → top 100 |
| **Stage 3** | OpenAI / Gemini embeddings | 100 → top 50 *(optional)* |

An optional **LLM enrichment** step can generate table titles/descriptions (and
expand queries) with a small open-source model before indexing.

---

## Install

```bash
pip install craft-tabqa                 # core (Stages 1 & 2)
pip install "craft-tabqa[vllm]"         # + LLM enrichment via vLLM
pip install "craft-tabqa[openai]"       # + Stage 3 with OpenAI embeddings
pip install "craft-tabqa[serve]"        # + web UI / retrieval API
pip install "craft-tabqa[tui]"          # + interactive terminal app (craft tui)
pip install "craft-tabqa[all]"          # everything except vllm
```

From source:

```bash
git clone https://github.com/corallab-asu/CRAFT.git
cd CRAFT
pip install -e ".[dev]"
```

Secrets and hardware settings come from a `.env` file (see `.env.example`):

```ini
CUDA_VISIBLE_DEVICES=0,1
HF_HOME=/path/to/hf_cache
HF_TOKEN=hf_...          # gated models (e.g. JINA v3)
OPENAI_API_KEY=sk-...    # enables Stage 3
GEMINI_API_KEY=...       # enables Stage 3 (Gemini)
```

---

## Quick start

Every command reads one YAML config and shares a single cache directory
(`<output_dir>/cache`), so preprocessing runs once and retrieval reuses it.

```bash
# 1. Build the SPLADE index and row embeddings (once per corpus)
craft preprocess --config configs/nq_tables.yaml

# 2. Retrieve for the query set (Stage 1 → 2 → optional 3)
craft retrieve   --config configs/nq_tables.yaml

# 3. (optional) Serve an interactive retrieval UI over the preprocessed corpus
craft serve      --config configs/nq_tables.yaml --port 8000

# 3. (alternative) Explore results in an interactive terminal app
craft tui        # menu-driven; pick a config, a question, and a pipeline scope
```

Retrieval writes `stage{1,2,3}_results.jsonl` and a `recall_summary.csv` into a
fresh timestamped run directory.

Useful flags:

```bash
craft preprocess --config <cfg> --rebuild-index   # rebuild just the SPLADE index
craft retrieve   --config <cfg> --skip-stage1     # reuse saved Stage 1 results
craft retrieve   --config <cfg> --no-stage3       # skip Stage 3 even if a key is set
```

---

## LLM enrichment (optional)

Set `generation.enabled: true` in your config to generate a title and
description for every table (and, with `expand_queries: true`, a sub-question
and description for every query) before indexing. The prompts live in
[`src/craft_tabqa/prompts/`](src/craft_tabqa/prompts) and are easy to edit.

```bash
craft generate --config configs/nq_tables.yaml   # enrichment only
craft preprocess --config configs/nq_tables.yaml # enrichment runs automatically if enabled
```

Backends: `vllm` (default, batched local generation — recommended at corpus
scale), `transformers` (dependency-light), or `openai`. Generated fields are
cached to `table_enrichment.jsonl` / `query_expansion.jsonl`, so the step is
resumable.

---

## Stage 2 modes

Set `stage2.mode` in your config:

| Mode | How it works |
|------|--------------|
| `representative_row` *(default)* | A table's score is its best-matching row's cosine similarity. Rows are pre-encoded during preprocessing, so nothing is re-encoded at query time. |
| `mini_table` | For each candidate, join its top rows into a mini-table and re-encode it at query time (the original paper method). |

---

## Custom datasets

Point a config at your own JSONL files (see
[`configs/custom_template.yaml`](configs/custom_template.yaml)).

**corpus.jsonl** — one table per line:

```json
{"table_id": "solar_0", "title": "Planets", "headers": ["Planet", "Moons"],
 "rows": [["Mercury", "0"], ["Saturn", "146"]], "description": "optional"}
```

**queries.jsonl** — one question per line:

```json
{"qid": "q1", "question": "Which planet has the most moons?",
 "gold_table_ids": ["solar_0"], "subquestion": "optional", "answer": "optional"}
```

`gold_table_ids` is only used to compute recall — omit it if you have no ground truth.

---

## Data setup (NQ-Tables and OTT-QA)

- **NQ-Tables** — download `tables.jsonl` and `combined.jsonl` from the
  [TAPAS NQ-Tables release](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md)
  into `datasets/NQ_Tables/`.
- **OTT-QA** — download `all_plain_tables.json` and `released_data/dev.json`
  following the [OTT-QA repo](https://github.com/wenhuchen/OTT-QA#step1-1-download-the-necessary-files)
  into `datasets/OTT-QA/`.

See the config files for the exact paths CRAFT expects.

---

## Package layout

```
src/craft_tabqa/
├── config.py            typed YAML config
├── cli.py               the `craft` command
├── core/                text builders, SPLADE, dense encoding, metrics, file I/O
├── loaders/             NQ-Tables / OTT-QA / generic JSONL loaders
├── preprocessing/       LLM enrichment, SPLADE index, row encoder, orchestrator
├── retrieval/           stage 1/2/3 + orchestrator + recall reporting
├── serve/               FastAPI app, resident engine, demo UI
├── tui/                 interactive menu-driven terminal app
└── prompts/             editable enrichment prompt templates
```

---

## Citation

```bibtex
@misc{singh2025crafttrainingfreecascadedretrieval,
      title={CRAFT: Training-Free Cascaded Retrieval for Tabular QA},
      author={Adarsh Singh and Kushal Raj Bhandari and Jianxi Gao and Soham Dan and Vivek Gupta},
      year={2025},
      eprint={2505.14984},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14984},
}
```

## License

MIT — see [LICENSE](LICENSE).

# CRAFT QA Evaluation Scripts

This directory contains scripts for end-to-end question answering evaluation on CRAFT retrieval results.

## Files

- **`qa_evaluation.py`**: Main QA evaluation script with command-line interface
- **`qa_evaluation_notebook.ipynb`**: Interactive Jupyter notebook for QA analysis  
- **`run_qa_example.py`**: Simple example script showing basic usage

## Quick Start

### 1. Basic Example
```bash
# Run a quick evaluation example (requires API keys)
python run_qa_example.py
```

### 2. Command Line Evaluation
```bash
# Full evaluation with custom parameters
python qa_evaluation.py \
    --metadata ../datasets/nq_tables_metadata_updated.csv \
    --questions ../datasets/combined.jsonl \
    --corpus ../results/stage3/CRAFT_NQ_Final_Splade_ST_OpenAI_100_ranks_966_q.pkl \
    --top-rows ../results/stage2/corpus_2nd_stage_row_rerank_with_ST_original_q.pkl \
    --row-data ../datasets/nq_row_tables.json \
    --model gpt-4o \
    --tables 1 5 \
    --mini-table \
    --max-queries 100 \
    --output qa_results.jsonl
```

### 3. Jupyter Notebook
```bash
# Launch interactive notebook
jupyter notebook qa_evaluation_notebook.ipynb
```

## Configuration

### API Keys
Set environment variables for the models you want to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

### Supported Models
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`  
- **Gemini**: `gemini-2.0-flash-exp`, `gemini-1.5-pro`

## Key Features

### Token Efficiency Analysis
Compare mini-table vs full-table token usage:
- Mini-tables use only top 5 rows per table
- Full-tables include complete table content
- Automatic token counting and efficiency reporting

### Multi-Configuration Testing
Test different combinations of:
- Number of tables (1, 3, 5, 10)
- Table types (mini vs full)
- Different models (GPT-4o vs Gemini)

### Evaluation Metrics
- **F1 Score**: Token-level overlap between generated and ground truth answers
- **Token Count**: Input prompt length for cost estimation
- **Processing Time**: Evaluation speed benchmarking

### Batch Processing
- Process large query sets efficiently
- Result persistence in JSONL format
- Resume interrupted evaluations
- Progress tracking with tqdm

## Example Outputs

### Token Efficiency Results
```
📈 Token Efficiency Results:
   Mini-table average tokens: 342.1
   Full-table average tokens: 1,248.7
   Token reduction: 72.6%
   Cost savings: ~72.6% on LLM API costs
```

### QA Evaluation Summary
```
📊 QA Evaluation Summary:
Configuration            F1 Score   Avg Tokens
GPT-4o + 1 mini-table   0.847      342        
GPT-4o + 5 mini-tables  0.892      1,458      
Gemini + 1 mini-table   0.831      338        
Gemini + 5 mini-tables  0.878      1,442      
```

## Usage Tips

1. **Start Small**: Use `--max-queries 10` for initial testing
2. **Monitor Costs**: Track token usage to estimate API costs
3. **Rate Limiting**: Built-in delays for Gemini API compliance
4. **Error Handling**: Robust error handling with detailed logging
5. **Reproducibility**: Set `temperature=0.0` for consistent results

## File Dependencies

The QA evaluation requires these files from running the CRAFT pipeline:

### Required Files
- `nq_tables_metadata_updated.csv`: Table metadata
- `combined.jsonl`: Questions with answers  
- `CRAFT_*_Final_*.pkl`: Stage 3 retrieval results
- `corpus_2nd_stage_*.pkl`: Stage 2 row rankings
- `nq_row_tables.json`: Row-level table data

### File Locations
```
CRAFT/
├── datasets/           # Questions and metadata
├── results/
│   ├── stage2/        # Dense reranking results
│   └── stage3/        # Final neural reranking results
└── scripts/           # This directory
```

## Troubleshooting

### Common Issues

**Missing API Keys**:
```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

**File Not Found**:
Check that CRAFT pipeline has been run and all result files exist in the expected locations.

**Memory Issues**:
Use `--max-queries` to limit evaluation size, or process in smaller batches.

**Rate Limiting**:
Built-in delays are included, but you may need to adjust for your API tier.

### Getting Help
```bash
python qa_evaluation.py --help
```

## Integration with CRAFT Pipeline

The QA evaluation integrates seamlessly with CRAFT results:

1. **Run CRAFT Pipeline**: Execute `run_nq.py` or `run_ottqa.py`
2. **Generate Results**: Pipeline creates all required `.pkl` and `.jsonl` files  
3. **Run QA Evaluation**: Use scripts in this directory to test QA performance
4. **Analyze Results**: Compare different configurations and models

This provides end-to-end evaluation from table retrieval through question answering.
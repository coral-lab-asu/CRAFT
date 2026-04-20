#!/usr/bin/env python3
"""
CRAFT QA Evaluation Example

Simple example script showing how to run QA evaluation on CRAFT results.
This script assumes the standard CRAFT directory structure.
"""

import os
import sys
from pathlib import Path

# Add the CRAFT scripts directory to Python path
craft_dir = Path(__file__).parent.parent
sys.path.append(str(craft_dir / "scripts"))

from qa_evaluation import QAEvaluator

def main():
    """Run example QA evaluation."""
    
    # Standard CRAFT paths (adjust if needed)
    craft_root = Path(__file__).parent.parent
    
    # Data paths
    metadata_path = craft_root / "datasets" / "nq_tables_metadata_updated.csv"
    questions_path = craft_root / "datasets" / "combined.jsonl"  # Questions file
    
    # Results paths (these should exist after running CRAFT pipeline)
    corpus_path = craft_root / "results" / "stage3" / "nq_stage3_results.pkl"
    top_rows_path = craft_root / "results" / "stage2" / "nq_stage2_results.pkl"
    row_data_path = craft_root / "datasets" / "nq_row_tables.json"
    
    # Check if files exist
    required_files = [metadata_path, questions_path, corpus_path, top_rows_path, row_data_path]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   {f}")
        print("\nPlease run the CRAFT pipeline first to generate all required files.")
        return
    
    print("✅ All required files found")
    
    # Initialize evaluator (API keys are read from OPENAI_API_KEY / GEMINI_API_KEY env vars)
    print("🔧 Initializing QA evaluator...")
    
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("⚠️  No API keys found. Set OPENAI_API_KEY or GEMINI_API_KEY environment variables.")
        print("   Example: export OPENAI_API_KEY='your-key-here'")
        print("   For now, running in token analysis mode only...")
    
    evaluator = QAEvaluator(
        metadata_path=str(metadata_path),
        questions_path=str(questions_path),
        corpus_path=str(corpus_path),
        top_rows_path=str(top_rows_path),
        row_data_path=str(row_data_path),
    )
    
    # Run token efficiency analysis
    print("\n📊 Running token efficiency analysis...")
    
    # Compare mini-table vs full-table token usage
    sample_questions = evaluator.questions[:20]  # Sample first 20 questions
    mini_tokens = []
    full_tokens = []
    
    for i, q_data in enumerate(sample_questions):
        if str(i) not in evaluator.corpus:
            continue
            
        qid = q_data['qid']
        if qid not in evaluator.top_rows_per_table:
            qid = qid + "_0"
            
        question = q_data['OriginalQuestion']
        table_ids = [tid for (tid, _) in evaluator.corpus[str(i)][:5]]  # Top 5 tables
        
        # Mini-table version
        mini_tables, valid_ids = evaluator.get_table_text(qid, table_ids, use_mini_table=True)
        if mini_tables:
            mini_prompt = evaluator.create_prompt(question, mini_tables, valid_ids)
            mini_tokens.append(evaluator.count_tokens(mini_prompt))
        
        # Full-table version  
        full_tables, valid_ids = evaluator.get_table_text(qid, table_ids, use_mini_table=False)
        if full_tables:
            full_prompt = evaluator.create_prompt(question, full_tables, valid_ids)
            full_tokens.append(evaluator.count_tokens(full_prompt))
    
    if mini_tokens and full_tokens:
        avg_mini = sum(mini_tokens) / len(mini_tokens)
        avg_full = sum(full_tokens) / len(full_tokens)
        reduction = (avg_full - avg_mini) / avg_full * 100
        
        print(f"📈 Token Efficiency Results:")
        print(f"   Mini-table average tokens: {avg_mini:.1f}")
        print(f"   Full-table average tokens: {avg_full:.1f}")
        print(f"   Token reduction: {reduction:.1f}%")
        print(f"   Cost savings: ~{reduction:.1f}% on LLM API costs")
    
    # Run QA evaluation if API keys are available
    if openai_key or gemini_keys:
        print("\n🤖 Running QA evaluation...")
        
        # Test different configurations
        configs = [
            {"model": "gpt-4o", "tables": 1, "mini": True, "name": "GPT-4o + 1 mini-table"},
            {"model": "gpt-4o", "tables": 5, "mini": True, "name": "GPT-4o + 5 mini-tables"},
        ]
        
        if gemini_keys and gemini_keys[0]:
            configs.extend([
                {"model": "gemini-2.0-flash-exp", "tables": 1, "mini": True, "name": "Gemini + 1 mini-table"},
                {"model": "gemini-2.0-flash-exp", "tables": 5, "mini": True, "name": "Gemini + 5 mini-tables"},
            ])
        
        results = []
        for config in configs:
            print(f"\n⚙️  Testing: {config['name']}")
            
            try:
                result = evaluator.evaluate_qa(
                    model_name=config["model"],
                    num_tables=[config["tables"]],
                    use_mini_table=config["mini"],
                    max_queries=10,  # Small sample for quick test
                    output_file=f"qa_results_{config['model'].replace('-', '_')}_{config['tables']}tables.jsonl"
                )
                
                if result and config["tables"] in result['results_by_n_tables']:
                    stats = result['results_by_n_tables'][config["tables"]]
                    results.append({
                        'config': config['name'],
                        'f1': stats['avg_f1'],
                        'tokens': stats['avg_tokens'],
                        'queries': stats['queries_processed']
                    })
            except Exception as e:
                print(f"   Error: {e}")
        
        # Display comparison
        if results:
            print(f"\n📊 QA Evaluation Summary:")
            print("-" * 60)
            print(f"{'Configuration':<25} {'F1 Score':<10} {'Avg Tokens':<12}")
            print("-" * 60)
            for r in results:
                print(f"{r['config']:<25} {r['f1']:<10.3f} {r['tokens']:<12.0f}")
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Detailed results saved to qa_results_*.jsonl files")
    print(f"\nTo run full evaluation:")
    print(f"   python qa_evaluation.py --help")

if __name__ == "__main__":
    main()
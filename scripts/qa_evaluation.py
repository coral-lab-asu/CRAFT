#!/usr/bin/env python3
"""
CRAFT End-to-End QA Evaluation Script

This script performs question answering on retrieved tables using various LLMs
and computes evaluation metrics like F1 score and token efficiency.
"""

import re
import json
import os
import time
import argparse
import pickle
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm


def _api_call_with_backoff(fn, max_retries: int = 5, base_delay: float = 1.0):
    """Execute fn() with exponential backoff on rate-limit / transient errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"API call failed ({exc}). Retrying in {delay:.1f}s...")
            time.sleep(delay)

# Optional imports - install as needed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini not available. Install with: pip install google-generativeai")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")


class QAEvaluator:
    """End-to-end QA evaluator for CRAFT pipeline results."""
    
    def __init__(self, 
                 metadata_path: str,
                 questions_path: str, 
                 corpus_path: str,
                 top_rows_path: str,
                 row_data_path: str):
        """
        Initialize QA evaluator with data paths.
        API keys are read from environment variables:
          OPENAI_API_KEY  — for GPT models
          GEMINI_API_KEY  — for Gemini models

        Args:
            metadata_path: Path to table metadata CSV
            questions_path: Path to questions JSONL file  
            corpus_path: Path to CRAFT retrieval results PKL
            top_rows_path: Path to top rows per table PKL
            row_data_path: Path to row data JSON
        """
        self.metadata_path = metadata_path
        self.questions_path = questions_path
        self.corpus_path = corpus_path
        self.top_rows_path = top_rows_path
        self.row_data_path = row_data_path
        
        # Initialize APIs from environment variables only
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if openai_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=openai_key)
        else:
            self.openai_client = None
            
        if gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=gemini_key)
            self.gemini_available = True
        else:
            self.gemini_available = False
        
        # Load data
        self._load_data()
        
        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        else:
            self.encoding = None
    
    def _load_data(self):
        """Load all required data files."""
        print("Loading data files...")
        
        # Load metadata
        metadata_columns = ["index", "TableID", "Table_Title", "Table_Headers", "Table_CellValues"]
        dtypes = {
            "index": int,
            "TableID": str, 
            "Table_Title": str,
            "Table_Headers": str,
            "Table_CellValues": str
        }
        self.metadata = pd.read_csv(
            self.metadata_path,
            names=metadata_columns,
            dtype=dtypes,
            skiprows=1
        )
        
        # Load questions
        with open(self.questions_path, 'r') as f:
            self.questions = [json.loads(line) for line in f]
        
        # Load corpus results
        with open(self.corpus_path, 'rb') as f:
            self.corpus = pickle.load(f)
            
        # Load top rows per table
        with open(self.top_rows_path, 'rb') as f:
            self.top_rows_per_table = pickle.load(f)
        
        # Normalise QIDs in top_rows_per_table once here so lookup is O(1)
        # Some entries have a "_0" suffix appended; strip it for consistency.
        self.top_rows_per_table = {
            k.rstrip("_0") if k.endswith("_0") else k: v
            for k, v in self.top_rows_per_table.items()
        }
            
        # Load row data
        with open(self.row_data_path, 'r') as f:
            row_table_data = json.load(f)
        self.rowid_to_data = {entry['Row Number']: entry for entry in row_table_data}
        
        print(f"Loaded {len(self.questions)} questions, {len(self.metadata)} tables")
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer by removing punctuation and extra spaces."""
        s = re.sub(r'[^\w\s]', '', s.lower())
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    @staticmethod
    def tokenize_answer(s: str) -> List[str]:
        """Tokenize normalized answer."""
        return QAEvaluator.normalize_answer(s).split()
    
    @staticmethod
    def compute_f1_score(generated_answer: str, ground_truth_answers: List[str]) -> float:
        """
        Compute maximum F1 score between generated answer and any ground truth.
        
        Args:
            generated_answer: Model-generated answer
            ground_truth_answers: List of acceptable ground truth answers
            
        Returns:
            Maximum F1 score
        """
        generated_tokens = QAEvaluator.tokenize_answer(generated_answer)
        if not generated_tokens:
            return 0.0
        
        max_f1 = 0.0
        for gt_answer in ground_truth_answers:
            gt_tokens = QAEvaluator.tokenize_answer(gt_answer)
            if not gt_tokens:
                continue
            
            generated_counter = Counter(generated_tokens)
            gt_counter = Counter(gt_tokens)
            common = sum((generated_counter & gt_counter).values())
            
            precision = common / sum(generated_counter.values()) if generated_counter else 0.0
            recall = common / sum(gt_counter.values()) if gt_counter else 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    def get_table_text(self, qid: str, table_ids: List[str], use_mini_table: bool = True) -> Tuple[List[str], List[str]]:
        """
        Retrieve table text for given table IDs.
        
        Args:
            qid: Query ID
            table_ids: List of table IDs to retrieve
            use_mini_table: If True, use top 5 rows only; if False, use all rows
            
        Returns:
            Tuple of (table_texts, valid_table_ids)
        """
        table_texts = []
        valid_table_ids = []
        
        for table_id in table_ids:
            rows = self.top_rows_per_table.get(str(qid), {}).get(table_id, [])
            if not rows:
                continue
            
            row_texts = []
            for rid in rows:
                if rid in self.rowid_to_data:
                    row_texts.append(self.rowid_to_data[rid]["Row Data"])
            
            if row_texts:
                if use_mini_table:
                    table_text = " ".join(row_texts[:5])  # Mini-table: top 5 rows
                else:
                    table_text = " ".join(row_texts)  # Full table: all rows
                table_texts.append(table_text)
                valid_table_ids.append(table_id)
        
        return table_texts, valid_table_ids
    
    def create_prompt(self, question: str, table_texts: List[str], valid_table_ids: List[str]) -> str:
        """
        Create QA prompt for language model.
        
        Args:
            question: Input question
            table_texts: List of table content strings
            valid_table_ids: Corresponding table IDs
            
        Returns:
            Formatted prompt string
        """
        tables_str = "\n".join([
            f"Table Title: {self.metadata.loc[self.metadata['TableID'] == tid, 'Table_Title'].iloc[0]}, "
            f"Table Content: {content}"
            for tid, content in zip(valid_table_ids, table_texts)
        ])
        
        prompt = f"""You are given a set of tables along with their titles. Based on the information provided in the most relevant table(s) answer the following question. Do **not** include any explanations or extra text.
Ensure the final answer format is **only**: 
'FINAL ANSWER: Answer Value'

[TABLES]
{tables_str}

QUESTION: {question}
FINAL ANSWER:"""
        
        return prompt
    
    def query_openai(self, prompt: str, model: str = "gpt-4o", max_answer_tokens: int = 200) -> str:
        """Query OpenAI model."""
        if not self.openai_client:
            return "Error: OpenAI client not initialized"
            
        def _call():
            return self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_answer_tokens,
                temperature=0.0
            )
        
        try:
            response = _api_call_with_backoff(_call)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"
    
    def query_gemini(self, prompt: str, model: str = "gemini-2.0-flash-exp", max_answer_tokens: int = 200) -> str:
        """Query Gemini model."""
        if not self.gemini_available:
            return "Error: Gemini not initialized"
            
        def _call():
            model_obj = genai.GenerativeModel(model)
            return model_obj.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_answer_tokens,
                    "temperature": 0.0
                }
            )
        
        try:
            response = _api_call_with_backoff(_call)
            return response.text.strip()
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in text."""
        if model.startswith("gpt") and self.encoding:
            return len(self.encoding.encode(text))
        elif model.startswith("gemini") and GEMINI_AVAILABLE:
            try:
                model_obj = genai.GenerativeModel('gemini-2.0-flash-exp')
                return model_obj.count_tokens(text).total_tokens
            except:
                return len(text.split())  # Fallback to word count
        else:
            return len(text.split())  # Fallback to word count
    
    def extract_answer(self, response: str) -> str:
        """Extract answer from model response."""
        answer_match = re.search(r'FINAL ANSWER: (.*)', response)
        return answer_match.group(1).strip() if answer_match else response
    
    def evaluate_qa(self, 
                   model_name: str = "gpt-4o",
                   num_tables: List[int] = [1, 5],
                   use_mini_table: bool = True,
                   max_queries: int = None,
                   output_file: str = "qa_results.jsonl") -> Dict:
        """
        Run end-to-end QA evaluation.
        
        Args:
            model_name: Model to use ("gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp")
            num_tables: List of table counts to test (e.g., [1, 3, 5])
            use_mini_table: Whether to use mini-tables or full tables
            max_queries: Maximum number of queries to process (None for all)
            output_file: Output file for detailed results
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Running QA evaluation with {model_name}")
        print(f"Table configurations: {num_tables} tables")
        print(f"Using {'mini-tables' if use_mini_table else 'full tables'}")
        
        results = []
        token_stats = {n: [] for n in num_tables}
        f1_stats = {n: [] for n in num_tables}
        
        query_count = 0
        for q_data in tqdm(self.questions, desc=f"Processing queries with {model_name}"):
            if max_queries and query_count >= max_queries:
                break
                
            qid = q_data['qid']
            # QID is already normalised during _load_data; look it up directly.
            
            if str(query_count) not in self.corpus:
                query_count += 1
                continue
                
            question = q_data['OriginalQuestion']
            gold_table_id = q_data['gold_table_id']
            ground_truth = q_data['AnswerTexts']
            
            # Get ranked table IDs from corpus
            table_ids = [tid for (tid, _) in self.corpus[str(query_count)]]
            gold_rank = next((i+1 for i, (tid, _) in enumerate(self.corpus[str(query_count)]) 
                            if tid == gold_table_id), len(table_ids) + 1)
            
            query_result = {
                'query_num': query_count,
                'qid': qid,
                'question': question,
                'gold_table_id': gold_table_id,
                'gold_rank': gold_rank,
                'ground_truth': ground_truth,
                'model': model_name,
                'use_mini_table': use_mini_table
            }
            
            # Test different numbers of tables
            for n in num_tables:
                top_n_tables = table_ids[:n]
                table_texts, valid_table_ids = self.get_table_text(qid, top_n_tables, use_mini_table)
                
                if not table_texts:
                    continue
                
                # Create prompt and count tokens
                prompt = self.create_prompt(question, table_texts, valid_table_ids)
                token_count = self.count_tokens(prompt, model_name)
                token_stats[n].append(token_count)
                
                # Query model
                if model_name.startswith("gpt"):
                    response = self.query_openai(prompt, model_name)
                elif model_name.startswith("gemini"):
                    response = _api_call_with_backoff(
                        lambda p=prompt, m=model_name: self.query_gemini(p, m)
                    )
                else:
                    response = f"Error: Unknown model {model_name}"
                
                # Extract and evaluate answer
                generated_answer = self.extract_answer(response)
                f1_score = self.compute_f1_score(generated_answer, ground_truth)
                f1_stats[n].append(f1_score)
                
                # Store result
                query_result[f'n{n}_response'] = response
                query_result[f'n{n}_answer'] = generated_answer
                query_result[f'n{n}_f1'] = f1_score
                query_result[f'n{n}_tokens'] = token_count
            
            results.append(query_result)
            query_count += 1
        
        # Compute summary statistics
        summary = {
            'model': model_name,
            'use_mini_table': use_mini_table,
            'total_queries': len(results),
            'results_by_n_tables': {}
        }
        
        for n in num_tables:
            if f1_stats[n]:
                summary['results_by_n_tables'][n] = {
                    'avg_f1': sum(f1_stats[n]) / len(f1_stats[n]),
                    'avg_tokens': sum(token_stats[n]) / len(token_stats[n]),
                    'queries_processed': len(f1_stats[n])
                }
        
        # Save detailed results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\nResults saved to {output_file}")
        print("Summary:")
        for n in num_tables:
            if n in summary['results_by_n_tables']:
                stats = summary['results_by_n_tables'][n]
                print(f"  n={n} tables: F1={stats['avg_f1']:.3f}, Avg Tokens={stats['avg_tokens']:.1f}")
        
        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="CRAFT QA Evaluation")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--questions", required=True, help="Path to questions JSONL")
    parser.add_argument("--corpus", required=True, help="Path to corpus PKL")
    parser.add_argument("--top-rows", required=True, help="Path to top rows PKL")
    parser.add_argument("--row-data", required=True, help="Path to row data JSON")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--tables", nargs="+", type=int, default=[1, 5], help="Numbers of tables to test")
    parser.add_argument("--mini-table", action="store_true", help="Use mini-tables")
    parser.add_argument("--max-queries", type=int, help="Maximum queries to process")
    parser.add_argument("--output", default="qa_results.jsonl", help="Output file")
    parser.add_argument("--openai-key", help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    parser.add_argument("--gemini-keys", nargs="+", help="Gemini API keys (overrides GEMINI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Allow CLI key overrides to propagate via environment so the class
    # picks them up from os.environ (avoids passing secrets through objects).
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.gemini_keys:
        os.environ["GEMINI_API_KEY"] = args.gemini_keys[0]
    
    # Initialize evaluator
    evaluator = QAEvaluator(
        metadata_path=args.metadata,
        questions_path=args.questions,
        corpus_path=args.corpus,
        top_rows_path=args.top_rows,
        row_data_path=args.row_data
    )
    
    # Run evaluation
    results = evaluator.evaluate_qa(
        model_name=args.model,
        num_tables=args.tables,
        use_mini_table=args.mini_table,
        max_queries=args.max_queries,
        output_file=args.output
    )
    
    print("\nEvaluation complete!")
    return results


if __name__ == "__main__":
    main()
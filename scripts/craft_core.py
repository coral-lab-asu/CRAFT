#!/usr/bin/env python3
"""
CRAFT Core Utilities

Shared utility functions and classes used across all CRAFT pipeline stages.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CRAFTConfig:
    """Configuration class for CRAFT pipeline."""
    
    def __init__(self, dataset: str = "nq"):
        self.dataset = dataset.lower()
        self.base_dir = Path(__file__).parent.parent
        
        # Stage-specific configurations
        self.stage1_candidates = 5000
        self.stage2_candidates = 100
        self.mini_table_rows = 5
        
        # Model configurations
        self.models = {
            "nq": {
                "stage1": "naver-splade/splade-v2-distil",
                "stage2": "sentence-transformers/all-mpnet-base-v2", 
                "stage3": "text-embedding-3-large"
            },
            "ottqa": {
                "stage1": "naver-splade/splade-v2-distil",
                "stage2": "jinaai/jina-embeddings-v3",
                "stage3": "models/gemini-embedding-001"  # Gemini
            }
        }
        
        # File paths
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup default file paths based on dataset."""
        datasets_dir = self.base_dir / "datasets"
        results_dir = self.base_dir / "results"
        
        self.paths = {
            "metadata": datasets_dir / f"{self.dataset}_tables_metadata_updated.csv",
            "questions": datasets_dir / f"{self.dataset}_queries_test_metadata.tsv",
            "stage1_output": results_dir / "stage1" / f"{self.dataset}_stage1_splade_corpus.jsonl",
            "stage2_output": results_dir / "stage2" / f"{self.dataset}_stage2_results.pkl",
            "stage3_output": results_dir / "stage3" / f"{self.dataset}_stage3_results.pkl"
        }
        
        # Create directories if they don't exist
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_model(self, stage: int) -> str:
        """Get model name for specific stage."""
        stage_key = f"stage{stage}"
        return self.models[self.dataset][stage_key]
    
    def get_path(self, key: str) -> Path:
        """Get file path for specific key."""
        return self.paths[key]


class DataLoader:
    """Utility class for loading CRAFT datasets."""
    
    @staticmethod
    def load_metadata(metadata_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load table metadata with proper typing and error handling.
        
        Args:
            metadata_path: Path to metadata CSV file
            
        Returns:
            DataFrame with table metadata
        """
        try:
            dtypes = {
                "index": int,
                "TableID": str,
                "Table_Title": str, 
                "Table_Headers": str,
                "Table_CellValues": str
            }
            columns = ["index", "TableID", "Table_Title", "Table_Headers", "Table_CellValues"]
            
            metadata = pd.read_csv(
                metadata_path,
                names=columns,
                dtype=dtypes,
                skiprows=1
            )
            
            logger.info(f"Loaded {len(metadata)} tables from {metadata_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {e}")
            raise
    
    @staticmethod  
    def load_questions(questions_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load questions with proper formatting.
        
        Args:
            questions_path: Path to questions file (TSV or JSONL)
            
        Returns:
            DataFrame with questions
        """
        try:
            questions_path = Path(questions_path)
            
            if questions_path.suffix == '.jsonl':
                questions = []
                with open(questions_path, 'r') as f:
                    for line in f:
                        questions.append(json.loads(line))
                df = pd.DataFrame(questions)
            else:
                # Assume TSV format
                df = pd.read_csv(questions_path, sep='\t')
            
            logger.info(f"Loaded {len(df)} questions from {questions_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading questions from {questions_path}: {e}")
            raise
    
    @staticmethod
    def load_pickle(pickle_path: Union[str, Path]) -> any:
        """Load pickle file with error handling."""
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded pickle file from {pickle_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading pickle from {pickle_path}: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: any, pickle_path: Union[str, Path]) -> None:
        """Save data to pickle file with error handling."""
        try:
            pickle_path = Path(pickle_path)
            pickle_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved pickle file to {pickle_path}")
        except Exception as e:
            logger.error(f"Error saving pickle to {pickle_path}: {e}")
            raise


class TableProcessor:
    """Utility class for table processing operations."""
    
    @staticmethod
    def create_table_text(row: pd.Series, include_summary: bool = False) -> str:
        """
        Create text representation of a table.
        
        Args:
            row: Pandas Series with table data
            include_summary: Whether to include generated summaries
            
        Returns:
            Formatted table text
        """
        parts = []
        
        if 'Table_Title' in row and pd.notna(row['Table_Title']):
            parts.append(f"Title: {row['Table_Title']}")
        
        if 'Table_Headers' in row and pd.notna(row['Table_Headers']):
            headers = row['Table_Headers'].replace('|', ' ')
            parts.append(f"Headers: {headers}")
        
        if 'Table_CellValues' in row and pd.notna(row['Table_CellValues']):
            cells = row['Table_CellValues'].replace('|', ' ')
            parts.append(f"Content: {cells}")
        
        if include_summary and 'summary' in row and pd.notna(row['summary']):
            parts.append(f"Summary: {row['summary']}")
        
        return " ".join(parts)
    
    @staticmethod
    def create_mini_table(table_id: str, 
                         top_rows: Dict[str, List[str]], 
                         row_data: Dict[str, Dict], 
                         max_rows: int = 5) -> str:
        """
        Create mini-table from top rows.
        
        Args:
            table_id: Table identifier
            top_rows: Dictionary mapping table IDs to lists of row IDs
            row_data: Dictionary mapping row IDs to row data
            max_rows: Maximum number of rows to include
            
        Returns:
            Mini-table text representation
        """
        if table_id not in top_rows:
            return ""
        
        row_ids = top_rows[table_id][:max_rows]
        row_texts = []
        
        for row_id in row_ids:
            if row_id in row_data:
                row_texts.append(row_data[row_id].get("Row Data", ""))
        
        return " ".join(row_texts)


class ResultsManager:
    """Utility class for managing and saving results."""
    
    @staticmethod
    def save_stage_results(results: Dict, 
                          stage: int,
                          dataset: str, 
                          output_dir: Union[str, Path]) -> Path:
        """
        Save stage results with standardized naming.
        
        Args:
            results: Results dictionary to save
            stage: Stage number (1, 2, or 3)
            dataset: Dataset name
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset.lower()}_stage{stage}_results_{timestamp}.pkl"
        output_path = output_dir / filename
        
        DataLoader.save_pickle(results, output_path)
        return output_path
    
    @staticmethod
    def format_retrieval_results(query_results: Dict, 
                               query_id: str,
                               question: str,
                               gold_table_id: Optional[str] = None) -> str:
        """
        Format retrieval results for text output.
        
        Args:
            query_results: Dictionary with ranked results
            query_id: Query identifier  
            question: Question text
            gold_table_id: Gold standard table ID (if available)
            
        Returns:
            Formatted result string
        """
        # Find gold rank if available
        gold_rank = "N/A"
        if gold_table_id:
            for i, (table_id, score) in enumerate(query_results.items()):
                if table_id == gold_table_id:
                    gold_rank = i + 1
                    break
        
        # Format top results
        top_results = list(query_results.items())[:100]  # Top 100
        results_str = ", ".join([f"({tid}, {score:.4f})" for tid, score in top_results])
        
        return (f"Query Number: {query_id}, QID: {query_id}, "
               f"Gold Rank: {gold_rank}, Question Text: {question}, "
               f"Top 100 TableIds: [{results_str}]")


class EvaluationMetrics:
    """Utility class for computing evaluation metrics."""
    
    @staticmethod
    def compute_recall_at_k(rankings: Dict[str, List[str]], 
                           gold_answers: Dict[str, str], 
                           k_values: List[int] = [1, 10, 50]) -> Dict[int, float]:
        """
        Compute Recall@K metrics.
        
        Args:
            rankings: Dictionary mapping query IDs to ranked table IDs
            gold_answers: Dictionary mapping query IDs to gold table IDs
            k_values: List of K values to compute recall for
            
        Returns:
            Dictionary mapping K values to recall scores
        """
        recalls = {k: 0.0 for k in k_values}
        valid_queries = 0
        
        for query_id, ranked_tables in rankings.items():
            if query_id not in gold_answers:
                continue
                
            gold_table = gold_answers[query_id]
            valid_queries += 1
            
            for k in k_values:
                if gold_table in ranked_tables[:k]:
                    recalls[k] += 1.0
        
        # Convert to percentages
        if valid_queries > 0:
            for k in k_values:
                recalls[k] = (recalls[k] / valid_queries) * 100
        
        return recalls
    
    @staticmethod
    def compute_mrr(rankings: Dict[str, List[str]], 
                   gold_answers: Dict[str, str]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            rankings: Dictionary mapping query IDs to ranked table IDs
            gold_answers: Dictionary mapping query IDs to gold table IDs
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for query_id, ranked_tables in rankings.items():
            if query_id not in gold_answers:
                continue
                
            gold_table = gold_answers[query_id]
            
            for i, table_id in enumerate(ranked_tables):
                if table_id == gold_table:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def get_device() -> str:
    """Get optimal device (GPU if available, otherwise CPU)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


# ── From TableRAG/utils/get_gemini_table_title_description.py ─────────────────

def parse_table_summary(csv_file: Union[str, Path]) -> Dict[str, str]:
    """
    Load Gemini-generated table titles and descriptions from a CSV.
    Returns a dict mapping table index (str) → "Title Description" string.
    Mirrors TableRAG get_gemini_table_title_description.py.
    """
    df = pd.read_csv(csv_file)
    table_info = {
        str(row["Table Index"]): f"{row['Table Title']} {row['Table Description']}"
        for _, row in df.iterrows()
    }
    return table_info


def parse_table_paths(json_file: Union[str, Path]) -> Dict[str, str]:
    """
    Load table ID → file path mapping from a JSON file.
    Returns reversed mapping: file path → table ID.
    Mirrors TableRAG get_gemini_table_title_description.py.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    table_paths = {}
    for entry in data:
        for key, value in entry.items():
            table_paths[str(value)] = str(key)  # file path -> table ID
    return table_paths


# ── From TableRAG/utils/gen_table_description_nq.py ──────────────────────────

def format_output(response: str) -> str:
    """
    Clean and normalise a raw LLM response for table title/description generation.
    Strips whitespace, removes markdown artefacts, and extracts the content
    after chain-of-thought tags.
    Mirrors TableRAG gen_table_description_nq.py.
    """
    response = str(response).strip().replace("\n", "").replace('"', "")
    response = response.replace("*", "")
    if "</think>" in response:
        response = response.split("</think>")[-1]
    elif "</thinking>" in response:
        response = response.split("</thinking>")[-1]
    elif "Table Title:" in response:
        response = "Table Title:" + response.split("Table Title:")[-1]
    return response
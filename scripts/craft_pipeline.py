#!/usr/bin/env python3
"""
CRAFT Pipeline Orchestrator

Main pipeline controller for running CRAFT end-to-end or individual stages.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Import core utilities
from craft_core import (
    CRAFTConfig, DataLoader, TableProcessor, ResultsManager, 
    EvaluationMetrics, setup_logging
)

logger = logging.getLogger(__name__)


class CRAFTPipeline:
    """Main CRAFT pipeline orchestrator."""
    
    def __init__(self, dataset: str = "nq", config_path: Optional[str] = None):
        """
        Initialize CRAFT pipeline.
        
        Args:
            dataset: Dataset name ("nq" or "ottqa")
            config_path: Optional path to custom configuration file
        """
        self.config = CRAFTConfig(dataset)
        self.data_loader = DataLoader()
        self.table_processor = TableProcessor()
        self.results_manager = ResultsManager()
        self.metrics = EvaluationMetrics()
        
        # Load data
        self._load_base_data()
    
    def _load_base_data(self):
        """Load base datasets required for all stages."""
        logger.info("Loading base datasets...")
        
        try:
            self.metadata = self.data_loader.load_metadata(self.config.get_path("metadata"))
            self.questions = self.data_loader.load_questions(self.config.get_path("questions"))
            logger.info("✅ Base data loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load base data: {e}")
            raise
    
    def run_stage1(self, **kwargs) -> Dict[str, Any]:
        """
        Run Stage 1: SPLADE sparse retrieval.
        
        Args:
            **kwargs: Additional parameters for Stage 1
            
        Returns:
            Stage 1 results dictionary
        """
        logger.info("🔍 Running Stage 1: SPLADE Sparse Retrieval")
        
        try:
            # Import from the canonical stage implementations module
            from craft_stages import SpladeRetriever
            
            # Initialize retriever
            retriever = SpladeRetriever(
                model_name=self.config.get_model(1),
                device=kwargs.get('device', 'cpu')
            )
            
            # Create table corpus
            logger.info("Creating table text representations...")
            table_texts = []
            table_ids = []
            
            for _, row in self.metadata.iterrows():
                table_text = self.table_processor.create_table_text(row, include_summary=True)
                table_texts.append(table_text)
                table_ids.append(row['TableID'])
            
            # Build index
            logger.info("Building SPLADE index...")
            retriever.build_index(table_texts, table_ids)
            
            # Process queries
            logger.info("Processing queries...")
            results = {}
            
            for _, query_row in self.questions.iterrows():
                query_id = str(query_row['query_id'])
                question = query_row['question']
                
                # Retrieve candidates
                candidates = retriever.retrieve(
                    question, 
                    top_k=self.config.stage1_candidates
                )
                
                results[query_id] = candidates
            
            # Save results
            output_path = self.results_manager.save_stage_results(
                results, 1, self.config.dataset, self.config.get_path("stage1_output").parent
            )
            
            logger.info(f"✅ Stage 1 complete. Results saved to {output_path}")
            return {"results": results, "output_path": output_path}
            
        except Exception as e:
            logger.error(f"❌ Stage 1 failed: {e}")
            raise
    
    def run_stage2(self, stage1_results: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Run Stage 2: Dense semantic reranking.
        
        Args:
            stage1_results: Stage 1 results (if None, will load from file)
            **kwargs: Additional parameters for Stage 2
            
        Returns:
            Stage 2 results dictionary
        """
        logger.info("🧠 Running Stage 2: Dense Semantic Reranking")
        
        try:
            # Load Stage 1 results if not provided
            if stage1_results is None:
                stage1_path = kwargs.get('stage1_path')
                if not stage1_path:
                    raise ValueError("Either stage1_results or stage1_path must be provided")
                stage1_results = self.data_loader.load_pickle(stage1_path)
            
            # Import from the canonical stage implementations module
            from craft_stages import DenseReranker
            
            # Initialize reranker
            reranker = DenseReranker(
                model_name=self.config.get_model(2),
                device=kwargs.get('device', 'cpu')
            )
            
            # Load additional data for mini-tables
            top_rows_path = kwargs.get('top_rows_path')
            row_data_path = kwargs.get('row_data_path')
            
            if top_rows_path and row_data_path:
                top_rows = self.data_loader.load_pickle(top_rows_path)
                
                with open(row_data_path, 'r') as f:
                    import json
                    row_data = json.load(f)
                    row_data_dict = {entry['Row Number']: entry for entry in row_data}
            else:
                logger.warning("No row data provided, using full table content")
                top_rows, row_data_dict = None, None
            
            # Process queries
            logger.info("Reranking with dense embeddings...")
            results = {}
            
            # Build index lookup once (avoids O(N²) per-iteration .astype(str) call)
            question_index = set(self.questions.index.astype(str))
            questions_str_index = self.questions.copy()
            questions_str_index.index = questions_str_index.index.astype(str)
            
            for query_id, candidates in stage1_results.items():
                if query_id not in question_index:
                    continue
                
                query_row = questions_str_index.loc[query_id]
                question = query_row['question']
                
                # Create mini-tables or use full tables
                if top_rows and row_data_dict:
                    # Create mini-tables
                    candidate_texts = []
                    for table_id, _ in candidates:
                        mini_table = self.table_processor.create_mini_table(
                            table_id, top_rows.get(query_id, {}), row_data_dict,
                            max_rows=self.config.mini_table_rows
                        )
                        
                        # Add table title if available
                        table_info = self.metadata[self.metadata['TableID'] == table_id]
                        if not table_info.empty:
                            title = table_info.iloc[0]['Table_Title']
                            mini_table = f"Title: {title}. Content: {mini_table}"
                        
                        candidate_texts.append(mini_table)
                else:
                    # Use full table content
                    candidate_texts = []
                    for table_id, _ in candidates:
                        table_row = self.metadata[self.metadata['TableID'] == table_id].iloc[0]
                        table_text = self.table_processor.create_table_text(table_row)
                        candidate_texts.append(table_text)
                
                # Rerank
                reranked = reranker.rerank(
                    question, 
                    candidate_texts,
                    [table_id for table_id, _ in candidates],
                    top_k=self.config.stage2_candidates
                )
                
                results[query_id] = reranked
            
            # Save results
            output_path = self.results_manager.save_stage_results(
                results, 2, self.config.dataset, self.config.get_path("stage2_output").parent
            )
            
            logger.info(f"✅ Stage 2 complete. Results saved to {output_path}")
            return {"results": results, "output_path": output_path}
            
        except Exception as e:
            logger.error(f"❌ Stage 2 failed: {e}")
            raise
    
    def run_stage3(self, stage2_results: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Run Stage 3: Neural reranking.
        
        Args:
            stage2_results: Stage 2 results (if None, will load from file)
            **kwargs: Additional parameters for Stage 3
            
        Returns:
            Stage 3 results dictionary
        """
        logger.info("🚀 Running Stage 3: Neural Reranking")
        
        try:
            # Load Stage 2 results if not provided
            if stage2_results is None:
                stage2_path = kwargs.get('stage2_path')
                if not stage2_path:
                    raise ValueError("Either stage2_results or stage2_path must be provided")
                stage2_results = self.data_loader.load_pickle(stage2_path)
            
            # Import from the canonical stage implementations module
            from craft_stages import NeuralReranker
            
            # Initialize reranker (API key is read from environment variables)
            use_gemini = kwargs.get('use_gemini', self.config.dataset == 'ottqa')
            
            reranker = NeuralReranker(
                model_name=self.config.get_model(3),
                use_gemini=use_gemini
            )
            
            # Process queries
            logger.info("Final neural reranking...")
            results = {}
            
            # Build index lookup once (avoids O(N²) per-iteration .astype(str) call)
            question_index = set(self.questions.index.astype(str))
            questions_str_index = self.questions.copy()
            questions_str_index.index = questions_str_index.index.astype(str)
            
            for query_id, candidates in stage2_results.items():
                if query_id not in question_index:
                    continue
                
                query_row = questions_str_index.loc[query_id]
                question = query_row['question']
                
                # Get table texts
                candidate_texts = []
                table_ids = []
                
                for table_id, _ in candidates:
                    table_row = self.metadata[self.metadata['TableID'] == table_id]
                    if not table_row.empty:
                        table_text = self.table_processor.create_table_text(table_row.iloc[0])
                        candidate_texts.append(table_text)
                        table_ids.append(table_id)
                
                # Final reranking
                final_ranking = reranker.rerank(question, candidate_texts, table_ids)
                results[query_id] = final_ranking
            
            # Save results
            output_path = self.results_manager.save_stage_results(
                results, 3, self.config.dataset, self.config.get_path("stage3_output").parent
            )
            
            logger.info(f"✅ Stage 3 complete. Results saved to {output_path}")
            return {"results": results, "output_path": output_path}
            
        except Exception as e:
            logger.error(f"❌ Stage 3 failed: {e}")
            raise
    
    def run_full_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Run complete CRAFT pipeline (all 3 stages).
        
        Args:
            **kwargs: Parameters for all stages
            
        Returns:
            Complete pipeline results
        """
        logger.info("🏭 Running full CRAFT pipeline")
        
        # Run Stage 1
        stage1_results = self.run_stage1(**kwargs)
        
        # Run Stage 2
        stage2_results = self.run_stage2(stage1_results["results"], **kwargs)
        
        # Run Stage 3
        stage3_results = self.run_stage3(stage2_results["results"], **kwargs)
        
        # Compute final metrics if gold answers available
        if 'gold_table_id' in self.questions.columns:
            gold_answers = dict(zip(
                self.questions.index.astype(str), 
                self.questions['gold_table_id']
            ))
            
            # Convert results to rankings
            rankings = {}
            for query_id, ranked_results in stage3_results["results"].items():
                rankings[query_id] = [table_id for table_id, _ in ranked_results]
            
            # Compute metrics
            recall_scores = self.metrics.compute_recall_at_k(rankings, gold_answers)
            mrr_score = self.metrics.compute_mrr(rankings, gold_answers)
            
            logger.info("📊 Final Pipeline Metrics:")
            for k, score in recall_scores.items():
                logger.info(f"   Recall@{k}: {score:.2f}%")
            logger.info(f"   MRR: {mrr_score:.4f}")
        
        return {
            "stage1": stage1_results,
            "stage2": stage2_results, 
            "stage3": stage3_results
        }
    
    def evaluate_results(self, results_path: str, gold_answers_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate pipeline results against gold standard.
        
        Args:
            results_path: Path to results file
            gold_answers_path: Optional path to gold answers
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("📊 Evaluating pipeline results")
        
        results = self.data_loader.load_pickle(results_path)
        
        # Use gold answers from questions if not provided separately
        if gold_answers_path:
            with open(gold_answers_path, 'r') as f:
                import json
                gold_data = json.load(f)
                gold_answers = {str(item['query_id']): item['gold_table_id'] for item in gold_data}
        elif 'gold_table_id' in self.questions.columns:
            gold_answers = dict(zip(
                self.questions.index.astype(str),
                self.questions['gold_table_id']
            ))
        else:
            raise ValueError("No gold answers available for evaluation")
        
        # Convert results to rankings
        rankings = {}
        for query_id, ranked_results in results.items():
            if isinstance(ranked_results, list):
                rankings[query_id] = [table_id for table_id, _ in ranked_results]
            else:
                rankings[query_id] = list(ranked_results.keys())
        
        # Compute metrics
        recall_scores = self.metrics.compute_recall_at_k(rankings, gold_answers)
        mrr_score = self.metrics.compute_mrr(rankings, gold_answers)
        
        metrics = {
            'recall_at_1': recall_scores.get(1, 0.0),
            'recall_at_10': recall_scores.get(10, 0.0), 
            'recall_at_50': recall_scores.get(50, 0.0),
            'mrr': mrr_score
        }
        
        logger.info("📈 Evaluation Results:")
        for metric, score in metrics.items():
            logger.info(f"   {metric}: {score:.4f}")
        
        return metrics


def main():
    """Command-line interface for CRAFT pipeline."""
    parser = argparse.ArgumentParser(description="CRAFT: Cascaded Retrieval for Tabular QA")
    
    parser.add_argument("--dataset", choices=["nq", "ottqa"], default="nq",
                       help="Dataset to use (default: nq)")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all",
                       help="Stage to run (default: all)")
    parser.add_argument("--device", default="cpu", 
                       help="Device to use (default: cpu)")
    parser.add_argument("--log-level", default="INFO", 
                       help="Logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log file path")
    
    # Stage-specific arguments
    parser.add_argument("--stage1-path", help="Path to Stage 1 results")
    parser.add_argument("--stage2-path", help="Path to Stage 2 results")
    parser.add_argument("--top-rows-path", help="Path to top rows data")
    parser.add_argument("--row-data-path", help="Path to row data")
    parser.add_argument("--api-key", help="API key for Stage 3")
    parser.add_argument("--use-gemini", action="store_true", 
                       help="Use Gemini instead of OpenAI for Stage 3")
    
    # Evaluation
    parser.add_argument("--evaluate", help="Path to results file to evaluate")
    parser.add_argument("--gold-answers", help="Path to gold answers file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Initialize pipeline
        pipeline = CRAFTPipeline(args.dataset)
        
        # Run evaluation if requested
        if args.evaluate:
            metrics = pipeline.evaluate_results(args.evaluate, args.gold_answers)
            return metrics
        
        # Run specified stage(s)
        kwargs = {
            'device': args.device,
            'stage1_path': args.stage1_path,
            'stage2_path': args.stage2_path,
            'top_rows_path': args.top_rows_path,
            'row_data_path': args.row_data_path,
            'api_key': args.api_key,
            'use_gemini': args.use_gemini
        }
        
        if args.stage == "1":
            results = pipeline.run_stage1(**kwargs)
        elif args.stage == "2":
            results = pipeline.run_stage2(**kwargs)
        elif args.stage == "3":
            results = pipeline.run_stage3(**kwargs)
        else:  # "all"
            results = pipeline.run_full_pipeline(**kwargs)
        
        logger.info("🎉 CRAFT pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
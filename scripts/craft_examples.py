#!/usr/bin/env python3
"""
CRAFT Modular Usage Examples

This script demonstrates how to use CRAFT's modular components
for different use cases and configurations.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from craft_core import CRAFTConfig, DataLoader, setup_logging
from craft_stages import create_splade_retriever, create_dense_reranker, create_neural_reranker
from craft_pipeline import CRAFTPipeline

def example_individual_components():
    """Example: Using individual CRAFT components."""
    print("🔧 Example: Individual Components Usage")
    
    # Setup
    setup_logging("INFO")
    config = CRAFTConfig("nq")
    data_loader = DataLoader()
    
    # Load data
    metadata = data_loader.load_metadata(config.get_path("metadata"))
    questions = data_loader.load_questions(config.get_path("questions"))
    
    print(f"✅ Loaded {len(metadata)} tables, {len(questions)} questions")
    
    # Example: Create individual stage components
    
    # Stage 1: SPLADE retriever
    stage1_retriever = create_splade_retriever(model_name="splade-v2")
    print("✅ SPLADE retriever created")
    
    # Stage 2: Dense reranker  
    stage2_reranker = create_dense_reranker(model_name="sentence-transformers/all-mpnet-base-v2")
    print("✅ Dense reranker created")
    
    # Stage 3: Neural reranker (reads OPENAI_API_KEY from environment)
    if os.getenv("OPENAI_API_KEY"):
        stage3_reranker = create_neural_reranker(
            model_name="text-embedding-3-large",
            use_gemini=False
        )
        print("✅ Neural reranker created")
    else:
        print("⚠️ OPENAI_API_KEY not set, skipping Stage 3")
    
    return metadata, questions


def example_pipeline_usage():
    """Example: Using the full CRAFT pipeline."""
    print("\n🏭 Example: Full Pipeline Usage")
    
    # Initialize pipeline
    pipeline = CRAFTPipeline(dataset="nq")
    print("✅ Pipeline initialized")
    
    # Run individual stages (would require appropriate data files)
    print("📝 Individual stage usage:")
    print("   pipeline.run_stage1()")
    print("   pipeline.run_stage2(stage1_path='path/to/stage1.pkl')")
    print("   pipeline.run_stage3(stage2_path='path/to/stage2.pkl', api_key='your-key')")
    
    # Or run full pipeline
    print("📝 Full pipeline usage:")
    print("   pipeline.run_full_pipeline(api_key='your-key')")
    
    return pipeline


def example_evaluation():
    """Example: Evaluating CRAFT results."""
    print("\n📊 Example: Results Evaluation")
    
    pipeline = CRAFTPipeline(dataset="nq")
    
    # Example evaluation (would require results file)
    print("📝 Evaluation usage:")
    print("   metrics = pipeline.evaluate_results('path/to/results.pkl')")
    print("   print(f'Recall@1: {metrics[\"recall_at_1\"]:.2f}%')")


def example_configuration():
    """Example: Custom configuration."""
    print("\n⚙️ Example: Custom Configuration")
    
    # Custom configuration for different datasets
    nq_config = CRAFTConfig("nq")
    ottqa_config = CRAFTConfig("ottqa")
    
    print(f"NQ Stage 1 model: {nq_config.get_model(1)}")
    print(f"OTT-QA Stage 2 model: {ottqa_config.get_model(2)}")
    
    # Custom paths
    print(f"NQ metadata path: {nq_config.get_path('metadata')}")
    print(f"Stage 1 candidates: {nq_config.stage1_candidates}")


def example_notebook_integration():
    """Example: Using components in notebooks."""
    print("\n📓 Example: Notebook Integration")
    
    print("Jupyter notebook usage:")
    print("""
# Cell 1: Setup
from craft_core import CRAFTConfig, DataLoader
from craft_stages import setup_nq_pipeline

config = CRAFTConfig("nq")
pipeline_config = setup_nq_pipeline()

# Cell 2: Load data
data_loader = DataLoader()
metadata = data_loader.load_metadata(config.get_path("metadata"))

# Cell 3: Process stage
from craft_stages import create_splade_retriever
retriever = create_splade_retriever(pipeline_config['stage1'])
# ... continue processing
""")


def main():
    """Run all examples."""
    print("🚀 CRAFT Modular Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_individual_components()
        example_pipeline_usage()
        example_evaluation()
        example_configuration()
        example_notebook_integration()
        
        print("\n🎉 All examples completed!")
        
        print("\n📋 Quick Reference:")
        print("   Individual components: craft_stages.py")
        print("   Core utilities: craft_core.py")
        print("   Full pipeline: craft_pipeline.py")
        print("   Notebooks: stage{1,2,3}_*.ipynb")
        print("   QA evaluation: qa_evaluation.py")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
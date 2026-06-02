"""
CRAFT package setup.
After running `pip install -e .`, you can import from anywhere:
  from pipeline.retrieval.stage1 import SpladeRetriever
  from pipeline.config import load_config
"""
from setuptools import setup, find_packages

setup(
    name="craft-tabular-qa",
    version="1.0.0",
    description="CRAFT: Training-Free Cascaded Retrieval for Tabular QA (ACL 2025)",
    author="Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta",
    license="MIT",
    # The pipeline/ package lives in the repo root
    packages=find_packages(exclude=["scripts", "scripts.*", "craft", "craft.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "sentence-transformers>=2.2.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        # Install these if you want Stage 3 with OpenAI
        "openai": ["openai>=1.0.0"],
        # Install these if you want Stage 3 with Gemini
        "gemini": ["google-generativeai>=0.3.0"],
        # Full install with everything
        "all": [
            "openai>=1.0.0",
            "google-generativeai>=0.3.0",
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
        ],
        "dev": ["pytest>=7.0.0", "jupyter>=1.0.0"],
    },
)

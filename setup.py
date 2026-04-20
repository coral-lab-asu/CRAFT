"""
CRAFT package setup.
Installs the scripts/ directory as the 'craft' package so that
  from craft_core import CRAFTConfig
  from craft_stages import SpladeRetriever
work without manual sys.path manipulation.
"""
from setuptools import setup, find_packages

setup(
    name="craft-tabular-qa",
    version="1.0.0",
    description="CRAFT: Training-Free Cascaded Retrieval for Tabular QA (ACL 2025)",
    author="Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta",
    license="MIT",
    packages=find_packages(where="scripts"),
    package_dir={"": "scripts"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "sentence-transformers>=2.2.0",
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
        "requests>=2.28.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "datasets>=2.0.0",
        "tiktoken>=0.5.0",
        "faiss-cpu>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "jupyter>=1.0.0"],
    },
)

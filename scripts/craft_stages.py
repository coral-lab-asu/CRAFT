"""
CRAFT Stage Implementations

Modular implementations for each CRAFT pipeline stage.
These can be used in notebooks, scripts, or the main pipeline.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── From TableRAG/utils/simple_retrieval.py ─────────────────────────────────
def _cosine_similarity(embeddings: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between a 2-D tensor of candidate embeddings
    and a 1-D query embedding.  Directly mirrors TableRAG simple_retrieval.py.
    """
    return torch.nn.functional.cosine_similarity(query.unsqueeze(0), embeddings, dim=-1)
# ─────────────────────────────────────────────────────────────────────────────

def _api_call_with_backoff(fn, max_retries: int = 5, base_delay: float = 1.0):
    """
    Execute fn() with exponential backoff on rate-limit / transient errors.
    Raises the last exception if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"API call failed ({exc}). Retrying in {delay:.1f}s...")
            time.sleep(delay)


class SpladeRetriever:
    """Stage 1: SPLADE sparse retrieval implementation."""
    
    def __init__(self, model_name: str = "naver-splade/splade-v2-distil", device: str = "cpu"):
        """Initialize SPLADE retriever."""
        self.model_name = model_name
        self.device = device
        self.index = None
        self.table_ids = None
        
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info(f"✅ SPLADE model {model_name} loaded from HuggingFace")
        except Exception as e:
            logger.error(f"❌ Failed to load SPLADE model: {e}")
            raise
    
    def _encode_splade(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Encode texts with SPLADE to produce sparse lexical vectors."""
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # SPLADE aggregation: max-pooling over tokens then log-saturation
            vectors = torch.log(1 + torch.relu(outputs.logits))
            vectors = torch.max(vectors * inputs["attention_mask"].unsqueeze(-1), dim=1).values
            all_vectors.append(vectors.cpu().float().numpy())
        return np.vstack(all_vectors)
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build SPLADE index from documents."""
        logger.info(f"Building SPLADE index for {len(documents)} documents...")
        self.table_ids = list(doc_ids)
        self.doc_vectors = self._encode_splade(documents)
        logger.info("✅ SPLADE index built")
    
    def retrieve(self, query: str, top_k: int = 5000) -> List[Tuple[str, float]]:
        """Retrieve top-k candidates for query using dot-product over sparse vectors."""
        if self.doc_vectors is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_vec = self._encode_splade([query])[0]  # (vocab_size,)
        scores = self.doc_vectors @ query_vec          # (N,)
        
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(self.table_ids[i], float(scores[i])) for i in top_indices]


class DenseReranker:
    """Stage 2: Dense semantic reranking implementation."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize dense reranker."""
        self.model_name = model_name
        self.device = device
        
        try:
            if "jina" in model_name.lower():
                # Use JINA embeddings
                import requests
                self.use_jina = True
                self.jina_url = "https://api.jina.ai/v1/embeddings"
                logger.info(f"✅ JINA embeddings configured: {model_name}")
            else:
                # Use Sentence Transformers
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=device)
                self.use_jina = False
                logger.info(f"✅ Sentence Transformer loaded: {model_name}")
                
        except ImportError as e:
            logger.error(f"❌ Required library not available: {e}")
            raise
    
    def _encode_texts(self, texts: List[str]):
        """Encode texts to embeddings, returning a torch.Tensor."""
        if self.use_jina:
            # Use JINA API
            import os
            api_key = os.getenv("JINA_API_KEY")
            if not api_key:
                raise ValueError("JINA_API_KEY environment variable required")
            
            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "input": texts,
                "model": self.model_name
            }
            
            response = requests.post(self.jina_url, headers=headers, json=data)
            result = response.json()
            
            embeddings = [item["embedding"] for item in result["data"]]
            return torch.tensor(embeddings)
        else:
            # Use Sentence Transformers — returns torch.Tensor with convert_to_tensor=True
            return self.model.encode(texts, convert_to_tensor=True)
    def rerank(self, 
              query: str, 
              candidate_texts: List[str], 
              candidate_ids: List[str],
              top_k: int = 100) -> List[Tuple[str, float]]:
        """Rerank candidates using cosine similarity (TableRAG simple_retrieval.py approach)."""
        
        query_embedding = self._encode_texts([query])[0]         # (D,)
        candidate_embeddings = self._encode_texts(candidate_texts) # (N, D)
        
        # torch.nn.functional.cosine_similarity — mirrors TableRAG simple_retrieval.py
        cosine_scores = _cosine_similarity(candidate_embeddings, query_embedding)
        
        top_indices = torch.topk(cosine_scores, min(top_k, len(candidate_ids))).indices.tolist()
        
        return [(candidate_ids[i], float(cosine_scores[i])) for i in top_indices]
class NeuralReranker:
    """Stage 3: Neural reranking implementation."""
    
    def __init__(self, 
                 model_name: str, 
                 use_gemini: bool = False):
        """
        Initialize neural reranker.  API keys are read from environment
        variables (OPENAI_API_KEY / GEMINI_API_KEY) to avoid passing
        credentials as constructor arguments.
        """
        self.model_name = model_name
        self.use_gemini = use_gemini
        
        if use_gemini:
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable is required")
                genai.configure(api_key=api_key)
                self.client = genai
                logger.info(f"✅ Gemini configured: {model_name}")
            except ImportError:
                logger.error("❌ google-generativeai not available")
                raise
        else:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required")
                self.client = OpenAI(api_key=api_key)
                logger.info(f"✅ OpenAI configured: {model_name}")
            except ImportError:
                logger.error("❌ openai not available")
                raise
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings from API as a torch.Tensor."""
        if self.use_gemini:
            embeddings = []
            for text in texts:
                result = self.client.embed_content(
                    model=self.model_name,
                    content=text
                )
                embeddings.append(result['embedding'])
            return torch.tensor(embeddings)
        else:
            # OpenAI
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return torch.tensor(embeddings)

    def rerank(self, query: str, candidate_texts: List[str], candidate_ids: List[str],
              top_k: int = 100, batch_size: int = 50) -> List[Tuple[str, float]]:
        """Rerank candidates using neural embeddings with cosine similarity (TableRAG approach)."""
        
        query_embedding = self._get_embeddings([query])[0]  # (D,)
        
        all_embeddings = []
        
        
        for i in range(0, len(candidate_texts), batch_size):
            batch_texts = candidate_texts[i:i+batch_size]
            batch_embeddings = _api_call_with_backoff(
                lambda bt=batch_texts: self._get_embeddings(bt)
            )
            all_embeddings.append(batch_embeddings)
        
        candidate_embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
        
        # torch.nn.functional.cosine_similarity — mirrors TableRAG simple_retrieval.py
        cosine_scores = _cosine_similarity(candidate_embeddings, query_embedding)
        
        top_indices = torch.argsort(cosine_scores, descending=True).tolist()
        
        return [(candidate_ids[i], float(cosine_scores[i])) for i in top_indices]


def create_neural_reranker(model_name: str, use_gemini: bool = False) -> NeuralReranker:
    """Create neural reranker instance (reads API key from environment)."""
    return NeuralReranker(model_name, use_gemini)


# Quick setup functions for common configurations
def setup_nq_pipeline():
    """Setup pipeline for NQ dataset."""
    return {
        'stage1': 'naver-splade/splade-v2-distil',
        'stage2': 'sentence-transformers/all-mpnet-base-v2',
        'stage3': 'text-embedding-3-large',
        'use_gemini': False
    }


def setup_ottqa_pipeline():
    """Setup pipeline for OTT-QA dataset."""
    return {
        'stage1': 'naver-splade/splade-v2-distil',
        'stage2': 'jinaai/jina-embeddings-v3',
        'stage3': 'models/gemini-embedding-001',
        'use_gemini': True
    }

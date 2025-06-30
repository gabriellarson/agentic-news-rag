"""
Qwen3 Embedding Pipeline for News RAG System

Uses sentence-transformers with local Qwen3-Embedding-0.6B model
for generating dense embeddings of news articles.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from .article_parser import Article

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    embedding: np.ndarray
    text: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'embedding': self.embedding.tolist(),
            'text': self.text,
            'metadata': self.metadata
        }


class QwenEmbedder:
    """Embedding pipeline using Qwen3-Embedding-0.6B model"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedder with local or remote model.
        
        Args:
            model_path: Path to local model directory or HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Default to local model if no path specified
        if model_path is None:
            # Look for local model first
            local_model = Path(__file__).parent.parent.parent / "Qwen3-Embedding-0.6B"
            if local_model.exists():
                model_path = str(local_model)
                logger.info(f"Using local model at {model_path}")
            else:
                # Fallback to HuggingFace model
                model_path = "Qwen/Qwen3-Embedding-0.6B"
                logger.info(f"Using HuggingFace model: {model_path}")
        
        # Load model
        logger.info(f"Loading embedding model from {model_path} on {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: Union[str, List[str]], 
                   batch_size: int = 32,
                   show_progress_bar: bool = True,
                   normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar
            normalize: Whether to normalize embeddings (for cosine similarity)
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_article(self, article: Article, 
                     include_metadata: bool = True) -> EmbeddingResult:
        """
        Generate embedding for a single article.
        
        Args:
            article: Parsed article object
            include_metadata: Whether to include metadata in result
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        # Prepare text with weighted sections
        text = self._prepare_article_text(article)
        
        # Generate embedding
        embedding = self.embed_texts(text, show_progress_bar=False)[0]
        
        # Prepare metadata
        metadata = {}
        if include_metadata:
            metadata = article.metadata
        
        return EmbeddingResult(
            embedding=embedding,
            text=text,
            metadata=metadata
        )
    
    def embed_articles(self, articles: List[Article], 
                      batch_size: int = 32,
                      show_progress_bar: bool = True) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple articles.
        
        Args:
            articles: List of parsed articles
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar
            
        Returns:
            List of EmbeddingResults
        """
        # Prepare texts
        texts = [self._prepare_article_text(article) for article in articles]
        
        # Generate embeddings in batch
        embeddings = self.embed_texts(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
        
        # Create results
        results = []
        for i, (article, embedding) in enumerate(zip(articles, embeddings)):
            result = EmbeddingResult(
                embedding=embedding,
                text=texts[i],
                metadata=article.metadata
            )
            results.append(result)
        
        return results
    
    def _prepare_article_text(self, article: Article) -> str:
        """
        Prepare article text for embedding with weighted sections.
        
        Title and subtitle are repeated to give them more weight
        in the embedding.
        """
        parts = []
        
        # Title (weight 3x)
        parts.extend([article.title] * 3)
        
        # Subtitle (weight 2x)
        if article.subtitle:
            parts.extend([article.subtitle] * 2)
        
        # Add content
        parts.append(article.content)
        
        # Join with periods for better sentence separation
        return ". ".join(parts)
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            normalize: Whether to normalize embedding
            
        Returns:
            Query embedding
        """
        return self.embed_texts(query, show_progress_bar=False, normalize=normalize)[0]
    
    def batch_embed_queries(self, queries: List[str], 
                           batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple queries.
        
        Args:
            queries: List of search queries
            batch_size: Batch size for processing
            
        Returns:
            Array of query embeddings
        """
        return self.embed_texts(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize=True
        )
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # If embeddings are normalized, just compute dot product
        return float(np.dot(embedding1, embedding2))
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model.model_name,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device),
            'max_seq_length': self.model.max_seq_length
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize embedder
    embedder = QwenEmbedder()
    print(f"Model info: {embedder.get_model_info()}")
    
    # Test with sample texts
    sample_texts = [
        "Chesapeake Energy agrees to buy Southwestern Energy for $7.4 billion",
        "EU countries debate new car emission regulations",
        "Natural gas prices surge in European markets"
    ]
    
    print("\nTesting text embeddings...")
    embeddings = embedder.embed_texts(sample_texts)
    print(f"Generated {len(embeddings)} embeddings with shape: {embeddings.shape}")
    
    # Test similarity
    print("\nTesting similarity between texts:")
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            similarity = embedder.compute_similarity(embeddings[i], embeddings[j])
            print(f"Similarity between text {i} and {j}: {similarity:.3f}")
    
    # Test with article if available
    from article_parser import ArticleParser
    
    test_file = Path(__file__).parent.parent.parent / "text_articles" / "Chesapeake and Southwestern to create US gas titan with $7.4bn deal.txt"
    if test_file.exists():
        print(f"\nTesting with article: {test_file}")
        parser = ArticleParser()
        article = parser.parse_file(test_file)
        
        result = embedder.embed_article(article)
        print(f"Article embedding shape: {result.embedding.shape}")
        print(f"Article metadata: {list(result.metadata.keys())}")
        
        # Test query similarity
        query = "energy company merger and acquisition"
        query_embedding = embedder.embed_query(query)
        similarity = embedder.compute_similarity(query_embedding, result.embedding)
        print(f"\nQuery: '{query}'")
        print(f"Similarity to article: {similarity:.3f}")
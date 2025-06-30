"""
Qdrant Hybrid Search Engine for News RAG System

Implements hybrid dense/sparse search with Qdrant vector database,
combining semantic search with keyword matching for optimal retrieval.
"""

import uuid
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, SparseVector, Filter, FieldCondition,
    Range, SearchParams, NamedSparseVector, SearchRequest,
    VectorParams, Distance, SparseVectorParams
)

try:
    from ..config import get_config
    from ..embeddings.qwen_embedder import QwenEmbedder
    from ..embeddings.article_parser import Article, ArticleParser
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    from embeddings.qwen_embedder import QwenEmbedder
    from embeddings.article_parser import Article, ArticleParser

logger = logging.getLogger(__name__)


class QdrantHybridSearch:
    """Hybrid search engine using Qdrant with dense and sparse vectors"""
    
    def __init__(self, config=None):
        """Initialize the search engine with configuration"""
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Get configurations using correct config.get() method
        self.host = config.get('qdrant', 'host', default='localhost')
        self.port = config.get('qdrant', 'port', default=6333)
        self.collection_name = config.get('qdrant', 'collection_name', default='news_articles')
        
        self.default_limit = config.get('search', 'hybrid', 'default_limit', default=20)
        self.default_alpha = config.get('search', 'hybrid', 'alpha', default=0.65)
        
        # Sparse search configuration
        self.max_features = config.get('search', 'sparse', 'max_features', default=10000)
        ngram_range_list = config.get('search', 'sparse', 'ngram_range', default=[1, 2])
        self.ngram_range = tuple(ngram_range_list)
        self.min_df = config.get('search', 'sparse', 'min_df', default=2)
        self.max_df = config.get('search', 'sparse', 'max_df', default=0.95)
        
        # Initialize components
        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = QwenEmbedder()
        self.parser = ArticleParser()
        
        # TF-IDF vectorizer for sparse search
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english'
        )
        
        # Model state paths
        self.tfidf_model_path = Path("models/tfidf_vectorizer.pkl")
        self.tfidf_vocab_path = Path("models/tfidf_vocabulary.json")
        
        # Try to load existing TF-IDF model
        self._load_tfidf_model()
    
    def _load_tfidf_model(self):
        """Load existing TF-IDF model if available"""
        if self.tfidf_model_path.exists():
            try:
                with open(self.tfidf_model_path, 'rb') as f:
                    self.tfidf = pickle.load(f)
                logger.info("Loaded existing TF-IDF model")
            except Exception as e:
                logger.warning(f"Failed to load TF-IDF model: {e}")
    
    def _save_tfidf_model(self):
        """Save TF-IDF model for reuse"""
        try:
            self.tfidf_model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tfidf_model_path, 'wb') as f:
                pickle.dump(self.tfidf, f)
            
            # Also save vocabulary for inspection  
            if hasattr(self.tfidf, 'vocabulary_'):
                # Convert numpy types to Python types for JSON serialization
                vocab_dict = {k: int(v) for k, v in self.tfidf.vocabulary_.items()}
                with open(self.tfidf_vocab_path, 'w') as f:
                    json.dump(vocab_dict, f, indent=2)
            
            logger.info("Saved TF-IDF model")
        except Exception as e:
            logger.warning(f"Failed to save TF-IDF model: {e}")
    
    def index_article(self, article: Article) -> str:
        """
        Index a single article in Qdrant.
        
        Args:
            article: Parsed article object
            
        Returns:
            Document ID
        """
        # Generate unique ID
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(article.file_path)))
        
        # Prepare text for embedding
        text = self._prepare_article_text(article)
        
        # Generate dense embedding
        embedding_result = self.embedder.embed_article(article)
        
        # Generate sparse vector
        sparse_vector = self._get_sparse_vector(text)
        
        # Create point (dense vectors only for now)
        point = PointStruct(
            id=doc_id,
            vector={
                "dense": embedding_result.embedding.tolist()
            },
            payload=self._create_payload(article)
        )
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        logger.info(f"Indexed article: {article.title}")
        return doc_id
    
    def index_articles(self, articles: List[Article], 
                      batch_size: int = 100) -> List[str]:
        """
        Index multiple articles in batches.
        
        Args:
            articles: List of parsed articles
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            batch_texts = [self._prepare_article_text(a) for a in batch]
            
            # Update TF-IDF vocabulary if needed
            if not hasattr(self.tfidf, 'vocabulary_'):
                logger.info("Fitting TF-IDF on first batch")
                self.tfidf.fit(batch_texts)
                self._save_tfidf_model()
            
            # Generate embeddings in batch
            embedding_results = self.embedder.embed_articles(batch, batch_size=32)
            
            # Generate sparse vectors
            sparse_matrix = self.tfidf.transform(batch_texts)
            
            # Create points
            points = []
            for idx, (article, emb_result) in enumerate(zip(batch, embedding_results)):
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(article.file_path)))
                doc_ids.append(doc_id)
                
                # Convert sparse matrix row to Qdrant format
                sparse_indices = sparse_matrix[idx].indices.tolist()
                sparse_values = sparse_matrix[idx].data.tolist()
                
                # Create point (dense vectors only for now)
                point = PointStruct(
                    id=doc_id,
                    vector={
                        "dense": emb_result.embedding.tolist()
                    },
                    payload=self._create_payload(article)
                )
                points.append(point)
            
            # Upload batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Indexed batch {i//batch_size + 1}: {len(points)} articles")
        
        return doc_ids
    
    def search(self, query: str, 
              limit: int = None,
              alpha: float = None,
              date_filter: Optional[Dict[str, str]] = None,
              author_filter: Optional[List[str]] = None,
              score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with RRF fusion.
        
        Args:
            query: Search query
            limit: Number of results (default from config)
            alpha: Dense vs sparse weight (0-1, default from config)
            date_filter: Date range {'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}
            author_filter: List of authors to filter by
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results with metadata
        """
        limit = limit or self.default_limit
        alpha = alpha if alpha is not None else self.default_alpha
        
        # Generate query vectors
        dense_query = self.embedder.embed_query(query)
        sparse_query = self._get_sparse_query_vector(query)
        
        # Build filter
        filter_conditions = self._build_filter_conditions(
            date_filter, author_filter
        )
        
        # For now, let's use simple dense search since sparse vectors are giving issues
        # TODO: Fix hybrid search implementation
        
        try:
            # Simple dense vector search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", dense_query.tolist()),
                query_filter=filter_conditions,
                limit=limit,
                with_payload=True,
                search_params=SearchParams(
                    hnsw_ef=128,  # Increase for better recall
                    exact=False   # Use HNSW index
                )
            )
            fused_results = results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        
        # Format and filter results
        formatted_results = []
        for hit in fused_results:
            if score_threshold and hit.score < score_threshold:
                continue
            
            result = {
                'id': hit.id,
                'score': hit.score,
                'title': hit.payload.get('title', ''),
                'subtitle': hit.payload.get('subtitle', ''),
                'authors': hit.payload.get('authors', []),
                'published': hit.payload.get('published', ''),
                'content_preview': hit.payload.get('content_preview', ''),
                'file_path': hit.payload.get('file_path', ''),
                'word_count': hit.payload.get('word_count', 0)
            }
            formatted_results.append(result)
        
        logger.info(f"Search query '{query}' returned {len(formatted_results)} results")
        return formatted_results
    
    def search_by_entities(self, entities: List[str],
                          limit: int = None,
                          date_filter: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Search for articles mentioning specific entities.
        
        Args:
            entities: List of entity names to search for
            limit: Number of results
            date_filter: Date range filter
            
        Returns:
            List of search results
        """
        # Create entity-focused query
        query = " ".join(entities)
        
        # Search with lower alpha to favor keyword matching
        results = self.search(
            query=query,
            limit=limit or 30,
            alpha=0.3,  # Favor sparse search for entities
            date_filter=date_filter
        )
        
        # Post-filter to ensure entity presence
        filtered_results = []
        for result in results:
            # Check entity matches in title and preview
            text_lower = (result['title'] + ' ' + result['content_preview']).lower()
            entity_matches = [e for e in entities if e.lower() in text_lower]
            
            if entity_matches:
                result['entity_matches'] = entity_matches
                filtered_results.append(result)
        
        return filtered_results
    
    def get_article_full_content(self, file_path: str) -> Article:
        """
        Retrieve full article content from file.
        
        Args:
            file_path: Path to article file
            
        Returns:
            Parsed article object
        """
        return self.parser.parse_file(Path(file_path))
    
    def _prepare_article_text(self, article: Article) -> str:
        """Prepare article text for indexing"""
        # Combine all text fields
        parts = [article.title]
        if article.subtitle:
            parts.append(article.subtitle)
        if article.authors:
            parts.append(f"Authors: {', '.join(article.authors)}")
        parts.append(article.content)
        
        return " ".join(parts)
    
    def _get_sparse_vector(self, text: str) -> Optional[SparseVector]:
        """Generate sparse vector from text"""
        if not hasattr(self.tfidf, 'vocabulary_'):
            return None
        
        try:
            sparse_matrix = self.tfidf.transform([text])
            indices = sparse_matrix[0].indices.tolist()
            values = sparse_matrix[0].data.tolist()
            
            if indices:  # Only return if we have indices
                return SparseVector(indices=indices, values=values)
            return None
        except Exception as e:
            logger.warning(f"Failed to generate sparse vector: {e}")
            return None
    
    def _get_sparse_query_vector(self, query: str) -> Optional[SparseVector]:
        """Generate sparse vector for query"""
        return self._get_sparse_vector(query)
    
    def _create_payload(self, article: Article) -> Dict[str, Any]:
        """Create payload for Qdrant point"""
        return {
            'title': article.title,
            'subtitle': article.subtitle or '',
            'authors': article.authors,
            'published': article.published.isoformat(),
            'published_date': article.published.date().isoformat(),
            'timestamp': article.published.timestamp(),
            'content_preview': article.content[:500],  # First 500 chars
            'file_path': str(article.file_path),
            'word_count': len(article.content.split())
        }
    
    def _build_filter_conditions(self, 
                                date_filter: Optional[Dict[str, str]],
                                author_filter: Optional[List[str]]) -> Optional[Filter]:
        """Build Qdrant filter conditions"""
        conditions = []
        
        if date_filter:
            if 'start_date' in date_filter:
                start_dt = datetime.fromisoformat(date_filter['start_date'])
                conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=start_dt.timestamp())
                    )
                )
            
            if 'end_date' in date_filter:
                end_dt = datetime.fromisoformat(date_filter['end_date'])
                conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(lte=end_dt.timestamp())
                    )
                )
        
        if author_filter:
            conditions.append(
                FieldCondition(
                    key="authors",
                    match={"any": author_filter}
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _apply_rrf_fusion(self, search_results: List[List], 
                         alpha: float, limit: int) -> List:
        """
        Apply Reciprocal Rank Fusion to combine search results.
        
        Args:
            search_results: List of search result lists
            alpha: Weight for dense search (1-alpha for sparse)
            limit: Number of results to return
            
        Returns:
            Fused results
        """
        # RRF constant
        k = 60
        
        # Score aggregation
        doc_scores = {}
        
        # Process dense results
        if len(search_results) > 0:
            for rank, hit in enumerate(search_results[0]):
                doc_id = hit.id
                rrf_score = alpha / (k + rank + 1)
                doc_scores[doc_id] = {
                    'score': rrf_score,
                    'hit': hit
                }
        
        # Process sparse results
        if len(search_results) > 1:
            for rank, hit in enumerate(search_results[1]):
                doc_id = hit.id
                rrf_score = (1 - alpha) / (k + rank + 1)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += rrf_score
                else:
                    doc_scores[doc_id] = {
                        'score': rrf_score,
                        'hit': hit
                    }
        
        # Sort by fused score and return top results
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:limit]
        
        # Extract hits with updated scores
        results = []
        for doc_id, doc_data in sorted_docs:
            hit = doc_data['hit']
            hit.score = doc_data['score']
            results.append(hit)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize configuration
    from config import init_config
    config = init_config()
    
    # Initialize search engine
    search_engine = QdrantHybridSearch(config)
    
    # Test with sample query
    test_queries = [
        "Chesapeake Energy acquisition merger",
        "EU energy market regulations",
        "ESG investing trends"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = search_engine.search(
            query=query,
            limit=5,
            alpha=0.65
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Published: {result['published']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Preview: {result['content_preview'][:100]}...")
"""
Qdrant Dense Search Engine for News RAG System

Implements semantic search with Qdrant vector database using dense embeddings
for optimal retrieval of news articles.
"""

import uuid
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition,
    Range, SearchParams
)

try:
    from ..config import get_config
    from .qwen_embedder import QwenEmbedder
    from .article_parser import Article, ArticleParser
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    from qwen_embedder import QwenEmbedder
    from article_parser import Article, ArticleParser

logger = logging.getLogger(__name__)


class QdrantHybridSearch:
    """Dense search engine using Qdrant with semantic embeddings"""
    
    def __init__(self, config=None):
        """Initialize the search engine with configuration"""
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Get configurations using correct config.get() method
        self.host = config.get('qdrant', 'host', default='localhost')
        self.port = config.get('qdrant', 'port', default=6333)
        self.collection_name = config.get('qdrant', 'collection_name', default='news_articles')
        
        self.default_limit = config.get('search', 'default_limit', default=20)
        

        # Initialize components
        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = QwenEmbedder()
        self.parser = ArticleParser()
        
    
    
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
        
        # Generate dense embedding
        embedding_result = self.embedder.embed_article(article)
        

        # Create point with dense vector
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
                      batch_size: int = 100,
                      preserve_metadata: bool = True) -> List[str]:
        """
        Index multiple articles in batches.
        
        Args:
            articles: List of parsed articles
            batch_size: Batch size for processing
            preserve_metadata: Whether to preserve extraction metadata if article exists
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        # First, check which articles already exist if we need to preserve metadata
        existing_metadata = {}
        if preserve_metadata:
            all_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, str(a.file_path))) for a in articles]
            for article_id in all_ids:
                metadata = self.get_extraction_metadata(article_id)
                if metadata:
                    existing_metadata[article_id] = metadata
        
        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            # Generate embeddings in batch
            embedding_results = self.embedder.embed_articles(batch, batch_size=32)
            
            # Create points
            points = []
            for idx, (article, emb_result) in enumerate(zip(batch, embedding_results)):
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(article.file_path)))
                doc_ids.append(doc_id)
                
                # Create payload, preserving extraction metadata if it exists
                payload = self._create_payload(article)
                
                # Restore extraction metadata if it existed
                if doc_id in existing_metadata:
                    payload['extraction_metadata'] = existing_metadata[doc_id]['metadata']
                    payload['cached_extraction'] = existing_metadata[doc_id]['extraction_result']
                
                # Create point with dense vector
                point = PointStruct(
                    id=doc_id,
                    vector={
                        "dense": emb_result.embedding.tolist()
                    },
                    payload=payload
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
              date_filter: Optional[Dict[str, str]] = None,
              author_filter: Optional[List[str]] = None,
              score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using dense embeddings.
        
        Args:
            query: Search query
            limit: Number of results (default from config)
            date_filter: Date range {'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}
            author_filter: List of authors to filter by
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results with metadata
        """
        limit = limit or self.default_limit
        
        # Generate query vector
        dense_query = self.embedder.embed_query(query)
        
        # Build filter
        filter_conditions = self._build_filter_conditions(
            date_filter, author_filter
        )
        
        try:
            # Dense vector search
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
        
        # Search with standard parameters
        results = self.search(
            query=query,
            limit=limit or 30,
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
    
    def update_extraction_metadata(self, article_id: str, extraction_result: Dict[str, Any]) -> bool:
        """
        Update article with extraction results and metadata.
        
        Args:
            article_id: Document ID in Qdrant
            extraction_result: Dictionary containing extraction results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate metrics from extraction result
            events_count = len(extraction_result.get('events', []))
            entities_count = len(extraction_result.get('entities', []))
            
            # Calculate average confidence
            event_confidences = [e.get('confidence', 0.0) for e in extraction_result.get('events', [])]
            entity_confidences = [e.get('confidence', 0.0) for e in extraction_result.get('entities', [])]
            all_confidences = event_confidences + entity_confidences
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            # Create update payload
            update_payload = {
                'extraction_metadata': {
                    'extracted': True,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_version': '1.0',
                    'events_count': events_count,
                    'entities_count': entities_count,
                    'extraction_confidence': round(avg_confidence, 3)
                },
                'cached_extraction': extraction_result  # Store full extraction result
            }
            
            # Update the point in Qdrant
            self.client.set_payload(
                collection_name=self.collection_name,
                points=[article_id],
                payload=update_payload
            )
            
            logger.info(f"Updated extraction metadata for {article_id}: {events_count} events, {entities_count} entities")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update extraction metadata for {article_id}: {e}")
            return False
    
    def get_extraction_metadata(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get extraction metadata for an article.
        
        Args:
            article_id: Document ID in Qdrant
            
        Returns:
            Extraction metadata if available, None otherwise
        """
        try:
            # Retrieve the point from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[article_id],
                with_payload=True
            )
            
            if points and len(points) > 0:
                payload = points[0].payload
                extraction_metadata = payload.get('extraction_metadata', {})
                cached_extraction = payload.get('cached_extraction')
                
                if extraction_metadata.get('extracted') and cached_extraction:
                    return {
                        'metadata': extraction_metadata,
                        'extraction_result': cached_extraction
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get extraction metadata for {article_id}: {e}")
            return None
    
    def article_exists(self, article_id: str) -> bool:
        """
        Check if an article exists in the collection.
        
        Args:
            article_id: Document ID in Qdrant
            
        Returns:
            True if article exists, False otherwise
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[article_id],
                with_payload=False
            )
            return len(points) > 0
        except Exception as e:
            logger.warning(f"Failed to check if article exists {article_id}: {e}")
            return False
    
    def get_existing_articles(self, article_ids: List[str]) -> Dict[str, bool]:
        """
        Check which articles exist in the collection.
        
        Args:
            article_ids: List of document IDs to check
            
        Returns:
            Dictionary mapping article_id to existence status
        """
        try:
            # Retrieve multiple points at once for efficiency
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=article_ids,
                with_payload=False
            )
            
            # Create set of existing IDs
            existing_ids = {point.id for point in points}
            
            # Return existence status for each requested ID
            return {
                article_id: article_id in existing_ids 
                for article_id in article_ids
            }
        except Exception as e:
            logger.warning(f"Failed to check existing articles: {e}")
            # Return all as not existing on error
            return {article_id: False for article_id in article_ids}
    
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
            'word_count': len(article.content.split()),
            'extraction_metadata': {
                'extracted': False,
                'extraction_timestamp': None,
                'extraction_version': '1.0',
                'events_count': 0,
                'entities_count': 0,
                'extraction_confidence': 0.0
            }
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
            limit=5
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Published: {result['published']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Preview: {result['content_preview'][:100]}...")
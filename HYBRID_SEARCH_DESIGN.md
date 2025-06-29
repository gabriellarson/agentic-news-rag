# Hybrid Search Design for News RAG System

## Overview
A hybrid search approach combining dense vector search with sparse keyword search using Qdrant vector database to optimize retrieval quality for news articles at scale (40k documents).

## Article Format
All articles follow this structured format:
```
Title: [Article Title]
Subtitle: [Article Subtitle]
Authors: [Comma-separated authors or empty]
Published: [ISO 8601 timestamp]

[Article body text...]
```

## Architecture

### 1. Dense Vector Search (Semantic)
- **Embedding Model**: Qwen3-Embedding-0.6B via sentence-transformers
- **Vector Store**: Qdrant with HNSW index
- **Dimension**: Model-specific (typically 1536 for Qwen3)
- **Distance Metric**: Cosine similarity

### 2. Sparse Search (Keyword/BM25)
- **Implementation**: Qdrant's built-in sparse vectors support
- **Indexing**: TF-IDF based sparse embeddings
- **Preprocessing**: 
  - Tokenization with spaCy
  - Stopword removal
  - Optional stemming/lemmatization

### 3. Hybrid Fusion Strategy
- **Method**: Reciprocal Rank Fusion (RRF) with configurable weights
- **Qdrant Native**: Leverages Qdrant's built-in hybrid search capabilities

## Implementation

### Article Parser
```python
import re
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class Article:
    title: str
    subtitle: str
    authors: list[str]
    published: datetime
    content: str
    file_path: str

class ArticleParser:
    @staticmethod
    def parse_article(file_path: str) -> Article:
        """Parse article from standard format text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Extract metadata
        title = ""
        subtitle = ""
        authors = []
        published = None
        body_start_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('Title: '):
                title = line[7:].strip()
            elif line.startswith('Subtitle: '):
                subtitle = line[10:].strip()
            elif line.startswith('Authors: '):
                authors_str = line[9:].strip()
                if authors_str:
                    authors = [a.strip() for a in authors_str.split(',')]
            elif line.startswith('Published: '):
                published_str = line[11:].strip()
                published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            elif line.strip() == '' and i > 3:  # Empty line after metadata
                body_start_idx = i + 1
                break
        
        # Extract body content
        body_content = '\n'.join(lines[body_start_idx:]).strip()
        
        return Article(
            title=title,
            subtitle=subtitle,
            authors=authors,
            published=published,
            content=body_content,
            file_path=file_path
        )
```

### Embedding Pipeline
```python
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict
import numpy as np

class QwenEmbeddingPipeline:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # For cosine similarity
            show_progress_bar=True,
            device=self.device
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.encode([text])[0]
    
    def prepare_article_text(self, article: Article) -> str:
        """Prepare article text for embedding with weighted sections"""
        # Give more weight to title and subtitle
        weighted_text = f"{article.title}. {article.subtitle}. {article.title}. {article.content}"
        return weighted_text
```

### Qdrant Integration
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SparseVector, NamedSparseVector,
    SearchParams, Filter, FieldCondition, Range
)
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from glob import glob

class QdrantHybridSearch:
    def __init__(self, 
                 collection_name: str = "news_articles",
                 qdrant_url: str = "localhost:6333",
                 embedding_dim: int = 1536):
        
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.embedder = QwenEmbeddingPipeline()
        self.parser = ArticleParser()
        
        # Initialize sparse encoder (TF-IDF)
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Create collection with hybrid search support
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection with dense and sparse vectors"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=True  # For large collections
                )
            },
            sparse_vectors_config={
                "sparse": {
                    "index": {
                        "on_disk": True
                    }
                }
            }
        )
    
    def index_articles_from_directory(self, articles_dir: str, batch_size: int = 100):
        """Index all articles from a directory"""
        article_files = glob(os.path.join(articles_dir, "*.txt"))
        print(f"Found {len(article_files)} articles to index")
        
        # Process in batches
        for i in range(0, len(article_files), batch_size):
            batch_files = article_files[i:i+batch_size]
            articles = []
            
            # Parse articles
            for file_path in batch_files:
                try:
                    article = self.parser.parse_article(file_path)
                    articles.append(article)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    continue
            
            # Index batch
            if articles:
                self._index_article_batch(articles)
                print(f"Indexed batch {i//batch_size + 1}/{(len(article_files) + batch_size - 1)//batch_size}")
    
    def _index_article_batch(self, articles: List[Article]):
        """Index a batch of parsed articles"""
        # Prepare texts for embedding
        texts = [self.embedder.prepare_article_text(article) for article in articles]
        
        # Generate dense embeddings
        dense_embeddings = self.embedder.encode(texts)
        
        # Update TF-IDF vocabulary and generate sparse embeddings
        if not hasattr(self.tfidf, 'vocabulary_'):
            # First batch - fit the TF-IDF
            self.tfidf.fit(texts)
        
        sparse_matrix = self.tfidf.transform(texts)
        
        # Prepare points for Qdrant
        points = []
        for idx, article in enumerate(articles):
            # Convert sparse matrix row to Qdrant format
            sparse_indices = sparse_matrix[idx].indices.tolist()
            sparse_values = sparse_matrix[idx].data.tolist()
            
            # Create unique ID based on file path
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, article.file_path))
            
            point = PointStruct(
                id=doc_id,
                vector={
                    "dense": dense_embeddings[idx].tolist()
                },
                sparse_vectors={
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={
                    "title": article.title,
                    "subtitle": article.subtitle,
                    "authors": article.authors,
                    "published": article.published.isoformat(),
                    "timestamp": article.published.timestamp(),
                    "content": article.content[:1000],  # Store first 1000 chars
                    "file_path": article.file_path,
                    "word_count": len(article.content.split())
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )
    
    def hybrid_search(self, 
                     query: str, 
                     limit: int = 20,
                     alpha: float = 0.65,
                     date_filter: Dict = None,
                     author_filter: List[str] = None) -> List[Dict]:
        """
        Perform hybrid search with RRF fusion
        
        Args:
            query: Search query
            limit: Number of results to return
            alpha: Weight for dense search (0-1), 1-alpha for sparse
            date_filter: Optional date range filter {'start_date': 'YYYY-MM-DD', 'end_date': 'YYYY-MM-DD'}
            author_filter: Optional list of authors to filter by
        """
        # Generate query vectors
        dense_query = self.embedder.encode_single(query)
        sparse_query = self._get_sparse_query_vector(query)
        
        # Build filter conditions
        filter_conditions = []
        
        if date_filter:
            if 'start_date' in date_filter:
                start_timestamp = datetime.fromisoformat(date_filter['start_date']).timestamp()
                filter_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(gte=start_timestamp)
                    )
                )
            if 'end_date' in date_filter:
                end_timestamp = datetime.fromisoformat(date_filter['end_date']).timestamp()
                filter_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(lte=end_timestamp)
                    )
                )
        
        if author_filter:
            # Filter for articles by specific authors
            filter_conditions.append(
                FieldCondition(
                    key="authors",
                    match={"any": author_filter}
                )
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Perform hybrid search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name="sparse",
                vector=sparse_query
            ),
            query_dense=dense_query.tolist(),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            params=SearchParams(
                fusion="rrf",  # Reciprocal Rank Fusion
                fusion_params={
                    "alpha": alpha  # Weight for dense vs sparse
                }
            )
        )
        
        # Format results
        formatted_results = []
        for hit in results:
            result = {
                'id': hit.id,
                'score': hit.score,
                'title': hit.payload['title'],
                'subtitle': hit.payload['subtitle'],
                'authors': hit.payload['authors'],
                'published': hit.payload['published'],
                'content_preview': hit.payload['content'],
                'file_path': hit.payload['file_path'],
                'word_count': hit.payload['word_count']
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _get_sparse_query_vector(self, query: str) -> SparseVector:
        """Convert query to sparse vector using fitted TF-IDF"""
        sparse_matrix = self.tfidf.transform([query])
        indices = sparse_matrix[0].indices.tolist()
        values = sparse_matrix[0].data.tolist()
        
        return SparseVector(indices=indices, values=values)
    
    def get_article_full_content(self, file_path: str) -> str:
        """Retrieve full article content from file"""
        article = self.parser.parse_article(file_path)
        return article.content
```

### Advanced Search Features

```python
class EnhancedSearch:
    def __init__(self, search_engine: QdrantHybridSearch, llm_client):
        self.search = search_engine
        self.llm = llm_client
        
    def search_with_temporal_context(self, query: str, context_window_days: int = 7) -> List[Dict]:
        """Search and include temporal context around found articles"""
        # Initial search
        initial_results = self.search.hybrid_search(query, limit=10)
        
        if not initial_results:
            return []
        
        # Find date range from initial results
        dates = [datetime.fromisoformat(r['published']) for r in initial_results]
        min_date = min(dates) - timedelta(days=context_window_days)
        max_date = max(dates) + timedelta(days=context_window_days)
        
        # Expanded search with date filter
        expanded_results = self.search.hybrid_search(
            query,
            limit=30,
            date_filter={
                'start_date': min_date.strftime('%Y-%m-%d'),
                'end_date': max_date.strftime('%Y-%m-%d')
            }
        )
        
        return expanded_results
    
    def search_by_entities(self, entities: List[str], date_filter: Dict = None) -> List[Dict]:
        """Search for articles mentioning specific entities"""
        # Create entity-focused query
        query = " ".join(entities)
        
        # Search with lower alpha to favor keyword matching
        results = self.search.hybrid_search(
            query,
            limit=30,
            alpha=0.3,  # Favor sparse search for entity names
            date_filter=date_filter
        )
        
        # Post-filter to ensure entity presence
        filtered_results = []
        for result in results:
            # Load full content if needed
            full_content = self.search.get_article_full_content(result['file_path'])
            
            # Check if all entities are mentioned
            content_lower = full_content.lower()
            if all(entity.lower() in content_lower for entity in entities):
                result['entity_matches'] = entities
                filtered_results.append(result)
        
        return filtered_results
```

### Configuration

```yaml
# config/search_config.yaml
embedding:
  model_name: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 32
  normalize: true
  device: "cuda"  # or "cpu"

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "news_articles"
  vector_size: 1536
  on_disk: true  # For large collections
  
search:
  hybrid:
    # Weight for dense vs sparse (0.0 = pure sparse, 1.0 = pure dense)
    alpha: 0.65
    
    # Number of results to retrieve
    default_limit: 20
    max_limit: 100
    
  sparse:
    # TF-IDF parameters
    max_features: 10000
    ngram_range: [1, 2]
    min_df: 2
    max_df: 0.95
    
  indexing:
    batch_size: 100
    articles_directory: "text_articles/"
    
performance:
  # Caching
  enable_query_cache: true
  cache_size: 1000
  cache_ttl_seconds: 3600
```

### Usage Example

```python
# index_articles.py
from qdrant_hybrid_search import QdrantHybridSearch
import yaml

def main():
    # Load config
    with open('config/search_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize search engine
    search_engine = QdrantHybridSearch(
        collection_name=config['qdrant']['collection_name'],
        qdrant_url=f"{config['qdrant']['host']}:{config['qdrant']['port']}"
    )
    
    # Index all articles
    search_engine.index_articles_from_directory(
        config['search']['indexing']['articles_directory'],
        batch_size=config['search']['indexing']['batch_size']
    )
    
    # Example searches
    results = search_engine.hybrid_search(
        "gas prices energy market",
        limit=10,
        date_filter={'start_date': '2024-01-01', 'end_date': '2024-01-31'}
    )
    
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Published: {result['published']}")
        print(f"Score: {result['score']:.3f}")
        print("---")

if __name__ == "__main__":
    main()
```
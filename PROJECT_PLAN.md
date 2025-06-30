# Agentic News RAG Analysis System - Project Plan

## Overview
An intelligent system for analyzing a corpus of ~40,000 news articles to answer user queries by constructing temporal timelines from semantically relevant articles.

## Architecture Components

### 1. Data Ingestion & Preprocessing
- **Article Storage**: Raw .txt files with timestamps
- **Metadata Management**: Article publication dates, sources, categories
- **Text Preprocessing**: Clean and normalize articles for embedding
- **Batch Processing**: Handle 40k articles efficiently

### 2. Embedding Pipeline
- **Model**: qwen3-embedding via llama.cpp server
- **Vector Database**: Store embeddings with metadata
  - Options: ChromaDB, Qdrant, or FAISS
- **Indexing Strategy**: Optimize for semantic similarity search
- **Chunking Strategy**: Handle long articles appropriately

### 3. Query Analysis Agent
- **Query Understanding**: Parse user intent and identify key concepts
- **Search Strategy Generation**: Create multiple semantic search queries
- **Query Expansion**: Generate related terms and concepts
- **Temporal Constraints**: Extract date ranges if specified

### 4. Semantic Search Engine
- **Multi-Query Search**: Execute parallel searches
- **Relevance Scoring**: Rank articles by semantic similarity
- **Result Filtering**: Apply temporal and relevance thresholds
- **Context Window Management**: Select optimal number of articles

### 5. Information Extraction Agent
- **Temporal Extraction**: 
  - Extract event dates and time references
  - Handle relative dates ("last week", "yesterday")
  - Disambiguate temporal expressions
- **Event Extraction**:
  - Identify key events and actions
  - Extract entities (people, organizations, locations)
  - Capture causal relationships
- **Fact Verification**: Cross-reference information across articles

### 6. Timeline Construction Agent
- **Temporal Ordering**: Sort events chronologically
- **Event Deduplication**: Merge similar events from multiple sources
- **Information Synthesis**: Combine related events
- **Confidence Scoring**: Rate reliability of temporal information

### 7. Report Generation Agent
- **Answer Synthesis**: Generate coherent response to query
- **Citation Management**: Reference source articles
- **Summary Generation**: Create concise reports
- **Uncertainty Handling**: Indicate conflicting information

## Technical Stack

### Core Dependencies
- **Language**: Python 3.10+
- **LLM Integration**: OpenAI-compatible API (for any LLM backend)
- **Vector Database**: Qdrant with hybrid search support
- **Embeddings**: Qwen/Qwen3-Embedding-0.6B via sentence-transformers
- **Framework**: FastAPI for API, asyncio for concurrent processing
- **Data Processing**: pandas, numpy
- **NLP Tools**: spaCy for entity recognition and temporal parsing
- **ML Libraries**: scikit-learn (for TF-IDF), torch (for embeddings)

### Project Structure
```
agentic-news-rag/
├── text_articles/      # Raw .txt article files
├── src/
│   ├── agents/         # Agent implementations
│   │   ├── query_analysis.py
│   │   ├── information_extraction.py
│   │   ├── timeline_construction.py
│   │   └── report_generation.py
│   ├── embeddings/     # Embedding pipeline
│   │   ├── qwen_embedder.py
│   │   └── article_parser.py
│   ├── search/         # Hybrid search implementation
│   │   ├── qdrant_search.py
│   │   ├── query_expansion.py
│   │   └── temporal_reranker.py
│   ├── models/         # Data models
│   │   ├── article.py
│   │   └── event.py
│   └── api/            # FastAPI endpoints
│       └── main.py
├── config/             # Configuration files
│   └── search_config.yaml
├── tests/              # Unit and integration tests
├── scripts/            # Utility scripts
│   ├── index_articles.py
│   └── setup_qdrant.py
└── requirements.txt    # Python dependencies
```

## Implementation Phases

### Phase 1: Foundation (Week 1) ✅ COMPLETED
1. ✅ Set up project structure - Complete folder structure implemented
2. ✅ LLM integration - OpenAI-compatible client for llama.cpp server at localhost:8001
3. ✅ Create embedding pipeline - Qwen3-Embedding-0.6B (1024-dim) fully implemented
4. ✅ Set up vector database - Qdrant server configured with dense vector support

### Phase 2: Core Pipeline (Week 2) ✅ COMPLETED
1. ✅ Implement query analysis agent - Complete with classification, entity extraction, temporal parsing
2. ✅ Build semantic search functionality - Qdrant dense search implemented and tested
3. ⏳ Create basic information extraction - Next component to implement
4. ✅ Test with sample articles - Successfully indexed and searched 7 articles

### Phase 3: Advanced Features (Week 3)
1. Implement temporal extraction
2. Build timeline construction logic
3. Create event deduplication
4. Develop confidence scoring

### Phase 4: Report Generation (Week 4)
1. Implement report synthesis
2. Add citation management
3. Create output formatting
4. Build API/CLI interface

### Phase 5: Optimization & Testing (Week 5)
1. Performance optimization
2. Comprehensive testing
3. Error handling
4. Documentation

## Current Implementation Status

### ✅ Completed Components

#### Core Infrastructure
- **Configuration System**: Full YAML-based config with environment overrides
- **Project Structure**: Complete folder hierarchy with Python packages
- **LLM Integration**: Working with Qwen3-30B model via llama.cpp server
  - Model path: `D:\AI\GGUFs\Qwen3-30B-A3B-UD-Q4_K_XL.gguf`
  - Thinking mode enabled (outputs to reasoning_content, final answer to content)
  - Requires `n=1` parameter and high max_tokens (10000) for proper operation

#### Embedding & Search Pipeline
- **Article Parser**: FULLY FUNCTIONAL
  - Parses structured text articles with metadata (title, subtitle, authors, published date, content)
  - Validates article format and content
  - Handles timezone-aware datetime comparisons
- **Embedding Pipeline**: FULLY FUNCTIONAL
  - Local Qwen3-Embedding-0.6B model (1024-dimensional vectors)
  - Batch processing with configurable parameters
  - Weighted text preparation (title and subtitle emphasized)
  - GPU acceleration support
- **Qdrant Vector Database**: FULLY FUNCTIONAL
  - Dense vector search with cosine similarity
  - Efficient indexing with payload filters
  - Collection configured for 40k+ articles
- **Search Engine**: FULLY FUNCTIONAL
  - Semantic search using dense vectors
  - Date and author filtering
  - Configurable result limits and scoring thresholds
  - Successfully tested with 7 sample articles

#### Query Processing
- **Query Analysis Agent**: FULLY FUNCTIONAL
  - Query classification (FACTUAL, CONCEPTUAL, TEMPORAL, ENTITY, COMPARATIVE) - 100% accuracy
  - Named entity extraction - Working correctly
  - Temporal constraint parsing (relative and explicit dates) - Working
  - Query expansion (generates 5 alternative search queries) - Working
  - Search alpha determination (dense vs sparse weights) - Properly calibrated

#### Indexing & Testing
- **Indexing Scripts**: FULLY FUNCTIONAL
  - Batch article processing with progress tracking
  - Error handling and validation
  - TF-IDF model persistence
  - Successfully indexed 7 articles in 1.05 seconds (0.15s per article)
- **Search Testing**: FULLY FUNCTIONAL
  - Test queries return relevant results with good semantic matching:
    - "Chesapeake Energy merger acquisition" → Chesapeake deal article (0.659 score)
    - "EU energy regulations" → EU car emissions article (0.524 score)
    - "ESG investing" → Greenwashing article (0.555 score)
    - "gas prices Europe" → ICE gas market article (0.537 score)

### ⏳ Next Components
- **Information Extraction Agent**: Extract events, entities, and temporal information from articles
- **Timeline Construction Agent**: Chronological event ordering and deduplication
- **Report Generation Agent**: Response synthesis with citations
- **Hybrid Search Enhancement**: Add sparse vector support for keyword matching

## Key Considerations

### Scalability
- Batch processing for 40k articles
- Efficient vector search
- Caching strategies
- Parallel processing

### Accuracy
- Temporal disambiguation
- Source credibility
- Conflicting information handling
- Fact verification

### User Experience
- Query flexibility
- Response time optimization
- Clear citations
- Confidence indicators

## Current Configuration (config/search_config.yaml)
```yaml
# LLM Configuration - Currently using Qwen3-30B via llama.cpp
llm:
  endpoint: "http://localhost:8001/v1"  # Your llama.cpp server
  model: "qwen3-30b"  # Qwen3-30B with thinking enabled, 128k context
  temperature: 0.3
  max_tokens: 2000
  timeout: 60

# Embedding Configuration - Planned
embeddings:
  model_name: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 32
  device: "cuda"  # or "cpu"
  normalize: true

# Vector Database - Qdrant configuration
qdrant:
  host: "localhost"
  port: 6333
  collection_name: "news_articles"
  on_disk: true

# Search Configuration
search:
  hybrid:
    alpha: 0.65  # Dense vs sparse weight
    default_limit: 20
    max_limit: 100
  sparse:
    max_features: 10000
    ngram_range: [1, 2]
    min_df: 2
    max_df: 0.95
  temporal:
    enable_reranking: true
    decay_half_life_days: 30

# Query Analysis - Implemented
query_analysis:
  max_expanded_queries: 5
  entity_extraction:
    enabled: true
    confidence_threshold: 0.8
  temporal_extraction:
    enabled: true
    default_window_days: 30
  query_classification:
    enabled: true
    type_weights:
      factual: 0.4      # Favor keyword search
      entity: 0.3       # Favor keyword search for entities
      conceptual: 0.8    # Favor semantic search
      temporal: 0.6      # Balanced approach
      comparative: 0.7   # Favor semantic search
      default: 0.65

extraction:
  max_articles_per_query: 10
  temporal_window_days: 30
  confidence_threshold: 0.7

reporting:
  max_report_length: 2000
  include_citations: true
  include_confidence_scores: true
```

## Success Metrics
- Query response accuracy
- Temporal ordering correctness
- Information completeness
- Processing speed
- User satisfaction

## Current Issues & Next Steps

### ✅ Resolved Issues
1. **LLM Empty Responses**: Fixed by using correct model path and `n=1` parameter
2. **Token Limit**: Increased max_tokens to 10000 to allow full thinking + answer
3. **Query Classification**: Now working at 100% accuracy
4. **Entity Extraction**: Fully functional with proper JSON parsing
5. **Query Expansion**: Consistently generating 5 variations

### Immediate Next Steps
1. **Implement Article Parser**: Parse structured text articles (Title, Subtitle, Authors, Published, Content)
2. **Build Embedding Pipeline**: Integrate Qwen3-Embedding-0.6B with sentence-transformers
3. **Set up Qdrant Server**: Initialize vector database for hybrid search
4. **Implement Qdrant Search Engine**: Hybrid dense/sparse search with RRF fusion
5. **Information Extraction Agent**: Extract events and entities from articles

### Data Notes
- **Article Format**: All articles in `text_articles/` follow structured format:
  ```
  Title: [Article Title]
  Subtitle: [Article Subtitle]
  Authors: [Comma-separated authors or empty]
  Published: [ISO 8601 timestamp]
  
  [Article body text...]
  ```
- **Sample Articles Available**: 7 sample articles ready for testing
- **Target Scale**: System designed for 40,000 articles

### Technology Stack Decisions Made
- **LLM**: Qwen3-30B via llama.cpp server (localhost:8001) ✅
- **Embeddings**: Qwen3-Embedding-0.6B via sentence-transformers ✅
- **Vector DB**: Qdrant with hybrid search support ✅
- **Framework**: Python 3.10+ with FastAPI ✅
- **Config**: YAML-based with environment overrides ✅
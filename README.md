# Agentic News RAG Analysis System

An intelligent system for analyzing a corpus of ~40,000 news articles to answer user queries by constructing temporal timelines from semantically relevant articles.

## 🏗️ Architecture

### Core Agents

1. **Query Analysis Agent** ✅ - Analyzes user queries and generates search strategies
2. **Information Extraction Agent** ✅ - Extracts events, entities, and temporal information
3. **Timeline Construction Agent** ✅ - Orders events chronologically and identifies relationships
4. **Report Generation Agent** ⏳ - Synthesizes final responses with citations

### Technology Stack

- **Language**: Python 3.10+
- **LLM**: Qwen3-30B via llama.cpp server (localhost:8001)
- **Embeddings**: Qwen3-Embedding-0.6B via sentence-transformers
- **Vector Database**: Qdrant
- **Framework**: FastAPI for API, asyncio for concurrency
- **Config**: YAML with environment variable overrides

## 🔧 Quick Start

### Prerequisites

- Python 3.10+
- llama.cpp server running Qwen3-30B at localhost:8001
  - Model: `D:\AI\GGUFs\Qwen3-30B-A3B-UD-Q4_K_XL.gguf`
  - Thinking mode enabled (uses reasoning_content field)
- Qdrant server running at localhost:6333
  - `docker run -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage qdrant/qdrant`
- CUDA-capable GPU (recommended for embeddings)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start Commands

```bash
# 1. Set up Qdrant database
python scripts/setup_qdrant.py

# 2. Index articles
python scripts/index_articles.py --test

# 3. Test query analysis
python scripts/test_full_workflow.py
```

### Configuration

Edit `config/search_config.yaml` or use environment variables:

```bash
export NEWS_RAG_LLM_ENDPOINT="http://localhost:8001/v1"
export NEWS_RAG_LLM_MODEL="qwen3-30b"
```

## 📊 Information Extraction & Timeline Features

### Information Extraction Pipeline ✨

- **Event Extraction**: LLM extracts key events with confidence scoring
- **Entity Recognition**: People, organizations, locations, and other entities
- **Temporal Processing**: Dynamic date resolution for complex temporal references
  - Absolute dates: "January 11, 2024", "March 2023", "2020"
  - Relative dates: "last year", "second quarter", "end of 2022"
  - Contextual dates: "when the pandemic began", "since the war started"
- **Robust JSON Parsing**: Error recovery from malformed LLM responses
- **Performance**: 83% timeline completeness achieved (up from 0% before fixes)

### Timeline Construction Pipeline ✨

- **Event Deduplication**: LLM-based grouping of similar events across articles
- **Chronological Ordering**: Smart sorting with date estimation for undated events
- **Importance Scoring**: Relevance filtering based on timeline topic
- **Causal Relationships**: Identification of cause-and-effect connections
- **Consistency Validation**: Timeline coherence and completeness scoring

### Recent Fixes & Improvements

- **Date Resolution Fixed**: Increased max_tokens from 50 to 5000 for LLM responses
- **Full Text Processing**: Removed 4000-character truncation limits
- **Debug Tooling**: Created comprehensive extraction debugging script
- **Performance Metrics**: Timeline completeness improved from 0% to 83%

## 📊 Query Analysis Features

### Query Classification

Automatically categorizes queries into:

- **FACTUAL**: Specific facts or events
- **CONCEPTUAL**: Broader concepts and themes
- **TEMPORAL**: Time-based events and chronology
- **ENTITY**: People, companies, organizations
- **COMPARATIVE**: Comparisons between events/entities

### Entity Extraction

Identifies and extracts:

- People names
- Company names
- Organizations
- Locations

### Temporal Parsing

Handles both relative and explicit temporal references:

- Relative: "yesterday", "last week", "this month"
- Explicit: "January 2024", "between X and Y"

### Query Expansion

Generates 3-4 alternative search queries using LLM to improve recall.

### Search Optimization

Determines optimal dense vs sparse search weights based on query type.

## 📄 Article Format

All articles follow this structured format:

```
Title: [Article Title]
Subtitle: [Article Subtitle]
Authors: [Comma-separated authors or empty]
Published: [ISO 8601 timestamp]

[Article body text...]
```

## 🚧 Development Notes

### Key Implementation Details

1. **LLM Configuration**:

   - Must use full model path from llama.cpp server
   - Requires `n=1` parameter to get content output
   - Needs high max_tokens (10000) for thinking + answer
   - Thinking output goes to `reasoning_content` (ignored)
   - Final answer goes to `content` field (what we use)

2. **Query Analysis Performance**:
   - Classification: 100% accuracy on test cases
   - Entity extraction: Correctly identifies companies, people, locations
   - Query expansion: Consistently generates 5 relevant variations
   - Processing time: ~15-20 seconds per full analysis

**Note**: This is an active development project.

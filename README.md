# Agentic News RAG

An agentic workflow for generating reports to answer user queries by constructing timelines from semantically relevant articles.

## Usage

### Prerequisites

- Python
- LM Studio serving an LLM and an embedding model
- Qdrant server running at localhost:6333

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml`

### Running

```bash
# 1. Set up Qdrant database collection
python scripts/setup_qdrant.py

# 2. Index articles (all articles must follow specified format)
python scripts/index_articles.py

# 3. Run agent
python agent.py
```

## Article Format

All articles follow this structured format, and be stored in a .txt file:

```
Title: [Article Title]
Subtitle: [Article Subtitle]
Authors: [Comma-separated authors or empty]
Published: [ISO 8601 timestamp]

[Article body text...]
```

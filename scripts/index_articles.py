"""
Index articles into Qdrant vector database.

This script:
1. Loads articles from the text_articles directory
2. Generates embeddings using Qwen3-Embedding-0.6B
3. Creates sparse vectors using TF-IDF
4. Indexes everything into Qdrant for hybrid search
"""

import sys
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.embeddings.article_parser import ArticleParser
from src.search.qdrant_search import QdrantHybridSearch

logger = logging.getLogger(__name__)


def index_articles(articles_dir: str, batch_size: int = 50, 
                   max_articles: int = None, skip_existing: bool = True):
    """
    Index articles from directory into Qdrant.
    
    Args:
        articles_dir: Directory containing article text files
        batch_size: Number of articles to process in each batch
        max_articles: Maximum number of articles to index (None for all)
        skip_existing: Whether to skip already indexed articles
    """
    # Initialize components
    config = get_config()
    parser = ArticleParser()
    search_engine = QdrantHybridSearch(config)
    
    # Get article files
    articles_path = Path(articles_dir)
    if not articles_path.exists():
        logger.error(f"Articles directory not found: {articles_dir}")
        return
    
    article_files = sorted(articles_path.glob("*.txt"))
    logger.info(f"Found {len(article_files)} article files")
    
    if max_articles:
        article_files = article_files[:max_articles]
        logger.info(f"Limiting to {max_articles} articles")
    
    # Track progress
    indexed_count = 0
    error_count = 0
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, len(article_files), batch_size):
        batch_end = min(batch_start + batch_size, len(article_files))
        batch_files = article_files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                   f"files {batch_start+1}-{batch_end}")
        
        # Parse articles
        articles = []
        for file_path in tqdm(batch_files, desc="Parsing articles"):
            try:
                article = parser.parse_file(file_path)
                
                # Validate article
                issues = parser.validate_article(article)
                if issues:
                    logger.warning(f"Validation issues for {file_path.name}: {issues}")
                
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                error_count += 1
                continue
        
        # Index batch
        if articles:
            try:
                logger.info(f"Indexing {len(articles)} articles...")
                doc_ids = search_engine.index_articles(articles, batch_size=32)
                indexed_count += len(doc_ids)
                logger.info(f"Successfully indexed {len(doc_ids)} articles")
                
            except Exception as e:
                logger.error(f"Error indexing batch: {e}")
                error_count += len(articles)
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\nIndexing complete:")
    logger.info(f"  Total files: {len(article_files)}")
    logger.info(f"  Successfully indexed: {indexed_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"  Average time per article: {elapsed_time/max(indexed_count, 1):.2f} seconds")


def test_search(search_engine: QdrantHybridSearch):
    """Run some test searches to verify indexing worked"""
    test_queries = [
        "Chesapeake Energy merger acquisition",
        "EU energy regulations",
        "ESG investing",
        "gas prices Europe"
    ]
    
    print("\n" + "="*60)
    print("Testing search functionality")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        try:
            results = search_engine.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n  {i}. {result['title']}")
                    print(f"     Published: {result['published'][:10]}")
                    print(f"     Score: {result['score']:.3f}")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Index articles into Qdrant")
    parser.add_argument(
        "--articles-dir",
        default="text_articles",
        help="Directory containing article text files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of articles to process in each batch"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        help="Maximum number of articles to index (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip articles that are already indexed"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test searches after indexing"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run indexing
    index_articles(
        articles_dir=args.articles_dir,
        batch_size=args.batch_size,
        max_articles=args.max_articles,
        skip_existing=args.skip_existing
    )
    
    # Run test searches if requested
    if args.test:
        config = get_config()
        search_engine = QdrantHybridSearch(config)
        test_search(search_engine)


if __name__ == "__main__":
    main()
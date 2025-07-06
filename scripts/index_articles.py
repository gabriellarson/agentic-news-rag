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
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.embeddings.article_parser import ArticleParser
from src.embeddings.qdrant_search import QdrantHybridSearch

logger = logging.getLogger(__name__)


def index_articles(articles_dir: str, batch_size: int = 50, 
                   max_articles: int = None, reindex_existing: bool = False):
    """
    Index articles from directory into Qdrant.
    
    Args:
        articles_dir: Directory containing article text files
        batch_size: Number of articles to process in each batch
        max_articles: Maximum number of articles to index (None for all)
        reindex_existing: Whether to reindex already indexed articles (default: skip existing)
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
    skipped_count = 0
    error_count = 0
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, len(article_files), batch_size):
        batch_end = min(batch_start + batch_size, len(article_files))
        batch_files = article_files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                   f"files {batch_start+1}-{batch_end}")
        
        # Parse articles and check which exist if not reindexing
        articles = []
        articles_to_index = []
        
        for file_path in tqdm(batch_files, desc="Parsing articles"):
            try:
                # Generate article ID to check existence
                article_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))
                
                # Skip if article exists and we're not reindexing
                if not reindex_existing and search_engine.article_exists(article_id):
                    skipped_count += 1
                    logger.debug(f"Skipping existing article: {file_path.name}")
                    continue
                
                # Parse the article
                article = parser.parse_file(file_path)
                
                # Validate article
                issues = parser.validate_article(article)
                if issues:
                    logger.warning(f"Validation issues for {file_path.name}: {issues}")
                
                articles_to_index.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                error_count += 1
                continue
        
        # Index batch if there are articles to index
        if articles_to_index:
            try:
                logger.info(f"Indexing {len(articles_to_index)} articles...")
                doc_ids = search_engine.index_articles(
                    articles_to_index, 
                    batch_size=32,
                    preserve_metadata=not reindex_existing  # Preserve metadata when not force reindexing
                )
                indexed_count += len(doc_ids)
                logger.info(f"Successfully indexed {len(doc_ids)} articles")
                
            except Exception as e:
                logger.error(f"Error indexing batch: {e}")
                error_count += len(articles_to_index)
        else:
            logger.info(f"No new articles to index in this batch (all {len(batch_files)} already exist)")
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\nIndexing complete:")
    logger.info(f"  Total files: {len(article_files)}")
    logger.info(f"  Successfully indexed: {indexed_count}")
    logger.info(f"  Skipped (already indexed): {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")
    if indexed_count > 0:
        logger.info(f"  Average time per indexed article: {elapsed_time/indexed_count:.2f} seconds")


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
        "--reindex-existing",
        action="store_true",
        help="Reindex articles that are already indexed (default: skip existing)"
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
        reindex_existing=args.reindex_existing
    )
    
    # Run test searches if requested
    if args.test:
        config = get_config()
        search_engine = QdrantHybridSearch(config)
        test_search(search_engine)


if __name__ == "__main__":
    main()
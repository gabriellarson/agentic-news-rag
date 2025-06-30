"""
Test script for Information Extraction Agent.

Tests extraction of events, entities, and temporal references
from sample articles using the LLM-based extraction agent.
"""

import sys
from pathlib import Path
import logging
import json
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.agents.information_extraction import InformationExtractionAgent
from src.embeddings.article_parser import ArticleParser

logger = logging.getLogger(__name__)


def test_extraction_on_articles(articles_dir: str = "text_articles", 
                               max_articles: int = 3):
    """
    Test information extraction on sample articles.
    
    Args:
        articles_dir: Directory containing article files
        max_articles: Maximum number of articles to test (None for all)
    """
    print("=" * 70)
    print("INFORMATION EXTRACTION AGENT TEST")
    print("=" * 70)
    
    # Initialize components
    config = get_config()
    extractor = InformationExtractionAgent(config)
    parser = ArticleParser()
    
    # Get article files
    articles_path = Path(articles_dir)
    if not articles_path.exists():
        print(f"❌ Articles directory not found: {articles_dir}")
        return
    
    article_files = sorted(articles_path.glob("*.txt"))
    print(f"📁 Found {len(article_files)} article files")
    
    if max_articles:
        article_files = article_files[:max_articles]
        print(f"🔢 Testing on first {max_articles} articles")
    
    if not article_files:
        print("❌ No article files found")
        return
    
    # Test extraction on each article
    all_results = []
    for i, file_path in enumerate(article_files, 1):
        print(f"\n{'='*50}")
        print(f"ARTICLE {i}: {file_path.name}")
        print(f"{'='*50}")
        
        try:
            # Parse article
            article = parser.parse_file(file_path)
            print(f"📄 Title: {article.title}")
            print(f"📅 Published: {article.published}")
            print(f"👥 Authors: {', '.join(article.authors) if article.authors else 'None'}")
            
            # Extract information
            print("\n🔍 Extracting information...")
            result = extractor.extract_from_article(article)
            all_results.append(result)
            
            # Display results
            print(f"\n📊 EXTRACTION RESULTS:")
            print(f"   Events: {len(result['events'])}")
            print(f"   Entities: {len(result['entities'])}")
            print(f"   Temporal refs: {len(result['temporal_references'])}")
            
            # Show events
            if result['events']:
                print(f"\n🎯 EVENTS ({len(result['events'])}):")
                for j, event in enumerate(result['events'], 1):
                    print(f"   {j}. {event['description']}")
                    print(f"      📅 Date: {event['date_text']} → {event['date']}")
                    print(f"      👥 Entities: {', '.join(event['entities'])}")
                    print(f"      📊 Confidence: {event['confidence']:.2f}")
                    if event['source_text']:
                        preview = event['source_text'][:100] + "..." if len(event['source_text']) > 100 else event['source_text']
                        print(f"      📝 Source: {preview}")
                    print()
            
            # Show entities
            if result['entities']:
                print(f"🏷️  ENTITIES ({len(result['entities'])}):")
                for entity in result['entities']:
                    mentions_str = ', '.join(entity['mentions'][:3])  # Show first 3 mentions
                    if len(entity['mentions']) > 3:
                        mentions_str += f" (+{len(entity['mentions'])-3} more)"
                    print(f"   • {entity['name']} ({entity['type']}) - {entity['confidence']:.2f}")
                    print(f"     Mentions: {mentions_str}")
                print()
            
            # Show temporal references
            if result['temporal_references']:
                print(f"⏰ TEMPORAL REFERENCES ({len(result['temporal_references'])}):")
                for temp in result['temporal_references']:
                    print(f"   • '{temp['text']}' → {temp['resolved_date']} ({temp['date_type']})")
                    print(f"     Confidence: {temp['confidence']:.2f}")
                print()
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        total_events = sum(len(r['events']) for r in all_results)
        total_entities = sum(len(r['entities']) for r in all_results)
        total_temporal = sum(len(r['temporal_references']) for r in all_results)
        
        print(f"📊 Total extracted:")
        print(f"   Events: {total_events}")
        print(f"   Entities: {total_entities}")
        print(f"   Temporal references: {total_temporal}")
        
        # Average confidence scores
        if total_events > 0:
            avg_event_confidence = sum(
                event['confidence'] for r in all_results for event in r['events']
            ) / total_events
            print(f"   Average event confidence: {avg_event_confidence:.2f}")
        
        if total_entities > 0:
            avg_entity_confidence = sum(
                entity['confidence'] for r in all_results for entity in r['entities']
            ) / total_entities
            print(f"   Average entity confidence: {avg_entity_confidence:.2f}")
        
        print(f"\n✅ Successfully processed {len(all_results)}/{len(article_files)} articles")
    else:
        print("❌ No articles were successfully processed")
    
    return all_results


def test_single_article(file_path: str):
    """Test extraction on a single article file"""
    print(f"Testing extraction on: {file_path}")
    
    # Initialize components
    config = get_config()
    extractor = InformationExtractionAgent(config)
    parser = ArticleParser()
    
    try:
        # Parse and extract
        article = parser.parse_file(Path(file_path))
        result = extractor.extract_from_article(article)
        
        # Pretty print results
        print("\nExtraction Results:")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Information Extraction Agent")
    parser.add_argument(
        "--articles-dir",
        default="text_articles",
        help="Directory containing article files"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=3,
        help="Maximum number of articles to test"
    )
    parser.add_argument(
        "--file",
        help="Test on a single specific file"
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
    
    if args.file:
        test_single_article(args.file)
    else:
        test_extraction_on_articles(args.articles_dir, args.max_articles)


if __name__ == "__main__":
    main()
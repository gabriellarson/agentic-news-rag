"""
Test script for Timeline Construction Agent.

Tests construction of chronological timelines from extracted events,
including deduplication, date estimation, and relationship identification.
"""

import sys
from pathlib import Path
import logging
import json
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.agents.timeline_construction import TimelineConstructionAgent
from src.agents.information_extraction import InformationExtractionAgent
from src.embeddings.article_parser import ArticleParser

logger = logging.getLogger(__name__)


def test_timeline_construction(articles_dir: str = "text_articles", 
                             max_articles: int = 5,
                             topic: str = "Energy Sector Developments"):
    """
    Test timeline construction on extracted events from articles.
    
    Args:
        articles_dir: Directory containing article files
        max_articles: Maximum number of articles to process
        topic: Topic/theme for the timeline
    """
    print("=" * 80)
    print("TIMELINE CONSTRUCTION AGENT TEST")
    print("=" * 80)
    
    # Initialize components
    config = get_config()
    timeline_agent = TimelineConstructionAgent(config)
    extraction_agent = InformationExtractionAgent(config)
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
        print(f"🔢 Processing first {max_articles} articles")
    
    if not article_files:
        print("❌ No article files found")
        return
    
    # Extract events from all articles
    print(f"\n🔍 EXTRACTING EVENTS FROM {len(article_files)} ARTICLES")
    print("=" * 50)
    
    extraction_results = []
    total_events = 0
    
    for i, file_path in enumerate(article_files, 1):
        print(f"\nProcessing {i}/{len(article_files)}: {file_path.name}")
        
        try:
            # Parse and extract
            article = parser.parse_file(file_path)
            result = extraction_agent.extract_from_article(article)
            extraction_results.append(result)
            
            event_count = len(result['events'])
            total_events += event_count
            print(f"  ✅ Extracted {event_count} events")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   Articles processed: {len(extraction_results)}")
    print(f"   Total events extracted: {total_events}")
    
    if not extraction_results:
        print("❌ No extraction results to build timeline from")
        return
    
    # Construct timeline
    print(f"\n🏗️  CONSTRUCTING TIMELINE: {topic}")
    print("=" * 50)
    
    try:
        timeline = timeline_agent.construct_timeline(extraction_results, topic)
        
        print(f"\n✅ TIMELINE CREATED SUCCESSFULLY!")
        print(f"   Topic: {timeline.topic}")
        print(f"   Total events: {len(timeline.events)}")
        print(f"   Date range: {timeline.date_range[0]} to {timeline.date_range[1]}")
        print(f"   Confidence: {timeline.confidence:.2f}")
        print(f"   Completeness: {timeline.completeness_score:.2f}")
        print(f"   Consistency: {timeline.consistency_score:.2f}")
        
        # Display timeline events
        print(f"\n📅 TIMELINE EVENTS ({len(timeline.events)}):")
        print("=" * 60)
        
        for i, event in enumerate(timeline.events, 1):
            print(f"\n{i}. {event.description}")
            
            # Date information
            if event.date:
                print(f"   📅 Date: {event.date.strftime('%Y-%m-%d %H:%M:%S')} ({event.date_text})")
            elif event.estimated_date:
                print(f"   📅 Estimated: {event.estimated_date.strftime('%Y-%m-%d')} ({event.date_text})")
            else:
                print(f"   📅 Date: Unknown ({event.date_text})")
            
            # Event details
            print(f"   🏷️  Type: {event.event_type}")
            print(f"   📊 Confidence: {event.confidence:.2f}")
            print(f"   ⭐ Importance: {event.importance_score:.2f}")
            
            # Entities
            if event.entities:
                entities_str = ', '.join(event.entities[:5])
                if len(event.entities) > 5:
                    entities_str += f" (+{len(event.entities)-5} more)"
                print(f"   👥 Entities: {entities_str}")
            
            # Sources
            if event.sources:
                sources_str = ', '.join([Path(s).stem for s in event.sources])
                print(f"   📄 Sources: {sources_str}")
            
            # Causal relationships
            if event.causal_relationships:
                print(f"   🔗 Causes {len(event.causal_relationships)} other event(s)")
            
            # Supporting events (merged)
            if event.supporting_events:
                print(f"   🔄 Merged from {len(event.supporting_events)} similar events")
        
        # Timeline analysis
        print(f"\n📈 TIMELINE ANALYSIS:")
        print("=" * 40)
        
        # Event types distribution
        event_types = {}
        for event in timeline.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        print(f"Event types:")
        for event_type, count in sorted(event_types.items()):
            print(f"  - {event_type}: {count}")
        
        # Date coverage
        dated_events = len([e for e in timeline.events if e.date])
        estimated_events = len([e for e in timeline.events if e.estimated_date and not e.date])
        undated_events = len([e for e in timeline.events if not e.date and not e.estimated_date])
        
        print(f"\nDate coverage:")
        print(f"  - Exact dates: {dated_events}")
        print(f"  - Estimated dates: {estimated_events}")
        print(f"  - Undated: {undated_events}")
        
        # Causal relationships
        total_relationships = sum(len(e.causal_relationships) for e in timeline.events)
        print(f"\nCausal relationships: {total_relationships}")
        
        # Timeline metadata
        if timeline.metadata:
            print(f"\nMetadata:")
            for key, value in timeline.metadata.items():
                print(f"  - {key}: {value}")
        
        print(f"\n🎯 TIMELINE QUALITY ASSESSMENT:")
        print("=" * 40)
        
        # Quality metrics
        quality_score = (timeline.confidence + timeline.completeness_score + timeline.consistency_score) / 3
        
        print(f"Overall Quality: {quality_score:.2f}/1.0")
        print(f"  - Event Confidence: {timeline.confidence:.2f}/1.0")
        print(f"  - Completeness: {timeline.completeness_score:.2f}/1.0")
        print(f"  - Consistency: {timeline.consistency_score:.2f}/1.0")
        
        if quality_score >= 0.8:
            print("✅ High quality timeline")
        elif quality_score >= 0.6:
            print("⚠️  Moderate quality timeline")
        else:
            print("❌ Low quality timeline - may need more data")
        
        return timeline
        
    except Exception as e:
        print(f"❌ Timeline construction failed: {e}")
        logger.error(f"Timeline construction error: {e}")
        return None


def save_timeline_json(timeline, output_file: str):
    """Save timeline to JSON file"""
    if not timeline:
        return
    
    timeline_data = {
        'topic': timeline.topic,
        'date_range': [
            timeline.date_range[0].isoformat() if timeline.date_range[0] else None,
            timeline.date_range[1].isoformat() if timeline.date_range[1] else None
        ],
        'confidence': timeline.confidence,
        'completeness_score': timeline.completeness_score,
        'consistency_score': timeline.consistency_score,
        'metadata': timeline.metadata,
        'events': [
            {
                'id': event.id,
                'description': event.description,
                'date': event.date.isoformat() if event.date else None,
                'estimated_date': event.estimated_date.isoformat() if event.estimated_date else None,
                'date_text': event.date_text,
                'entities': event.entities,
                'sources': event.sources,
                'confidence': event.confidence,
                'event_type': event.event_type,
                'importance_score': event.importance_score,
                'causal_relationships': event.causal_relationships,
                'supporting_events': event.supporting_events
            }
            for event in timeline.events
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Timeline saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Timeline Construction Agent")
    parser.add_argument(
        "--articles-dir",
        default="text_articles",
        help="Directory containing article files"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Maximum number of articles to process"
    )
    parser.add_argument(
        "--topic",
        default="Energy Sector Developments",
        help="Timeline topic/theme"
    )
    parser.add_argument(
        "--output",
        help="Save timeline to JSON file"
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
    
    # Test timeline construction
    timeline = test_timeline_construction(
        articles_dir=args.articles_dir,
        max_articles=args.max_articles,
        topic=args.topic
    )
    
    # Save to file if requested
    if args.output and timeline:
        save_timeline_json(timeline, args.output)


if __name__ == "__main__":
    main()
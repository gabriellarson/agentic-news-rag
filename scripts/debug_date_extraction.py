"""
Debug script to diagnose date extraction issues.

This script helps identify where the date extraction pipeline is failing:
1. Shows what date information is in the article text
2. Shows what the LLM extracts as date_text
3. Shows what the parsing function returns
"""

import sys
from pathlib import Path
import json
import logging

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.information_extraction import InformationExtractionAgent
from src.embeddings.article_parser import ArticleParser
from src.config import get_config


def debug_date_extraction(article_file=None):
    """Debug date extraction on a specific article or the first available one"""
    
    config = get_config()
    extractor = InformationExtractionAgent(config)
    parser = ArticleParser()
    
    # Get article file
    if article_file:
        article_path = Path(article_file)
    else:
        articles_dir = Path('text_articles')
        article_files = list(articles_dir.glob('*.txt'))
        if not article_files:
            print("❌ No article files found in text_articles directory")
            return
        article_path = article_files[0]
    
    print(f"🔍 DEBUGGING DATE EXTRACTION")
    print(f"📄 Article: {article_path.name}")
    print("=" * 70)
    
    # Parse article
    article = parser.parse_file(article_path)
    
    print(f"📋 ARTICLE INFO:")
    print(f"   Title: {article.title}")
    print(f"   Published: {article.published}")
    print(f"   Authors: {', '.join(article.authors) if article.authors else 'None'}")
    print()
    
    print(f"📝 ARTICLE TEXT (first 800 chars):")
    print("-" * 50)
    print(article.content[:800] + "..." if len(article.content) > 800 else article.content)
    print("-" * 50)
    print()
    
    # Look for date patterns in the text
    import re
    date_patterns = [
        r'\b(\d{4})\b',  # Years
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
        r'\b(last|this|next)\s+(year|month|week|quarter)\b',  # Relative dates
        r'\b(yesterday|today|tomorrow)\b',  # Relative days
        r'\b(end of \d{4})\b',  # End of year
        r'\b(second quarter|Q[1-4])\b',  # Quarters
    ]
    
    print(f"🔍 DATE PATTERNS FOUND IN TEXT:")
    found_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, article.content, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                found_dates.append(match)
                print(f"   ✓ '{match}'")
    
    if not found_dates:
        print("   ❌ No obvious date patterns found in article text")
    print()
    
    # Extract using LLM
    print(f"🤖 LLM EXTRACTION RESULTS:")
    print("-" * 50)
    
    try:
        # Temporarily patch the extractor to capture raw LLM response
        original_extract_events = extractor._extract_events
        raw_llm_response = None
        
        def capture_llm_response(text, article_id):
            nonlocal raw_llm_response
            
            # Call the original method but capture the response
            import json
            from openai import OpenAI
            
            # Reconstruct the prompt (this is a bit hacky but necessary for debugging)
            prompt = f"""
        Extract key events from this news article. For each event, identify:
        1. A clear description of what happened
        2. When it happened - CAREFULLY extract ANY temporal references from the article text
        3. Who/what was involved (entities)
        4. Your confidence in this being a significant event (0-1)

        Article text:
        {text}

        IMPORTANT FOR DATE EXTRACTION:
        - Look for specific dates: "January 11, 2024", "March 2023", "2020", etc.
        - Look for relative dates: "last year", "this quarter", "next month", "yesterday", etc.
        - Look for time periods: "second quarter", "Q4", "end of 2022", etc.
        - Look for contextual dates: "when the pandemic began", "since the war started", etc.
        - If you find ANY temporal reference in the article, extract it exactly as written
        - Only use "not specified" if there is absolutely no temporal information for that event

        Return a JSON array of events with this structure:
        [
          {{
            "description": "Clear description of the event",
            "date_text": "Exact temporal reference from article or 'not specified'",
            "entities": ["entity1", "entity2"],
            "confidence": 0.9,
            "source_text": "Relevant snippet from article"
          }}
        ]

        Focus on major events, announcements, decisions, transactions, or significant developments.
        """
            
            try:
                print(f"🔄 Making LLM request...")
                print(f"   Model: {extractor.llm_model}")
                print(f"   Endpoint: {extractor.llm_endpoint}")
                
                response = extractor.client.chat.completions.create(
                    model=extractor.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.1,
                    n=1
                )
                
                raw_llm_response = response.choices[0].message.content
                print(f"🔍 RAW LLM RESPONSE:")
                print("-" * 40)
                print(repr(raw_llm_response))  # Use repr to see any hidden characters
                print("-" * 40)
                print("Response length:", len(raw_llm_response))
                print("Response type:", type(raw_llm_response))
                print()
                
            except Exception as e:
                print(f"❌ Error capturing LLM response: {e}")
                import traceback
                traceback.print_exc()
            
            # Now call the original method
            return original_extract_events(text, article_id)
        
        # Temporarily replace the method
        extractor._extract_events = capture_llm_response
        
        result = extractor.extract_from_article(article)
        
        # Restore the original method
        extractor._extract_events = original_extract_events
        
        print(f"Events extracted: {len(result['events'])}")
        print()
        
        for i, event in enumerate(result['events']):
            print(f"Event {i+1}:")
            print(f"   Description: {event['description'][:80]}...")
            print(f"   📅 date_text: \"{event.get('date_text', 'MISSING')}\"")
            print(f"   📅 date (parsed): {event.get('date', 'MISSING')}")
            
            # Test manual parsing of the date_text
            if event.get('date_text') and event.get('date_text') != 'not specified':
                manual_parsed = extractor._parse_date_text(event['date_text'])
                print(f"   🔧 Manual parse test: {manual_parsed}")
            
            print(f"   👥 Entities: {', '.join(event.get('entities', []))}")
            print(f"   📊 Confidence: {event.get('confidence', 0)}")
            print()
        
        # Save full result for inspection
        debug_file = 'debug_extraction_result.json'
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        print(f"💾 Full extraction result saved to: {debug_file}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


def test_date_parsing():
    """Test the date parsing function directly"""
    print(f"\n🧪 TESTING DATE PARSING FUNCTION:")
    print("=" * 70)
    
    config = get_config()
    agent = InformationExtractionAgent(config)
    
    test_cases = [
        'January 11, 2024',
        'end of 2022', 
        '2018',
        'last year',
        'Thursday',
        'second quarter',
        'not specified',
        '2024-01-11',
        'October 7, 2023',
        'end of last year',
        'Q2 2024',
        'Nov 15, 2023'
    ]
    
    print("Testing _parse_date_text function:")
    print("-" * 40)
    for date_text in test_cases:
        result = agent._parse_date_text(date_text)
        status = "✓" if result else "❌"
        print(f"{status} {date_text:<20} -> {result}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug date extraction issues")
    parser.add_argument(
        "--article",
        help="Specific article file to test (default: first article found)"
    )
    parser.add_argument(
        "--test-parsing",
        action="store_true",
        help="Also test the date parsing function directly"
    )
    
    args = parser.parse_args()
    
    debug_date_extraction(args.article)
    
    if args.test_parsing:
        test_date_parsing()


if __name__ == "__main__":
    main()
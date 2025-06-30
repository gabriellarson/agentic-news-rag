"""
Information Extraction Agent for News RAG System

Extracts structured information from retrieved articles including:
- Temporal information (dates, events, time references)  
- Named entities (people, organizations, locations)
- Key events and relationships
- Factual claims with confidence scores
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re
from dataclasses import dataclass

# Import OpenAI for LLM integration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from ..config import get_config
    from ..embeddings.article_parser import Article
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    from embeddings.article_parser import Article

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEvent:
    """Represents an extracted event with temporal and contextual information"""
    description: str
    date: Optional[datetime]
    date_text: str  # Original date text from article
    entities: List[str]  # People, orgs, locations involved
    confidence: float  # 0-1 confidence score
    article_id: str
    source_text: str  # Text snippet where event was found


@dataclass
class ExtractedEntity:
    """Represents an extracted named entity"""
    name: str
    type: str  # PERSON, ORGANIZATION, LOCATION, etc.
    mentions: List[str]  # Different ways entity is mentioned
    confidence: float
    article_id: str


@dataclass
class TemporalReference:
    """Represents a temporal reference found in text"""
    text: str  # Original temporal expression
    resolved_date: Optional[datetime]  # Resolved absolute date
    date_type: str  # ABSOLUTE, RELATIVE, RANGE
    confidence: float


class InformationExtractionAgent:
    """Agent for extracting structured information from news articles"""
    
    def __init__(self, config=None):
        """Initialize the extraction agent"""
        if config is None:
            config = get_config()
            
        self.config = config
        self.llm_endpoint = config.get('llm', 'endpoint', default='http://localhost:8001/v1')
        self.llm_model = config.get('llm', 'model', default='qwen3-30b')
        self.confidence_threshold = config.get('extraction', 'confidence_threshold', default=0.7)
        
        # Initialize LLM client
        if OpenAI is None:
            logger.error("OpenAI package not found. Install with: pip install openai")
            raise ImportError("OpenAI package is required for information extraction. Install with: pip install openai")
        
        self.client = OpenAI(
            base_url=self.llm_endpoint,
            api_key="not-needed"  # llama.cpp doesn't require API key
        )
        logger.info(f"Initialized LLM client for {self.llm_endpoint}")
    
    def extract_from_article(self, article: Article) -> Dict[str, Any]:
        """
        Extract all information from a single article.
        
        Args:
            article: Parsed article object
            
        Returns:
            Dictionary containing extracted events, entities, and temporal info
        """
        article_text = f"{article.title}\n{article.subtitle or ''}\n{article.content}"
        
        # Extract different types of information
        events = self._extract_events(article_text, str(article.file_path))
        entities = self._extract_entities(article_text, str(article.file_path))
        temporal_refs = self._extract_temporal_references(article_text, article.published)
        
        return {
            'article_id': str(article.file_path),
            'article_title': article.title,
            'published_date': article.published.isoformat(),
            'events': [self._event_to_dict(e) for e in events],
            'entities': [self._entity_to_dict(e) for e in entities],
            'temporal_references': [self._temporal_to_dict(t) for t in temporal_refs],
            'extraction_timestamp': datetime.now().isoformat()
        }
    
    def extract_from_articles(self, articles: List[Article]) -> List[Dict[str, Any]]:
        """
        Extract information from multiple articles.
        
        Args:
            articles: List of parsed articles
            
        Returns:
            List of extraction results
        """
        results = []
        for article in articles:
            try:
                result = self.extract_from_article(article)
                results.append(result)
                logger.info(f"Extracted info from: {article.title}")
            except Exception as e:
                logger.error(f"Failed to extract from {article.title}: {e}")
                continue
        
        return results
    
    def _extract_events(self, text: str, article_id: str) -> List[ExtractedEvent]:
        """Extract events using LLM"""
        if not self.client:
            logger.error("LLM client not initialized - cannot extract events")
            return []
        
        logger.info("Extracting events using LLM...")
        
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
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw event response: {content[:200]}...")
            
            # Try to extract JSON from response
            try:
                events_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                # Try to extract and clean JSON from response
                events_data = self._extract_json_from_text(content, "events")
                if not events_data:
                    return []
            
            events = []
            for event_data in events_data:
                # Use LLM to resolve date from date_text
                resolved_date = self._resolve_date_with_llm(
                    event_data.get('date_text', ''),
                    article_text=text,
                    article_id=article_id
                )
                
                event = ExtractedEvent(
                    description=event_data['description'],
                    date=resolved_date,
                    date_text=event_data.get('date_text', 'not specified'),
                    entities=event_data.get('entities', []),
                    confidence=event_data.get('confidence', 0.5),
                    article_id=article_id,
                    source_text=event_data.get('source_text', '')[:200]  # Limit length
                )
                
                if event.confidence >= self.confidence_threshold:
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to extract events: {e}")
            return []
    
    def _extract_entities(self, text: str, article_id: str) -> List[ExtractedEntity]:
        """Extract named entities using LLM"""
        if not self.client:
            logger.error("LLM client not initialized - cannot extract entities")
            return []
        
        logger.info("Extracting entities using LLM...")
        
        prompt = f"""
        Extract important named entities from this news article. Focus on:
        - PERSON: Individual people (executives, officials, analysts, etc.)
        - ORGANIZATION: Companies, institutions, agencies, etc.
        - LOCATION: Countries, cities, regions, facilities
        - OTHER: Products, technologies, financial instruments, etc.

        Article text:
        {text}

        Return a JSON array with this structure:
        [
          {{
            "name": "Entity name (canonical form)",
            "type": "PERSON|ORGANIZATION|LOCATION|OTHER",
            "mentions": ["mention1", "mention2"],
            "confidence": 0.9
          }}
        ]

        Only include entities that are central to the article's content.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw entity response: {content[:200]}...")
            
            # Try to extract JSON from response
            try:
                entities_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                # Try to extract and clean JSON from response
                entities_data = self._extract_json_from_text(content, "entities")
                if not entities_data:
                    return []
            
            entities = []
            for entity_data in entities_data:
                entity = ExtractedEntity(
                    name=entity_data['name'],
                    type=entity_data.get('type', 'OTHER'),
                    mentions=entity_data.get('mentions', [entity_data['name']]),
                    confidence=entity_data.get('confidence', 0.5),
                    article_id=article_id
                )
                
                if entity.confidence >= self.confidence_threshold:
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    def _extract_temporal_references(self, text: str, 
                                   reference_date: datetime) -> List[TemporalReference]:
        """Extract and resolve temporal references"""
        temporal_refs = []
        
        # Common temporal patterns
        patterns = {
            'ABSOLUTE': [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY, MM-DD-YYYY
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b'
            ],
            'RELATIVE': [
                r'\b(yesterday|today|tomorrow)\b',
                r'\b(last|this|next)\s+(week|month|year|quarter)\b',
                r'\b\d+\s+(days?|weeks?|months?|years?)\s+ago\b',
                r'\b(earlier|later)\s+(this|that)\s+(week|month|year)\b',
                r'\b(recently|shortly|soon)\b'
            ]
        }
        
        for date_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    temporal_text = match.group(0)
                    resolved_date = self._resolve_temporal_reference(
                        temporal_text, reference_date
                    )
                    
                    temporal_ref = TemporalReference(
                        text=temporal_text,
                        resolved_date=resolved_date,
                        date_type=date_type,
                        confidence=0.8 if resolved_date else 0.3
                    )
                    temporal_refs.append(temporal_ref)
        
        return temporal_refs
    
    def _parse_date_text(self, date_text: str) -> Optional[datetime]:
        """Parse date from extracted text"""
        if not date_text or date_text.lower() in ['not specified', 'unknown', '']:
            return None
        
        # Common date patterns
        patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
        ]
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for pattern in patterns:
            match = re.search(pattern, date_text.lower())
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if groups[0].isdigit() and groups[2].isdigit():  # YYYY-MM-DD or MM/DD/YYYY
                            if len(groups[0]) == 4:  # YYYY-MM-DD
                                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                            else:  # MM/DD/YYYY
                                month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                        else:  # Month name format
                            if groups[0] in months:  # Month DD, YYYY
                                month, day, year = months[groups[0]], int(groups[1]), int(groups[2])
                            else:  # DD Month YYYY
                                day, month, year = int(groups[0]), months[groups[1]], int(groups[2])
                        
                        return datetime(year, month, day)
                except (ValueError, KeyError):
                    continue
        
        return None
    
    def _resolve_temporal_reference(self, temporal_text: str, 
                                  reference_date: datetime) -> Optional[datetime]:
        """Resolve relative temporal references to absolute dates"""
        text_lower = temporal_text.lower()
        
        if 'yesterday' in text_lower:
            return reference_date - timedelta(days=1)
        elif 'today' in text_lower:
            return reference_date
        elif 'tomorrow' in text_lower:
            return reference_date + timedelta(days=1)
        elif 'last week' in text_lower:
            return reference_date - timedelta(weeks=1)
        elif 'this week' in text_lower:
            return reference_date
        elif 'next week' in text_lower:
            return reference_date + timedelta(weeks=1)
        elif 'last month' in text_lower:
            return reference_date - timedelta(days=30)
        elif 'this month' in text_lower:
            return reference_date
        elif 'next month' in text_lower:
            return reference_date + timedelta(days=30)
        elif 'last year' in text_lower:
            return reference_date - timedelta(days=365)
        elif 'this year' in text_lower:
            return reference_date
        elif 'next year' in text_lower:
            return reference_date + timedelta(days=365)
        
        # Handle "X days/weeks/months ago"
        ago_match = re.search(r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago', text_lower)
        if ago_match:
            num = int(ago_match.group(1))
            unit = ago_match.group(2)
            
            if 'day' in unit:
                return reference_date - timedelta(days=num)
            elif 'week' in unit:
                return reference_date - timedelta(weeks=num)
            elif 'month' in unit:
                return reference_date - timedelta(days=num * 30)
            elif 'year' in unit:
                return reference_date - timedelta(days=num * 365)
        
        return None
    
    def _resolve_date_with_llm(self, date_text: str, article_text: str, article_id: str) -> Optional[datetime]:
        """Use LLM to resolve any date text to an absolute date"""
        if not date_text or date_text.lower() in ['not specified', 'unknown', '']:
            return None
        
        # Extract article publication date for context
        try:
            # Try to get publication date from article metadata if available
            pub_date_match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', article_id)
            if pub_date_match:
                reference_date = datetime.fromisoformat(pub_date_match.group(1))
            else:
                reference_date = datetime.now()
        except:
            reference_date = datetime.now()
        
        prompt = f"""
        Convert this temporal reference to an absolute date in ISO format (YYYY-MM-DD).
        
        Temporal reference: "{date_text}"
        Reference context date: {reference_date.strftime('%Y-%m-%d')}
        
        Article excerpt for context:
        {article_text[:1000]}
        
        Instructions:
        - Convert relative dates like "last year", "second quarter", "end of 2022" to absolute dates
        - For quarters, use the end date of that quarter
        - For years only (like "2020"), use January 1st of that year
        - For contextual references like "when the pandemic began", use known historical dates
        - For "last year", "this year" etc., calculate relative to the reference date
        - If the date cannot be reasonably determined, return "UNABLE_TO_RESOLVE"
        
        Examples:
        - "2020" → "2020-01-01"
        - "second quarter" → "2024-06-30" (assuming current year)
        - "end of last year" → "2023-12-31" (if reference is 2024)
        - "when the pandemic began" → "2020-03-01"
        - "since the war in Ukraine began" → "2022-02-24"
        
        Return only the ISO date (YYYY-MM-DD) or "UNABLE_TO_RESOLVE":
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                temperature=0.1,
                n=1
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"LLM date resolution result for '{date_text}': '{result}' (length: {len(result)})")
            
            # Try to parse the returned date
            if result and result != "UNABLE_TO_RESOLVE":
                try:
                    # Handle various date formats the LLM might return
                    if re.match(r'\d{4}-\d{1,2}-\d{1,2}', result):
                        resolved = datetime.fromisoformat(result)
                        logger.debug(f"Successfully parsed date: {date_text} -> {resolved}")
                        return resolved
                    elif re.match(r'\d{4}-\d{1,2}', result):
                        resolved = datetime.fromisoformat(result + "-01")
                        logger.debug(f"Successfully parsed date: {date_text} -> {resolved}")
                        return resolved
                    elif re.match(r'\d{4}', result):
                        resolved = datetime(int(result), 1, 1)
                        logger.debug(f"Successfully parsed date: {date_text} -> {resolved}")
                        return resolved
                except ValueError:
                    logger.warning(f"Could not parse LLM date result: {result}")
            else:
                logger.debug(f"LLM could not resolve date: {date_text} -> {result}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to resolve date with LLM: {e}")
            return None
    
    def _extract_json_from_text(self, text: str, data_type: str) -> List[Dict[str, Any]]:
        """
        Robust JSON extraction from LLM response text.
        
        Args:
            text: Raw LLM response text
            data_type: Type of data being extracted (for logging)
            
        Returns:
            List of parsed JSON objects or empty list
        """
        import re
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON array found in {data_type} response")
            return []
        
        json_str = json_match.group(0)
        
        # Clean common JSON formatting issues
        fixes = [
            (r',\s*}', '}'),                    # Remove trailing commas in objects
            (r',\s*]', ']'),                    # Remove trailing commas in arrays
            (r'\\n', ' '),                      # Replace newlines in strings
            (r'\\"', '"'),                      # Fix escaped quotes
            (r'"\s*\n\s*"', '"'),              # Fix split strings across lines
            (r'(\w+):', r'"\1":'),             # Quote unquoted keys (basic cases)
        ]
        
        for pattern, replacement in fixes:
            json_str = re.sub(pattern, replacement, json_str)
        
        # Try progressively more aggressive fixes
        attempts = [
            json_str,  # Original cleaned version
            # Try to fix incomplete JSON by finding valid start
            re.sub(r'^[^[]*(\[.*)', r'\1', json_str, flags=re.DOTALL),
            # Try to find just the first complete object and wrap in array
        ]
        
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse {data_type} JSON attempt: {e}")
                continue
        
        # Last resort: try to extract individual objects
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        objects = re.findall(object_pattern, json_str)
        
        if objects:
            parsed_objects = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    parsed_objects.append(obj)
                except json.JSONDecodeError:
                    continue
            
            if parsed_objects:
                logger.info(f"Recovered {len(parsed_objects)} {data_type} objects from malformed JSON")
                return parsed_objects
        
        logger.warning(f"Could not parse {data_type} JSON after all attempts: {json_str[:200]}...")
        return []
    
    def _event_to_dict(self, event: ExtractedEvent) -> Dict[str, Any]:
        """Convert ExtractedEvent to dictionary"""
        return {
            'description': event.description,
            'date': event.date.isoformat() if event.date else None,
            'date_text': event.date_text,
            'entities': event.entities,
            'confidence': event.confidence,
            'source_text': event.source_text
        }
    
    def _entity_to_dict(self, entity: ExtractedEntity) -> Dict[str, Any]:
        """Convert ExtractedEntity to dictionary"""
        return {
            'name': entity.name,
            'type': entity.type,
            'mentions': entity.mentions,
            'confidence': entity.confidence
        }
    
    def _temporal_to_dict(self, temporal: TemporalReference) -> Dict[str, Any]:
        """Convert TemporalReference to dictionary"""
        return {
            'text': temporal.text,
            'resolved_date': temporal.resolved_date.isoformat() if temporal.resolved_date else None,
            'date_type': temporal.date_type,
            'confidence': temporal.confidence
        }


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import init_config
    from embeddings.article_parser import ArticleParser
    
    # Initialize components
    config = init_config()
    extractor = InformationExtractionAgent(config)
    parser = ArticleParser()
    
    # Test with a sample article
    articles_dir = Path("text_articles")
    if articles_dir.exists():
        article_files = list(articles_dir.glob("*.txt"))
        if article_files:
            # Parse first article
            article = parser.parse_file(article_files[0])
            
            # Extract information
            result = extractor.extract_from_article(article)
            
            # Print results
            print(f"\nExtraction Results for: {article.title}")
            print("=" * 60)
            
            print(f"\nEvents ({len(result['events'])}):")
            for i, event in enumerate(result['events'], 1):
                print(f"{i}. {event['description']}")
                print(f"   Date: {event['date_text']} -> {event['date']}")
                print(f"   Entities: {', '.join(event['entities'])}")
                print(f"   Confidence: {event['confidence']:.2f}")
            
            print(f"\nEntities ({len(result['entities'])}):")
            for entity in result['entities']:
                print(f"- {entity['name']} ({entity['type']}) - {entity['confidence']:.2f}")
            
            print(f"\nTemporal References ({len(result['temporal_references'])}):")
            for temp in result['temporal_references']:
                print(f"- '{temp['text']}' -> {temp['resolved_date']} ({temp['date_type']})")
        else:
            print("No article files found in text_articles directory")
    else:
        print("text_articles directory not found")
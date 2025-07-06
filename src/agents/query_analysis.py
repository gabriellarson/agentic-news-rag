"""
Query Analysis Agent for News RAG System

This agent analyzes user queries to:
1. Understand user intent
2. Generate multiple search queries (query expansion)
3. Extract temporal constraints
4. Determine optimal search parameters
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging

try:
    from ..config import get_config
    from ..llm.interface import LLMInterface
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    from llm.interface import LLMInterface

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries based on intent"""
    FACTUAL = "factual"  # Looking for specific facts
    CONCEPTUAL = "conceptual"  # Looking for broader concepts/themes
    TEMPORAL = "temporal"  # Focused on time-based events
    ENTITY = "entity"  # Focused on specific people/organizations
    COMPARATIVE = "comparative"  # Comparing different events/entities


@dataclass
class TemporalConstraint:
    """Represents temporal constraints extracted from query"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    relative_terms: List[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'relative_terms': self.relative_terms or []
        }


@dataclass
class QueryAnalysis:
    """Complete analysis of a user query"""
    original_query: str
    query_type: QueryType
    expanded_queries: List[str]
    entities: List[str]
    temporal_constraints: TemporalConstraint
    search_alpha: float  # Weight for dense vs sparse search
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'original_query': self.original_query,
            'query_type': self.query_type.value,
            'expanded_queries': self.expanded_queries,
            'entities': self.entities,
            'temporal_constraints': self.temporal_constraints.to_dict(),
            'search_alpha': self.search_alpha,
            'confidence': self.confidence
        }


class QueryAnalysisAgent:
    """Agent for analyzing and expanding user queries"""
    
    def __init__(self, config=None):
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Initialize LLM interface
        self.llm = LLMInterface(config)
        
        # Get query analysis configuration
        qa_config = config.get_query_analysis_config()
        self.max_expanded_queries = qa_config.get('max_expanded_queries', 5)
        self.type_weights = qa_config.get('type_weights', {})
        
        # Relative date patterns
        self.relative_date_patterns = {
            r'\b(today|now)\b': 0,
            r'\b(yesterday)\b': -1,
            r'\b(tomorrow)\b': 1,
            r'\b(\d+)\s+days?\s+ago\b': lambda m: -int(m.group(1)),
            r'\blast\s+week\b': -7,
            r'\blast\s+month\b': -30,
            r'\blast\s+year\b': -365,
            r'\bthis\s+week\b': 0,
            r'\bthis\s+month\b': 0,
            r'\bthis\s+year\b': 0,
        }
    
    def analyze_query(self, query: str, current_date: datetime = None) -> QueryAnalysis:
        """Main method to analyze a user query"""
        if current_date is None:
            current_date = datetime.now()
        
        logger.info(f"Analyzing query: {query}")
        
        try:
            # Extract components
            query_type = self._classify_query_type(query)
            entities = self._extract_entities(query)
            temporal_constraints = self._extract_temporal_constraints(query, current_date)
            expanded_queries = self._expand_query(query, query_type, entities)
            search_alpha = self._determine_search_alpha(query_type, entities)
            
            analysis = QueryAnalysis(
                original_query=query,
                query_type=query_type,
                expanded_queries=expanded_queries,
                entities=entities,
                temporal_constraints=temporal_constraints,
                search_alpha=search_alpha,
                confidence=0.85  # TODO: Implement confidence scoring
            )
            
            logger.info(f"Query analysis complete: type={query_type.value}, "
                       f"entities={len(entities)}, alpha={search_alpha}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return basic analysis as fallback
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.CONCEPTUAL,
                expanded_queries=[query],
                entities=[],
                temporal_constraints=TemporalConstraint(),
                search_alpha=0.65,
                confidence=0.5
            )
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query using LLM"""
        prompt = f"""Classify this news search query into one category:

Query: "{query}"

Categories:
- FACTUAL: Looking for specific facts or events
- CONCEPTUAL: Looking for broader concepts, trends, or themes  
- TEMPORAL: Focused on time-based events or chronology
- ENTITY: Focused on specific people, companies, or organizations
- COMPARATIVE: Comparing different events or entities

Answer with only one word: FACTUAL, CONCEPTUAL, TEMPORAL, ENTITY, or COMPARATIVE"""

        try:
            content = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000,
                output_type="text"
            )
            
            # Extract the last word that matches our categories
            # Split by lines and look for category words
            lines = content.strip().split('\n')
            category = None
            
            # Try each line, looking for a valid category
            for line in reversed(lines):
                line = line.strip().upper()
                # Check if this line contains a valid category
                for query_type in QueryType:
                    if query_type.name in line:
                        # Found a category in this line
                        # Extract just the category word
                        words = line.split()
                        for word in words:
                            if word == query_type.name:
                                return query_type
            
            # If no match found in structured parsing, try simple extraction
            # Look for the first occurrence of any category word
            content_upper = content.upper()
            for query_type in QueryType:
                if query_type.name in content_upper:
                    return query_type
            
            # Default to conceptual if no match
            logger.warning(f"Unknown query category in response: '{content[:100]}...', defaulting to CONCEPTUAL")
            return QueryType.CONCEPTUAL
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return QueryType.CONCEPTUAL
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query"""
        if not self.config.get('query_analysis', 'entity_extraction', 'enabled', default=True):
            return []
        
        prompt = f"""Extract named entities from this query:

Query: "{query}"

Find all:
- People names
- Company names  
- Organization names
- Location names

Return ONLY a JSON array with the entities. No explanations.

Examples:
Query: "Apple CEO Tim Cook visited California"
["Apple", "Tim Cook", "California"]

Query: "What happened yesterday?"
[]

Query: "{query}"
JSON array:"""

        try:
            entities = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000,
                output_type="json_array"
            )
            
            # Filter out any non-string entries and clean up
            return [str(e).strip() for e in entities if e and str(e).strip()]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_temporal_constraints(self, query: str, current_date: datetime) -> TemporalConstraint:
        """Extract temporal constraints from the query"""
        if not self.config.get('query_analysis', 'temporal_extraction', 'enabled', default=True):
            return TemporalConstraint()
        
        constraint = TemporalConstraint()
        query_lower = query.lower()
        
        # Extract relative terms
        relative_terms = []
        for pattern, offset in self.relative_date_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                relative_terms.append(match.group(0))
                
                # Calculate date based on offset
                if callable(offset):
                    days_offset = offset(match)
                else:
                    days_offset = offset
                
                if days_offset != 0:
                    target_date = current_date + timedelta(days=days_offset)
                    
                    # Set appropriate constraint
                    if days_offset < 0:
                        constraint.start_date = target_date
                        constraint.end_date = current_date
                    else:
                        constraint.start_date = current_date
                        constraint.end_date = target_date
        
        constraint.relative_terms = relative_terms
        
        # Also check for explicit date ranges using LLM
        if not constraint.start_date and not constraint.end_date:
            constraint = self._extract_explicit_dates(query, current_date, constraint)
        
        return constraint
    
    def _extract_explicit_dates(self, query: str, current_date: datetime, 
                              constraint: TemporalConstraint) -> TemporalConstraint:
        """Extract explicit date mentions using LLM"""
        prompt = f"""Extract dates from this query:

Query: "{query}"
Current date: {current_date.strftime('%Y-%m-%d')}

Find any explicit dates like:
- Specific dates (January 1, 2024)
- Months/years (March 2023)
- Date ranges (between X and Y)

Return ONLY a JSON object with dates in YYYY-MM-DD format:
{{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "date_mentions": ["date1", "date2"]}}

If no dates found, return:
{{"start_date": null, "end_date": null, "date_mentions": []}}

JSON:"""

        try:
            date_info = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000,
                output_type="json_object"
            )
            
            # Process start_date
            if date_info.get('start_date') and date_info['start_date'] not in ['null', None, 'None']:
                try:
                    # Handle various date formats
                    date_str = str(date_info['start_date']).strip()
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                        constraint.start_date = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    pass
            
            # Process end_date
            if date_info.get('end_date') and date_info['end_date'] not in ['null', None, 'None']:
                try:
                    date_str = str(date_info['end_date']).strip()
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                        constraint.end_date = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    pass
                    
        except Exception as e:
            logger.error(f"Error extracting explicit dates: {e}")
        
        return constraint
    
    def _expand_query(self, query: str, query_type: QueryType, 
                     entities: List[str]) -> List[str]:
        """Generate multiple search queries from the original"""
        # Always include original
        expanded = [query]
        
        # Type-specific prompt
        type_guidance = {
            QueryType.FACTUAL: "Focus on specific facts, events, and precise terminology",
            QueryType.CONCEPTUAL: "Include related concepts, themes, and broader terms",
            QueryType.TEMPORAL: "Emphasize time-related aspects and chronological terms",
            QueryType.ENTITY: f"Focus on variations of entity names: {', '.join(entities)}",
            QueryType.COMPARATIVE: "Include comparison terms and contrasting elements"
        }
        
        prompt = f"""Generate exactly 4 concise alternative search queries for: "{query}"

Guidelines: {type_guidance.get(query_type, '')}

IMPORTANT: Each query must be:
- Short and concise (max 10-15 words)
- A complete search query
- Different from the original but related
- No explanations or bullet points

Format your response as a simple numbered list:
1. [query 1]
2. [query 2]
3. [query 3]
4. [query 4]

Do not include anything else in your response."""

        try:
            content = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.5,
                max_tokens=2000,
                output_type="text"
            )
            
            # Parse queries from response
            for line in content.split('\n'):
                line = line.strip()
                
                # Skip empty lines and very short lines
                if not line or len(line) < 5:
                    continue
                
                # Remove numbering (1., 2), etc.
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                
                # Remove quotes and extra punctuation
                line = line.strip('"\'')
                
                # Skip lines that are explanations or too long
                if any(keyword in line.lower() for keyword in ['okay', 'the user', 'let me', 'first', 'alternative']):
                    continue
                
                # Skip extremely long lines (likely not queries)
                if len(line) > 150:
                    continue
                
                # Skip lines with bullet points or special formatting
                if line.startswith(('*', '-', '•', '🔹', '🔸')):
                    # Extract the actual query part after the bullet
                    line = re.sub(r'^[\*\-•🔹🔸]\s*', '', line)
                
                # Add to expanded if it's a valid query
                if line and line not in expanded and 10 <= len(line) <= 150:
                    expanded.append(line)
                    
                    # Stop if we have enough queries
                    if len(expanded) >= self.max_expanded_queries:
                        break
            
            # If we didn't get enough expansions, try a simpler approach
            if len(expanded) < 3:
                # Generate simple variations
                if query_type == QueryType.FACTUAL:
                    expanded.append(f"latest news {query}")
                    expanded.append(f"{query} recent developments")
                elif query_type == QueryType.CONCEPTUAL:
                    expanded.append(f"{query} trends analysis")
                    expanded.append(f"{query} industry impact")
                elif query_type == QueryType.TEMPORAL:
                    expanded.append(f"{query} timeline")
                    expanded.append(f"chronology {query}")
                elif query_type == QueryType.ENTITY and entities:
                    for entity in entities[:2]:
                        expanded.append(f"{entity} news {query.replace(entity, '')}")
                elif query_type == QueryType.COMPARATIVE:
                    expanded.append(f"{query} comparison analysis")
                    expanded.append(f"{query} differences similarities")
            
            # Ensure uniqueness and limit
            unique_expanded = []
            seen = set()
            for q in expanded:
                q_lower = q.lower().strip()
                if q_lower not in seen:
                    seen.add(q_lower)
                    unique_expanded.append(q)
                    if len(unique_expanded) >= self.max_expanded_queries:
                        break
            
            return unique_expanded
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return expanded
    
    def _determine_search_alpha(self, query_type: QueryType, 
                               entities: List[str]) -> float:
        """Determine optimal alpha weight for hybrid search"""
        # Alpha: 0.0 = pure sparse (keyword), 1.0 = pure dense (semantic)
        
        # Use configured weights if available
        type_key = query_type.value
        if type_key in self.type_weights:
            return self.type_weights[type_key]
        
        # Fallback to hardcoded logic
        if query_type == QueryType.FACTUAL:
            return 0.4
        elif query_type == QueryType.ENTITY and entities:
            return 0.3
        elif query_type == QueryType.CONCEPTUAL:
            return 0.8
        elif query_type == QueryType.TEMPORAL:
            return 0.6
        elif query_type == QueryType.COMPARATIVE:
            return 0.7
        else:
            return self.type_weights.get('default', 0.65)


# Example usage and testing
if __name__ == "__main__":
    try:
        from ..config import init_config
    except ImportError:
        from config import init_config
    
    # Initialize configuration and logging
    config = init_config()
    
    # Initialize agent
    agent = QueryAnalysisAgent(config)
    
    # Test queries
    test_queries = [
        "What happened with Chesapeake Energy's acquisition last week?",
        "ESG investing trends in 2024",
        "Compare EU and US energy policies",
        "Israel Lebanon ceasefire violations",
        "Gas prices in Europe yesterday"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        analysis = agent.analyze_query(query)
        
        print(f"Type: {analysis.query_type.value}")
        print(f"Entities: {analysis.entities}")
        print(f"Search Alpha: {analysis.search_alpha}")
        print(f"Temporal: {analysis.temporal_constraints.to_dict()}")
        print(f"\nExpanded Queries:")
        for i, eq in enumerate(analysis.expanded_queries, 1):
            print(f"  {i}. {eq}")
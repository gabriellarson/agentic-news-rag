"""
Timeline Construction Agent for News RAG System

Constructs coherent timelines from extracted events across multiple articles:
- Chronological ordering of events
- Event deduplication and merging
- Causal relationship identification
- Confidence scoring and uncertainty handling
- Timeline validation and consistency checking
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import re

try:
    from ..config import get_config
    from .information_extraction import ExtractedEvent, ExtractedEntity
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    from agents.information_extraction import ExtractedEvent, ExtractedEntity

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """Represents a consolidated event in a timeline"""
    id: str
    description: str
    date: Optional[datetime]
    date_text: str
    estimated_date: Optional[datetime] = None  # Best guess for undated events
    entities: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # Source articles
    confidence: float = 0.0
    supporting_events: List[str] = field(default_factory=list)  # IDs of merged events
    causal_relationships: List[str] = field(default_factory=list)  # Events this caused/was caused by
    event_type: str = "general"  # announcement, transaction, decision, etc.
    importance_score: float = 0.0  # 0-1 importance in timeline


@dataclass
class Timeline:
    """Represents a complete timeline of events"""
    events: List[TimelineEvent]
    topic: str
    date_range: Tuple[Optional[datetime], Optional[datetime]]
    confidence: float
    completeness_score: float  # How complete the timeline appears
    consistency_score: float   # How consistent the events are
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimelineConstructionAgent:
    """Agent for constructing coherent timelines from extracted events"""
    
    def __init__(self, config=None):
        """Initialize the timeline construction agent"""
        if config is None:
            config = get_config()
            
        self.config = config
        self.llm_endpoint = config.get('llm', 'endpoint', default='http://localhost:8001/v1')
        self.llm_model = config.get('llm', 'model', default='qwen3-30b')
        
        # Timeline configuration
        self.max_timeline_events = config.get('timeline', 'max_events', default=50)
        self.min_confidence = config.get('timeline', 'min_confidence', default=0.5)
        self.merge_threshold = config.get('timeline', 'merge_threshold', default=0.8)
        
        # Initialize LLM client
        try:
            from openai import OpenAI
            if OpenAI is None:
                raise ImportError("OpenAI package required")
            
            self.client = OpenAI(
                base_url=self.llm_endpoint,
                api_key="not-needed"
            )
            logger.info(f"Initialized LLM client for timeline construction")
        except ImportError:
            raise ImportError("OpenAI package is required for timeline construction. Install with: pip install openai")
    
    def construct_timeline(self, extracted_events: List[Dict[str, Any]], 
                          topic: str = "General Timeline") -> Timeline:
        """
        Construct a timeline from extracted events.
        
        Args:
            extracted_events: List of extraction results from multiple articles
            topic: Topic/theme of the timeline
            
        Returns:
            Constructed Timeline object
        """
        logger.info(f"Constructing timeline for: {topic}")
        
        # Convert extraction results to ExtractedEvent objects
        all_events = []
        for result in extracted_events:
            for event_data in result.get('events', []):
                event = ExtractedEvent(
                    description=event_data['description'],
                    date=datetime.fromisoformat(event_data['date']) if event_data['date'] else None,
                    date_text=event_data['date_text'],
                    entities=event_data['entities'],
                    confidence=event_data['confidence'],
                    article_id=result['article_id'],
                    source_text=event_data['source_text']
                )
                all_events.append(event)
        
        logger.info(f"Processing {len(all_events)} extracted events")
        
        # Step 1: Filter and validate events
        valid_events = self._filter_events(all_events)
        logger.info(f"Filtered to {len(valid_events)} valid events")
        
        # Step 2: Deduplicate and merge similar events
        merged_events = self._deduplicate_events(valid_events)
        logger.info(f"Merged to {len(merged_events)} unique events")
        
        # Step 3: Estimate dates for undated events
        dated_events = self._estimate_dates(merged_events)
        logger.info(f"Estimated dates for undated events")
        
        # Step 4: Sort chronologically
        sorted_events = self._sort_chronologically(dated_events)
        
        # Step 5: Identify causal relationships
        linked_events = self._identify_relationships(sorted_events)
        
        # Step 6: Score importance and relevance
        scored_events = self._score_importance(linked_events, topic)
        
        # Step 7: Create final timeline
        timeline = self._create_timeline(scored_events, topic)
        
        logger.info(f"Created timeline with {len(timeline.events)} events")
        return timeline
    
    def _filter_events(self, events: List[ExtractedEvent]) -> List[ExtractedEvent]:
        """Filter events by confidence and relevance"""
        filtered = []
        for event in events:
            if event.confidence >= self.min_confidence:
                # Additional filtering criteria
                if len(event.description) > 10:  # Meaningful descriptions
                    filtered.append(event)
                    logger.debug(f"Kept event: {event.description[:50]}... (confidence: {event.confidence})")
        logger.info(f"Filtered {len(events)} events to {len(filtered)} based on confidence threshold {self.min_confidence}")
        return filtered
    
    def _deduplicate_events(self, events: List[ExtractedEvent]) -> List[TimelineEvent]:
        """Deduplicate and merge similar events using LLM"""
        if not events:
            return []
        
        # Group events by similarity
        event_groups = self._group_similar_events(events)
        
        # Merge each group into a single TimelineEvent
        merged_events = []
        for group in event_groups:
            merged_event = self._merge_event_group(group)
            merged_events.append(merged_event)
        
        return merged_events
    
    def _group_similar_events(self, events: List[ExtractedEvent]) -> List[List[ExtractedEvent]]:
        """Group similar events together using LLM-based comparison"""
        if len(events) <= 1:
            return [events]
        
        # Create event descriptions for comparison
        event_descriptions = [
            f"Event {i}: {event.description} (Entities: {', '.join(event.entities)})"
            for i, event in enumerate(events)
        ]
        
        prompt = f"""
        Analyze these {len(events)} news events and group them by similarity. Events should be grouped together if they:
        1. Describe the same fundamental occurrence or announcement
        2. Involve the same key entities and actions
        3. Refer to the same business deal, policy decision, or development
        
        Events:
        {chr(10).join(event_descriptions)}
        
        Return a JSON array of groups, where each group contains the event indices that should be merged:
        [
          [0, 3, 7],  // Events 0, 3, and 7 are the same
          [1],        // Event 1 is unique
          [2, 5],     // Events 2 and 5 are the same
          [4, 6, 8]   // Events 4, 6, and 8 are the same
        ]
        
        Only group events that are clearly about the same occurrence. When in doubt, keep events separate.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            groups_data = self._extract_json_from_text(content, "groups")
            
            # Convert indices to event groups
            groups = []
            for group_indices in groups_data:
                if isinstance(group_indices, list):
                    group = [events[i] for i in group_indices if 0 <= i < len(events)]
                    if group:
                        groups.append(group)
            
            # Ensure all events are included
            used_indices = set()
            for group_indices in groups_data:
                if isinstance(group_indices, list):
                    used_indices.update(group_indices)
            
            # Add ungrouped events as individual groups
            for i, event in enumerate(events):
                if i not in used_indices:
                    groups.append([event])
            
            logger.info(f"Grouped {len(events)} events into {len(groups)} groups")
            return groups
            
        except Exception as e:
            logger.error(f"Failed to group events: {e}")
            # Fallback: each event is its own group
            return [[event] for event in events]
    
    def _merge_event_group(self, events: List[ExtractedEvent]) -> TimelineEvent:
        """Merge a group of similar events into a single TimelineEvent"""
        if len(events) == 1:
            event = events[0]
            return TimelineEvent(
                id=f"timeline_event_{hash(event.description)}",
                description=event.description,
                date=event.date,
                date_text=event.date_text,
                entities=event.entities,
                sources=[event.article_id],
                confidence=event.confidence,
                supporting_events=[],
                event_type=self._classify_event_type(event.description)
            )
        
        # Merge multiple events using LLM
        event_details = []
        for i, event in enumerate(events):
            details = f"Source {i+1}: {event.description}"
            if event.date_text and event.date_text != "not specified":
                details += f" (Date: {event.date_text})"
            if event.entities:
                details += f" (Entities: {', '.join(event.entities)})"
            event_details.append(details)
        
        prompt = f"""
        Merge these {len(events)} similar news events into a single, comprehensive event description.
        Create a unified description that captures the key information from all sources.
        
        Events to merge:
        {chr(10).join(event_details)}
        
        Return JSON with this structure:
        {{
          "merged_description": "Comprehensive description combining all sources",
          "primary_date_text": "Best date reference from the sources",
          "all_entities": ["entity1", "entity2", "entity3"],
          "event_type": "announcement|transaction|decision|regulatory|market_action|other",
          "confidence": 0.85
        }}
        
        The merged description should be clear, comprehensive, and eliminate redundancy.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            merge_data = self._extract_json_from_text(content, "merge")
            
            if not merge_data or not isinstance(merge_data, dict):
                logger.warning(f"Failed to extract merge data, using fallback")
                raise ValueError("No valid JSON found in merge response")
            
            # Find best date from the group
            best_date = None
            best_date_text = merge_data.get('primary_date_text', 'not specified')
            
            for event in events:
                if event.date:
                    best_date = event.date
                    break
            
            # Combine all entities
            all_entities = list(set(merge_data.get('all_entities', [])))
            
            # Combine sources
            sources = list(set(event.article_id for event in events))
            
            # Average confidence
            avg_confidence = sum(event.confidence for event in events) / len(events)
            merged_confidence = min(merge_data.get('confidence', avg_confidence), avg_confidence)
            
            return TimelineEvent(
                id=f"timeline_event_{hash(merge_data['merged_description'])}",
                description=merge_data['merged_description'],
                date=best_date,
                date_text=best_date_text,
                entities=all_entities,
                sources=sources,
                confidence=merged_confidence,
                supporting_events=[],
                event_type=merge_data.get('event_type', 'general')
            )
            
        except Exception as e:
            logger.error(f"Failed to merge events: {e}")
            # Fallback: use the first event as primary
            primary = events[0]
            return TimelineEvent(
                id=f"timeline_event_{hash(primary.description)}",
                description=primary.description,
                date=primary.date,
                date_text=primary.date_text,
                entities=list(set(entity for event in events for entity in event.entities)),
                sources=list(set(event.article_id for event in events)),
                confidence=sum(event.confidence for event in events) / len(events),
                supporting_events=[],
                event_type=self._classify_event_type(primary.description)
            )
    
    def _classify_event_type(self, description: str) -> str:
        """Classify event type based on description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['announced', 'announces', 'unveiled', 'revealed']):
            return 'announcement'
        elif any(word in description_lower for word in ['acquired', 'merger', 'deal', 'purchased', 'bought']):
            return 'transaction'
        elif any(word in description_lower for word in ['decided', 'approved', 'voted', 'ruled']):
            return 'decision'
        elif any(word in description_lower for word in ['regulation', 'policy', 'law', 'rule']):
            return 'regulatory'
        elif any(word in description_lower for word in ['market', 'trading', 'price', 'exchange']):
            return 'market_action'
        else:
            return 'general'
    
    def _estimate_dates(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Estimate dates for undated events using context and LLM"""
        dated_events = [e for e in events if e.date]
        undated_events = [e for e in events if not e.date]
        
        if not undated_events or not dated_events:
            return events
        
        # Use LLM to estimate dates based on context
        for undated_event in undated_events:
            estimated_date = self._estimate_event_date(undated_event, dated_events)
            if estimated_date:
                undated_event.estimated_date = estimated_date
        
        return events
    
    def _estimate_event_date(self, undated_event: TimelineEvent, 
                           dated_events: List[TimelineEvent]) -> Optional[datetime]:
        """Estimate date for a single undated event"""
        # Create context from dated events
        context_events = []
        for event in dated_events[:10]:  # Use top 10 dated events for context
            context_events.append(f"- {event.description} ({event.date.strftime('%Y-%m-%d')})")
        
        if not context_events:
            return None
        
        prompt = f"""
        Given these dated events as context, estimate the most likely date for the undated event.
        
        Context (dated events):
        {chr(10).join(context_events)}
        
        Undated event: {undated_event.description}
        Date reference: {undated_event.date_text}
        
        Based on the context and any temporal clues in the undated event, provide your best estimate.
        
        Return JSON:
        {{
          "estimated_date": "YYYY-MM-DD or null if no reasonable estimate possible",
          "confidence": 0.7,
          "reasoning": "Brief explanation of the estimate"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            estimate_data = self._extract_json_from_text(content, "estimate")[0]
            
            if estimate_data.get('estimated_date'):
                try:
                    return datetime.fromisoformat(estimate_data['estimated_date'])
                except ValueError:
                    return None
            
        except Exception as e:
            logger.warning(f"Failed to estimate date for event: {e}")
        
        return None
    
    def _sort_chronologically(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Sort events chronologically"""
        def get_sort_date(event):
            if event.date:
                return event.date
            elif event.estimated_date:
                return event.estimated_date
            else:
                return datetime.max  # Put undated events at the end
        
        return sorted(events, key=get_sort_date)
    
    def _identify_relationships(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Identify causal relationships between events"""
        if len(events) < 2:
            return events
        
        # Use LLM to identify relationships
        event_summaries = []
        for i, event in enumerate(events):
            date_str = event.date.strftime('%Y-%m-%d') if event.date else 'undated'
            event_summaries.append(f"{i}: {event.description} ({date_str})")
        
        prompt = f"""
        Analyze these chronologically ordered events and identify causal relationships.
        Look for events that directly caused or led to other events.
        
        Events:
        {chr(10).join(event_summaries)}
        
        Return JSON array of relationships:
        [
          {{
            "cause_event_id": 0,
            "effect_event_id": 3,
            "relationship_type": "direct_cause|contributing_factor|reaction",
            "confidence": 0.8
          }}
        ]
        
        Only include relationships with high confidence. Focus on clear causal connections.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10000,  # Increased for complex relationship analysis
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            relationships = self._extract_json_from_text(content, "relationships")
            
            # Apply relationships to events
            for rel in relationships:
                try:
                    cause_idx = rel['cause_event_id']
                    effect_idx = rel['effect_event_id']
                    
                    if 0 <= cause_idx < len(events) and 0 <= effect_idx < len(events):
                        events[cause_idx].causal_relationships.append(events[effect_idx].id)
                        logger.debug(f"Added causal relationship: {cause_idx} -> {effect_idx}")
                except (KeyError, IndexError, TypeError):
                    continue
            
        except Exception as e:
            logger.warning(f"Failed to identify relationships: {e}")
        
        return events
    
    def _score_importance(self, events: List[TimelineEvent], topic: str) -> List[TimelineEvent]:
        """Score event importance relative to the timeline topic"""
        if not events:
            return events
        
        event_descriptions = [f"{i}: {event.description}" for i, event in enumerate(events)]
        
        prompt = f"""
        Score the importance of each event for a timeline about "{topic}".
        Consider:
        1. Direct relevance to the topic - events not related to "{topic}" should score very low (0.0-0.2)
        2. Impact and significance within the topic domain
        3. Whether it's a major milestone or minor detail
        
        For example, if the topic is "Energy Sector Mergers", only score energy/gas/oil company mergers, acquisitions, 
        and directly related market developments as important. Unrelated political events, other sector news, etc. should score very low.
        
        Events:
        {chr(10).join(event_descriptions)}
        
        Return JSON array of scores:
        [
          {{"event_id": 0, "importance_score": 0.9, "relevance_note": "Directly about energy merger"}},
          {{"event_id": 1, "importance_score": 0.1, "relevance_note": "Unrelated to energy mergers"}},
          ...
        ]
        
        Scores should be 0.0-1.0, where 1.0 is most important and highly relevant to the topic.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                temperature=0.1,
                n=1
            )
            
            content = response.choices[0].message.content
            scores = self._extract_json_from_text(content, "scores")
            
            # Apply scores
            for score_data in scores:
                try:
                    idx = score_data['event_id']
                    score = score_data['importance_score']
                    if 0 <= idx < len(events) and 0 <= score <= 1:
                        events[idx].importance_score = score
                except (KeyError, IndexError, TypeError):
                    continue
            
        except Exception as e:
            logger.warning(f"Failed to score importance: {e}")
            # Fallback: assign default scores
            for event in events:
                event.importance_score = 0.5
        
        return events
    
    def _create_timeline(self, events: List[TimelineEvent], topic: str) -> Timeline:
        """Create the final Timeline object"""
        # Filter out low-importance events (threshold of 0.3)
        importance_threshold = 0.3
        relevant_events = [e for e in events if e.importance_score >= importance_threshold]
        
        if relevant_events:
            logger.info(f"Filtered {len(events)} events to {len(relevant_events)} based on importance threshold {importance_threshold}")
            events = relevant_events
        
        if not events:
            return Timeline(
                events=[],
                topic=topic,
                date_range=(None, None),
                confidence=0.0,
                completeness_score=0.0,
                consistency_score=0.0
            )
        
        # Calculate date range
        dated_events = [e for e in events if e.date or e.estimated_date]
        if dated_events:
            dates = [e.date or e.estimated_date for e in dated_events]
            date_range = (min(dates), max(dates))
        else:
            date_range = (None, None)
        
        # Calculate timeline confidence
        avg_confidence = sum(event.confidence for event in events) / len(events)
        
        # Calculate completeness and consistency scores
        completeness = self._calculate_completeness(events)
        consistency = self._calculate_consistency(events)
        
        return Timeline(
            events=events,
            topic=topic,
            date_range=date_range,
            confidence=avg_confidence,
            completeness_score=completeness,
            consistency_score=consistency,
            metadata={
                'total_events': len(events),
                'dated_events': len([e for e in events if e.date]),
                'estimated_dates': len([e for e in events if e.estimated_date]),
                'causal_relationships': sum(len(e.causal_relationships) for e in events)
            }
        )
    
    def _calculate_completeness(self, events: List[TimelineEvent]) -> float:
        """Calculate how complete the timeline appears"""
        if not events:
            return 0.0
        
        # Factor in date coverage, entity consistency, and event density
        dated_ratio = len([e for e in events if e.date or e.estimated_date]) / len(events)
        
        # Simple completeness metric
        return min(1.0, dated_ratio * 1.2)
    
    def _calculate_consistency(self, events: List[TimelineEvent]) -> float:
        """Calculate how consistent the timeline is"""
        if len(events) < 2:
            return 1.0
        
        # Check for chronological consistency
        dated_events = [(e.date or e.estimated_date, e) for e in events if e.date or e.estimated_date]
        if len(dated_events) < 2:
            return 0.8
        
        # Simple consistency check
        return 0.9  # Placeholder - could be enhanced with more sophisticated checks
    
    def _extract_json_from_text(self, text: str, data_type: str) -> Any:
        """Extract JSON from LLM response"""
        import re
        
        # First try to parse the entire response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object for merge operations
        if data_type == "merge":
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            logger.warning(f"No JSON found in {data_type} response: {text[:200]}...")
            return [] if data_type != "merge" else {}
        
        json_str = json_match.group(0)
        
        # Clean common JSON formatting issues
        fixes = [
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            (r'\\n', ' '),
            (r'\\"', '"'),
            (r'"\s*\n\s*"', '"'),
        ]
        
        for pattern, replacement in fixes:
            json_str = re.sub(pattern, replacement, json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse {data_type} JSON after cleaning: {e}")
            return [] if data_type != "merge" else {}


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import init_config
    
    # Test timeline construction
    config = init_config()
    agent = TimelineConstructionAgent(config)
    
    # Mock extraction results for testing
    sample_extractions = [
        {
            'article_id': 'article1.txt',
            'events': [
                {
                    'description': 'Chesapeake Energy agreed to acquire Southwestern Energy in a $7.4bn deal',
                    'date': '2024-01-11T12:42:05.955000+00:00',
                    'date_text': 'January 11, 2024',
                    'entities': ['Chesapeake Energy', 'Southwestern Energy'],
                    'confidence': 0.9,
                    'source_text': 'Chesapeake Energy has agreed to buy...'
                }
            ]
        }
    ]
    
    timeline = agent.construct_timeline(sample_extractions, "Energy Sector Mergers")
    
    print(f"Timeline: {timeline.topic}")
    print(f"Events: {len(timeline.events)}")
    print(f"Confidence: {timeline.confidence:.2f}")
    print(f"Completeness: {timeline.completeness_score:.2f}")
"""
Report Generation Agent for News RAG System

Synthesizes timelines into coherent reports that answer user queries:
- Generates narrative from chronological events
- Manages citations and source attribution
- Creates executive summaries and key findings
- Handles conflicting information and uncertainty
- Formats output in multiple styles
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import re

try:
    from ..config import get_config
    from ..embeddings.article_parser import Article
    from .timeline_construction import Timeline, TimelineEvent
    from ..llm.interface import LLMInterface
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    from embeddings.article_parser import Article
    from agents.timeline_construction import Timeline, TimelineEvent
    from llm.interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation for source attribution"""
    article_id: str
    article_title: str
    published_date: datetime
    authors: List[str]
    reference_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'article_id': self.article_id,
            'article_title': self.article_title,
            'published_date': self.published_date.isoformat(),
            'authors': self.authors,
            'reference_number': self.reference_number
        }


@dataclass
class ReportSection:
    """Represents a section of the report"""
    title: str
    content: str
    citations: List[int] = field(default_factory=list)  # Reference numbers


@dataclass
class GeneratedReport:
    """Represents a complete generated report"""
    title: str
    executive_summary: str
    key_findings: List[str]
    sections: List[ReportSection]
    citations: List[Citation]
    metadata: Dict[str, Any]
    confidence_score: float
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'executive_summary': self.executive_summary,
            'key_findings': self.key_findings,
            'sections': [{'title': s.title, 'content': s.content, 'citations': s.citations} 
                        for s in self.sections],
            'citations': [c.to_dict() for c in self.citations],
            'metadata': self.metadata,
            'confidence_score': self.confidence_score,
            'generated_at': self.generated_at.isoformat()
        }


class ReportGenerationAgent:
    """Agent for generating reports from timelines"""
    
    def __init__(self, config=None):
        """Initialize the report generation agent"""
        if config is None:
            config = get_config()
            
        self.config = config
        
        # Report configuration
        self.max_report_length = config.get('reporting', 'max_report_length', default=2000)
        self.include_citations = config.get('reporting', 'include_citations', default=True)
        self.include_confidence = config.get('reporting', 'include_confidence_scores', default=True)
        self.executive_summary_length = config.get('reporting', 'executive_summary_length', default=200)
        self.min_event_confidence = config.get('reporting', 'min_event_confidence', default=0.5)
        self.citation_style = config.get('reporting', 'citation_style', default='footnote')
        
        # Initialize LLM interface
        self.llm = LLMInterface(config)
        logger.info(f"Initialized report generation agent")
    
    def generate_report(self, timeline: Timeline, query: str, 
                       articles: List[Article] = None) -> GeneratedReport:
        """
        Generate a complete report from a timeline.
        
        Args:
            timeline: Constructed timeline of events
            query: Original user query
            articles: Source articles for citation details
            
        Returns:
            GeneratedReport object
        """
        logger.info(f"Generating report for query: {query}")
        
        # Filter events by confidence if needed
        filtered_events = [e for e in timeline.events 
                          if e.confidence >= self.min_event_confidence]
        
        if not filtered_events:
            logger.warning("No events meet confidence threshold")
            return self._generate_empty_report(query, timeline.topic)
        
        # Create citations mapping
        citations = self._create_citations(timeline, articles)
        citation_map = self._build_citation_map(timeline, citations)
        
        # Generate report structure
        structure = self._determine_report_structure(filtered_events, query, timeline.topic)
        
        # Generate content for each section
        sections = self._generate_sections(structure, filtered_events, citation_map)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            sections, timeline, query
        )
        
        # Extract key findings
        key_findings = self._extract_key_findings(timeline, query, sections)
        
        # Calculate confidence score
        confidence = self._calculate_report_confidence(timeline, filtered_events)
        
        # Create metadata
        metadata = {
            'timeline_events': len(timeline.events),
            'included_events': len(filtered_events),
            'date_range': {
                'start': timeline.date_range[0].isoformat() if timeline.date_range[0] else None,
                'end': timeline.date_range[1].isoformat() if timeline.date_range[1] else None
            },
            'timeline_completeness': timeline.completeness_score,
            'timeline_consistency': timeline.consistency_score,
            'sources': len(set(e.sources[0] for e in filtered_events if e.sources))
        }
        
        report = GeneratedReport(
            title=f"Report: {timeline.topic}",
            executive_summary=executive_summary,
            key_findings=key_findings,
            sections=sections,
            citations=list(citations.values()),
            metadata=metadata,
            confidence_score=confidence
        )
        
        logger.info(f"Generated report with {len(sections)} sections and {len(citations)} citations")
        return report
    
    def _generate_empty_report(self, query: str, topic: str) -> GeneratedReport:
        """Generate a report when no qualifying events are found"""
        return GeneratedReport(
            title=f"Report: {topic}",
            executive_summary="No events meeting the confidence threshold were found for this query.",
            key_findings=[],
            sections=[],
            citations=[],
            metadata={'error': 'No qualifying events'},
            confidence_score=0.0
        )
    
    def _create_citations(self, timeline: Timeline, 
                         articles: List[Article] = None) -> Dict[str, Citation]:
        """Create citations from timeline events and articles"""
        citations = {}
        ref_num = 1
        
        # Get unique source articles
        source_ids = set()
        for event in timeline.events:
            source_ids.update(event.sources)
        
        # Create citations for each source
        for source_id in sorted(source_ids):
            # Try to find article details
            article = None
            if articles:
                article = next((a for a in articles if str(a.file_path) == source_id), None)
            
            if article:
                citation = Citation(
                    article_id=source_id,
                    article_title=article.title,
                    published_date=article.published,
                    authors=article.authors,
                    reference_number=ref_num
                )
            else:
                # Create citation with just ID if article not found
                citation = Citation(
                    article_id=source_id,
                    article_title=source_id,
                    published_date=datetime.now(),
                    authors=[],
                    reference_number=ref_num
                )
            
            citations[source_id] = citation
            ref_num += 1
        
        return citations
    
    def _build_citation_map(self, timeline: Timeline, 
                           citations: Dict[str, Citation]) -> Dict[str, List[int]]:
        """Build mapping from event IDs to citation reference numbers"""
        citation_map = {}
        
        for event in timeline.events:
            event_citations = []
            for source in event.sources:
                if source in citations:
                    event_citations.append(citations[source].reference_number)
            citation_map[event.id] = sorted(event_citations)
        
        return citation_map
    
    def _determine_report_structure(self, events: List[TimelineEvent], 
                                   query: str, topic: str) -> List[Dict[str, Any]]:
        """Determine the structure of the report based on events and query"""
        # For now, use chronological structure
        # Could be enhanced to use thematic grouping based on event types
        
        structure = []
        
        # Group events by time period
        if events:
            # Group by month/year
            grouped_events = defaultdict(list)
            for event in events:
                if event.date or event.estimated_date:
                    date = event.date or event.estimated_date
                    period = date.strftime("%B %Y")
                    grouped_events[period].append(event)
                else:
                    grouped_events["Undated Events"].append(event)
            
            # Create sections for each period
            for period in sorted(grouped_events.keys()):
                if period != "Undated Events":
                    structure.append({
                        'title': period,
                        'events': grouped_events[period],
                        'type': 'chronological'
                    })
            
            # Add undated events at the end if any
            if "Undated Events" in grouped_events:
                structure.append({
                    'title': "Additional Context",
                    'events': grouped_events["Undated Events"],
                    'type': 'contextual'
                })
        
        return structure
    
    def _generate_sections(self, structure: List[Dict[str, Any]], 
                          events: List[TimelineEvent],
                          citation_map: Dict[str, List[int]]) -> List[ReportSection]:
        """Generate content for each report section"""
        sections = []
        
        for section_info in structure:
            section_events = section_info['events']
            if not section_events:
                continue
            
            # Generate narrative for this section
            narrative = self._generate_section_narrative(
                section_events, 
                section_info['title'],
                section_info['type']
            )
            
            # Collect citations for this section
            section_citations = []
            for event in section_events:
                section_citations.extend(citation_map.get(event.id, []))
            section_citations = sorted(set(section_citations))
            
            section = ReportSection(
                title=section_info['title'],
                content=narrative,
                citations=section_citations
            )
            sections.append(section)
        
        return sections
    
    def _generate_section_narrative(self, events: List[TimelineEvent], 
                                   section_title: str, section_type: str) -> str:
        """Generate narrative content for a section using LLM"""
        # Prepare event summaries
        event_summaries = []
        for event in events:
            date_str = ""
            if event.date:
                date_str = event.date.strftime("%B %d, %Y")
            elif event.estimated_date:
                date_str = f"approximately {event.estimated_date.strftime('%B %Y')}"
            
            summary = f"- {event.description}"
            if date_str:
                summary = f"- {date_str}: {event.description}"
            if event.entities:
                summary += f" (Key entities: {', '.join(event.entities[:3])})"
            
            event_summaries.append(summary)
        
        prompt = f"""
        Generate a coherent narrative paragraph for this section of a news analysis report.
        
        Section: {section_title}
        Type: {section_type}
        
        Events to include:
        {chr(10).join(event_summaries)}
        
        Guidelines:
        - Create a flowing narrative that connects these events naturally
        - Maintain chronological order within the narrative
        - Use appropriate transition words between events
        - Keep factual accuracy - do not add information not in the events
        - Write in a professional, analytical tone
        - Aim for 150-300 words
        - If events show cause and effect, highlight these relationships
        
        Write the narrative paragraph:
        """
        
        try:
            narrative = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
                output_type="text"
            )
            
            # Add citation markers if needed
            if self.citation_style == 'inline':
                # Could enhance to add inline citations
                pass
            
            return narrative
            
        except Exception as e:
            logger.error(f"Failed to generate section narrative: {e}")
            # Fallback to simple event listing
            fallback = f"During {section_title}, the following events occurred:\n\n"
            fallback += "\n\n".join(event_summaries)
            return fallback
    
    def _generate_executive_summary(self, sections: List[ReportSection], 
                                   timeline: Timeline, query: str) -> str:
        """Generate executive summary of the report"""
        if not sections:
            return "No significant events were found for the specified query."
        
        # Combine section narratives for context
        full_narrative = "\n\n".join([s.content for s in sections[:3]])  # Use first 3 sections
        
        prompt = f"""
        Generate a concise executive summary for this report.
        
        Original Query: "{query}"
        Report Topic: {timeline.topic}
        
        Key sections covered:
        {chr(10).join([f"- {s.title}" for s in sections])}
        
        Sample content:
        {full_narrative[:1000]}...
        
        Guidelines:
        - Directly answer the user's query in the first sentence
        - Summarize the most important findings in 100-150 words
        - Highlight any significant trends or patterns
        - Mention the time period covered
        - Be specific with key facts (companies, amounts, dates)
        - Write in past tense for completed events
        
        Executive Summary:
        """
        
        try:
            summary = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000,
                output_type="text"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            # Fallback summary
            return (f"This report analyzes {timeline.topic} based on {len(timeline.events)} "
                   f"events from multiple sources. The timeline covers events "
                   f"from {timeline.date_range[0].strftime('%B %Y') if timeline.date_range[0] else 'various dates'}.")
    
    def _extract_key_findings(self, timeline: Timeline, query: str, 
                             sections: List[ReportSection]) -> List[str]:
        """Extract key findings from the timeline and report"""
        # Get high-importance events
        important_events = sorted(
            [e for e in timeline.events if e.importance_score >= 0.7],
            key=lambda x: x.importance_score,
            reverse=True
        )[:5]  # Top 5 most important events
        
        if not important_events:
            return []
        
        # Prepare event descriptions
        event_descriptions = []
        for event in important_events:
            desc = event.description
            if event.entities:
                desc += f" (involving {', '.join(event.entities[:2])})"
            event_descriptions.append(desc)
        
        prompt = f"""
        Extract 3-5 key findings from these important events for the query: "{query}"
        
        Important events:
        {chr(10).join([f"- {desc}" for desc in event_descriptions])}
        
        Guidelines:
        - Each finding should be a complete, standalone insight
        - Focus on answering the user's query
        - Include specific details (numbers, dates, entities)
        - Order by relevance to the query
        - Each finding should be 1-2 sentences
        
        Return as a JSON array of strings:
        ["Finding 1", "Finding 2", "Finding 3"]
        """
        
        try:
            findings = self.llm.prompt_llm(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1000,
                output_type="json_array"
            )
            
            if isinstance(findings, list):
                return [str(f) for f in findings if isinstance(f, str)][:5]
            
        except Exception as e:
            logger.error(f"Failed to extract key findings: {e}")
        
        # Fallback to simple extraction
        findings = []
        for event in important_events[:3]:
            finding = event.description
            if event.date:
                finding = f"On {event.date.strftime('%B %d, %Y')}, {finding}"
            findings.append(finding)
        
        return findings
    
    def _calculate_report_confidence(self, timeline: Timeline, 
                                    included_events: List[TimelineEvent]) -> float:
        """Calculate overall confidence score for the report"""
        if not included_events:
            return 0.0
        
        # Factors to consider:
        # 1. Average event confidence
        avg_event_confidence = sum(e.confidence for e in included_events) / len(included_events)
        
        # 2. Timeline completeness
        timeline_completeness = timeline.completeness_score
        
        # 3. Source diversity (more sources = higher confidence)
        unique_sources = len(set(source for e in included_events for source in e.sources))
        source_diversity = min(1.0, unique_sources / 3.0)  # Normalize to max of 3 sources
        
        # 4. Date coverage (events with dates vs estimated/undated)
        dated_ratio = len([e for e in included_events if e.date]) / len(included_events)
        
        # Weighted average
        confidence = (
            avg_event_confidence * 0.4 +
            timeline_completeness * 0.3 +
            source_diversity * 0.2 +
            dated_ratio * 0.1
        )
        
        return round(confidence, 2)
    
    def format_report(self, report: GeneratedReport, format: str = "markdown") -> str:
        """
        Format the report for output.
        
        Args:
            report: Generated report object
            format: Output format ('markdown', 'text', 'json')
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return json.dumps(report.to_dict(), indent=2)
        elif format == "markdown":
            return self._format_markdown(report)
        else:  # text
            return self._format_text(report)
    
    def _format_markdown(self, report: GeneratedReport) -> str:
        """Format report as markdown"""
        lines = []
        
        # Title
        lines.append(f"# {report.title}")
        lines.append("")
        
        # Metadata
        if self.include_confidence:
            lines.append(f"*Report Confidence: {report.confidence_score:.1%}*")
            lines.append(f"*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}*")
            lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(report.executive_summary)
        lines.append("")
        
        # Key Findings
        if report.key_findings:
            lines.append("## Key Findings")
            lines.append("")
            for finding in report.key_findings:
                lines.append(f"- {finding}")
            lines.append("")
        
        # Main Content Sections
        if report.sections:
            lines.append("## Timeline of Events")
            lines.append("")
            
            for section in report.sections:
                lines.append(f"### {section.title}")
                lines.append("")
                
                # Add content
                content = section.content
                
                # Add citation markers if using footnote style
                if self.include_citations and self.citation_style == "footnote" and section.citations:
                    citation_refs = ", ".join([f"[{ref}]" for ref in section.citations])
                    content += f" {citation_refs}"
                
                lines.append(content)
                lines.append("")
        
        # Citations
        if self.include_citations and report.citations:
            lines.append("## Sources")
            lines.append("")
            
            for citation in sorted(report.citations, key=lambda x: x.reference_number):
                authors_str = ", ".join(citation.authors) if citation.authors else "Unknown"
                date_str = citation.published_date.strftime("%Y-%m-%d")
                lines.append(f"[{citation.reference_number}] {citation.article_title} - "
                           f"{authors_str} ({date_str})")
            lines.append("")
        
        # Metadata footer
        if report.metadata:
            lines.append("---")
            lines.append("*Report Metadata:*")
            lines.append(f"- Events analyzed: {report.metadata.get('timeline_events', 0)}")
            lines.append(f"- Events included: {report.metadata.get('included_events', 0)}")
            lines.append(f"- Sources: {report.metadata.get('sources', 0)}")
            if report.metadata.get('date_range', {}).get('start'):
                lines.append(f"- Date range: {report.metadata['date_range']['start']} to "
                           f"{report.metadata['date_range']['end']}")
        
        return "\n".join(lines)
    
    def _format_text(self, report: GeneratedReport) -> str:
        """Format report as plain text"""
        # Similar to markdown but without formatting symbols
        lines = []
        
        # Title
        lines.append(report.title.upper())
        lines.append("=" * len(report.title))
        lines.append("")
        
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 17)
        lines.append(report.executive_summary)
        lines.append("")
        
        # Key Findings
        if report.key_findings:
            lines.append("KEY FINDINGS")
            lines.append("-" * 12)
            for i, finding in enumerate(report.key_findings, 1):
                lines.append(f"{i}. {finding}")
            lines.append("")
        
        # Sections
        for section in report.sections:
            lines.append(section.title.upper())
            lines.append("-" * len(section.title))
            lines.append(section.content)
            if self.include_citations and section.citations:
                lines.append(f"[References: {', '.join(map(str, section.citations))}]")
            lines.append("")
        
        # Citations
        if self.include_citations and report.citations:
            lines.append("SOURCES")
            lines.append("-" * 7)
            for citation in sorted(report.citations, key=lambda x: x.reference_number):
                lines.append(f"{citation.reference_number}. {citation.article_title}")
        
        return "\n".join(lines)
    


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import init_config
    from embeddings.article_parser import ArticleParser
    from agents.timeline_construction import Timeline, TimelineEvent
    
    # Initialize components
    config = init_config()
    report_agent = ReportGenerationAgent(config)
    
    # Create sample timeline for testing
    sample_events = [
        TimelineEvent(
            id="event1",
            description="Chesapeake Energy announced agreement to acquire Southwestern Energy for $7.4 billion",
            date=datetime(2024, 1, 11),
            date_text="January 11, 2024",
            entities=["Chesapeake Energy", "Southwestern Energy"],
            sources=["article1.txt"],
            confidence=0.9,
            importance_score=0.95,
            event_type="transaction"
        ),
        TimelineEvent(
            id="event2",
            description="The merger will create one of the largest natural gas producers in the United States",
            date=datetime(2024, 1, 11),
            date_text="January 11, 2024",
            entities=["Chesapeake Energy", "Southwestern Energy", "United States"],
            sources=["article1.txt", "article2.txt"],
            confidence=0.85,
            importance_score=0.8,
            event_type="announcement"
        )
    ]
    
    sample_timeline = Timeline(
        events=sample_events,
        topic="Energy Sector Mergers",
        date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
        confidence=0.87,
        completeness_score=0.9,
        consistency_score=0.95,
        metadata={'total_events': 2}
    )
    
    # Generate report
    query = "What major energy sector mergers occurred in early 2024?"
    report = report_agent.generate_report(sample_timeline, query)
    
    # Format and print report
    formatted_report = report_agent.format_report(report, "markdown")
    print(formatted_report)
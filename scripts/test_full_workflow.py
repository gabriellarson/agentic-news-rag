#!/usr/bin/env python3
"""
End-to-End Workflow Test for Agentic News RAG System

Tests the complete pipeline:
1. Query Analysis
2. Search (using indexed articles)
3. Information Extraction
4. Timeline Construction
5. Report Generation

Requires:
- LLM server running at localhost:8001
- Qdrant server running at localhost:6333
- Articles indexed in Qdrant (run index_articles.py first)
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import init_config
from agents.query_analysis import QueryAnalysisAgent
from agents.information_extraction import InformationExtractionAgent
from agents.timeline_construction import TimelineConstructionAgent
from agents.report_generation import ReportGenerationAgent
from embeddings.article_parser import ArticleParser
from embeddings.qdrant_search import QdrantHybridSearch


class WorkflowTester:
    """Test harness for the complete RAG workflow"""
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing Agentic News RAG System...")
        self.config = init_config()
        
        # Initialize agents
        print("  - Query Analysis Agent")
        self.query_agent = QueryAnalysisAgent(self.config)
        
        print("  - Search Engine")
        self.search_engine = QdrantHybridSearch(self.config)
        
        print("  - Information Extraction Agent")
        self.extraction_agent = InformationExtractionAgent(self.config)
        
        print("  - Timeline Construction Agent")
        self.timeline_agent = TimelineConstructionAgent(self.config)
        
        print("  - Report Generation Agent")
        self.report_agent = ReportGenerationAgent(self.config)
        
        # Article parser
        self.article_parser = ArticleParser()
        
        # Performance metrics
        self.metrics = {}
        
        print("System initialized successfully!\n")
    
    def test_workflow(self, query: str, expected_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Test the complete workflow with a query.
        
        Args:
            query: User query to process
            expected_results: Optional expected results for validation
            
        Returns:
            Dictionary with results and metrics
        """
        print(f"\n{'='*60}")
        print(f"TESTING WORKFLOW: {query}")
        print(f"{'='*60}")
        
        results = {}
        total_start = time.time()
        
        try:
            # Step 1: Query Analysis
            print("\n1. QUERY ANALYSIS")
            print("-" * 20)
            qa_start = time.time()
            
            query_analysis = self.query_agent.analyze_query(query)
            
            qa_time = time.time() - qa_start
            print(f"  ✓ Query Type: {query_analysis.query_type.value}")
            print(f"  ✓ Entities: {query_analysis.entities}")
            print(f"  ✓ Expanded Queries: {len(query_analysis.expanded_queries)}")
            print(f"  ✓ Time: {qa_time:.2f}s")
            
            results['query_analysis'] = {
                'type': query_analysis.query_type.value,
                'entities': query_analysis.entities,
                'expanded_queries': query_analysis.expanded_queries,
                'time': qa_time
            }
            
            # Step 2: Search
            print("\n2. SEARCH & RETRIEVAL")
            print("-" * 20)
            search_start = time.time()
            
            # Search with primary query and expanded queries
            all_results = []
            for search_query in query_analysis.expanded_queries[:3]:  # Use top 3 expanded queries
                try:
                    search_results = self.search_engine.search(
                        query=search_query,
                        limit=5  # Get top 5 per query
                    )
                    all_results.extend(search_results)
                except Exception as e:
                    print(f"  ⚠️  Search failed for '{search_query}': {e}")
            
            # Deduplicate results by article path
            seen_paths = set()
            unique_results = []
            for result in all_results:
                path = result.get('file_path', '')
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    unique_results.append(result)
            
            search_time = time.time() - search_start
            print(f"  ✓ Found {len(unique_results)} unique articles")
            print(f"  ✓ Time: {search_time:.2f}s")
            
            if unique_results:
                print("\n  Top Results:")
                for i, result in enumerate(unique_results[:3], 1):
                    title = result.get('title', 'Unknown')
                    score = result.get('score', 0)
                    print(f"    {i}. {title[:60]}... (score: {score:.3f})")
            
            results['search'] = {
                'total_results': len(unique_results),
                'time': search_time,
                'top_scores': [r.get('score', 0) for r in unique_results[:3]]
            }
            
            # Step 3: Parse Articles
            print("\n3. ARTICLE PARSING")
            print("-" * 20)
            parse_start = time.time()
            
            articles = []
            for result in unique_results[:10]:  # Process top 10 articles
                file_path = result.get('file_path', '')
                if file_path:
                    try:
                        article_path = Path("text_articles") / Path(file_path).name
                        if article_path.exists():
                            article = self.article_parser.parse_file(article_path)
                            articles.append(article)
                    except Exception as e:
                        print(f"  ⚠️  Failed to parse {file_path}: {e}")
            
            parse_time = time.time() - parse_start
            print(f"  ✓ Parsed {len(articles)} articles")
            print(f"  ✓ Time: {parse_time:.2f}s")
            
            results['parsing'] = {
                'articles_parsed': len(articles),
                'time': parse_time
            }
            
            if not articles:
                print("\n❌ No articles could be parsed. Aborting workflow.")
                return results
            
            # Step 4: Information Extraction
            print("\n4. INFORMATION EXTRACTION")
            print("-" * 20)
            extract_start = time.time()
            
            extraction_results = self.extraction_agent.extract_from_articles_with_cache(articles, self.search_engine)
            
            # Count total extracted items
            total_events = sum(len(r.get('events', [])) for r in extraction_results)
            total_entities = sum(len(r.get('entities', [])) for r in extraction_results)
            
            extract_time = time.time() - extract_start
            print(f"  ✓ Extracted {total_events} events")
            print(f"  ✓ Extracted {total_entities} entities")
            print(f"  ✓ Time: {extract_time:.2f}s")
            
            # Show sample events
            if total_events > 0:
                print("\n  Sample Events:")
                event_count = 0
                for result in extraction_results:
                    for event in result.get('events', []):
                        event_count += 1
                        print(f"    {event_count}. {event['description'][:80]}...")
                        if event.get('date'):
                            print(f"       Date: {event['date_text']} → {event['date']}")
                        if event_count >= 3:
                            break
                    if event_count >= 3:
                        break
            
            results['extraction'] = {
                'events': total_events,
                'entities': total_entities,
                'articles_processed': len(extraction_results),
                'time': extract_time
            }
            
            # Step 5: Timeline Construction
            print("\n5. TIMELINE CONSTRUCTION")
            print("-" * 20)
            timeline_start = time.time()
            
            timeline = self.timeline_agent.construct_timeline(
                extraction_results,
                topic=query  # Use query as topic
            )
            
            timeline_time = time.time() - timeline_start
            print(f"  ✓ Timeline Events: {len(timeline.events)}")
            print(f"  ✓ Date Range: {timeline.date_range[0]} to {timeline.date_range[1]}")
            print(f"  ✓ Completeness: {timeline.completeness_score:.2f}")
            print(f"  ✓ Consistency: {timeline.consistency_score:.2f}")
            print(f"  ✓ Time: {timeline_time:.2f}s")
            
            # Show timeline preview
            if timeline.events:
                print("\n  Timeline Preview:")
                for i, event in enumerate(timeline.events[:5], 1):
                    date_str = "undated"
                    if event.date:
                        date_str = event.date.strftime("%Y-%m-%d")
                    elif event.estimated_date:
                        date_str = f"~{event.estimated_date.strftime('%Y-%m-%d')}"
                    print(f"    {i}. [{date_str}] {event.description[:60]}...")
                    print(f"       Importance: {event.importance_score:.2f}, Confidence: {event.confidence:.2f}")
            
            results['timeline'] = {
                'events': len(timeline.events),
                'completeness': timeline.completeness_score,
                'consistency': timeline.consistency_score,
                'time': timeline_time
            }
            
            # Step 6: Report Generation
            print("\n6. REPORT GENERATION")
            print("-" * 20)
            report_start = time.time()
            
            report = self.report_agent.generate_report(timeline, query, articles)
            
            report_time = time.time() - report_start
            print(f"  ✓ Executive Summary: {len(report.executive_summary)} chars")
            print(f"  ✓ Key Findings: {len(report.key_findings)}")
            print(f"  ✓ Sections: {len(report.sections)}")
            print(f"  ✓ Citations: {len(report.citations)}")
            print(f"  ✓ Confidence: {report.confidence_score:.2f}")
            print(f"  ✓ Time: {report_time:.2f}s")
            
            # Show executive summary preview
            print("\n  Executive Summary Preview:")
            print(f"    {report.executive_summary[:200]}...")
            
            results['report'] = {
                'sections': len(report.sections),
                'citations': len(report.citations),
                'confidence': report.confidence_score,
                'time': report_time,
                'full_report': report
            }
            
            # Total metrics
            total_time = time.time() - total_start
            results['total_time'] = total_time
            
            print(f"\n{'='*60}")
            print(f"WORKFLOW COMPLETED in {total_time:.2f}s")
            print(f"{'='*60}")
            
            # Validation against expected results
            if expected_results:
                self._validate_results(results, expected_results)
            
            return results
            
        except Exception as e:
            print(f"\n❌ WORKFLOW FAILED: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            return results
    
    def _validate_results(self, actual: Dict[str, Any], expected: Dict[str, Any]):
        """Validate actual results against expected"""
        print("\nVALIDATION:")
        print("-" * 20)
        
        validations = []
        
        # Validate query analysis
        if 'query_type' in expected:
            match = actual.get('query_analysis', {}).get('type') == expected['query_type']
            validations.append(('Query Type', match))
        
        # Validate search results
        if 'min_search_results' in expected:
            match = actual.get('search', {}).get('total_results', 0) >= expected['min_search_results']
            validations.append(('Min Search Results', match))
        
        # Validate extraction
        if 'min_events' in expected:
            match = actual.get('extraction', {}).get('events', 0) >= expected['min_events']
            validations.append(('Min Events', match))
        
        # Validate timeline
        if 'min_timeline_events' in expected:
            match = actual.get('timeline', {}).get('events', 0) >= expected['min_timeline_events']
            validations.append(('Min Timeline Events', match))
        
        # Print validation results
        for check, passed in validations:
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        passed = sum(1 for _, p in validations if p)
        print(f"\nValidation: {passed}/{len(validations)} checks passed")
    
    def save_report(self, results: Dict[str, Any], output_file: str = "workflow_test_report.md"):
        """Save the generated report to file"""
        if 'report' in results and 'full_report' in results['report']:
            report = results['report']['full_report']
            formatted = self.report_agent.format_report(report, "markdown")
            
            output_path = Path(output_file)
            output_path.write_text(formatted)
            print(f"\n📄 Report saved to: {output_path}")
            
            # Also save metrics
            metrics_path = output_path.with_suffix('.metrics.json')
            metrics = {k: v for k, v in results.items() if k != 'report'}
            metrics['report'] = {k: v for k, v in results['report'].items() if k != 'full_report'}
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"📊 Metrics saved to: {metrics_path}")


def test_basic_queries():
    """Test basic query types"""
    print("\n" + "="*60)
    print("BASIC QUERY TESTS")
    print("="*60)
    
    tester = WorkflowTester()
    
    # Test 1: Factual query about specific event
    print("\nTest 1: Factual Query")
    results1 = tester.test_workflow(
        "What was the Chesapeake Energy acquisition deal in 2024?",
        expected_results={
            'query_type': 'FACTUAL',
            'min_search_results': 1,
            'min_events': 1,
            'min_timeline_events': 1
        }
    )
    
    # Test 2: Temporal query
    print("\nTest 2: Temporal Query")
    results2 = tester.test_workflow(
        "What happened in the energy sector in January 2024?",
        expected_results={
            'query_type': 'TEMPORAL',
            'min_search_results': 2,
            'min_events': 2,
            'min_timeline_events': 2
        }
    )
    
    # Test 3: Entity-focused query
    print("\nTest 3: Entity Query")
    results3 = tester.test_workflow(
        "What did the EU do regarding energy regulations?",
        expected_results={
            'query_type': 'ENTITY',
            'min_search_results': 1,
            'min_events': 1,
            'min_timeline_events': 1
        }
    )
    
    return [results1, results2, results3]


def test_complex_queries():
    """Test complex, multi-faceted queries"""
    print("\n" + "="*60)
    print("COMPLEX QUERY TESTS")
    print("="*60)
    
    tester = WorkflowTester()
    
    # Test comparative query
    print("\nTest: Comparative Query")
    results = tester.test_workflow(
        "Compare energy sector mergers with ESG investment trends in 2024",
        expected_results={
            'query_type': 'COMPARATIVE',
            'min_search_results': 2,
            'min_events': 3,
            'min_timeline_events': 2
        }
    )
    
    # Save the report
    tester.save_report(results, "complex_query_report.md")
    
    return results


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    tester = WorkflowTester()
    
    # Test 1: Query with no results
    print("\nTest 1: No Results Query")
    results1 = tester.test_workflow(
        "What happened with quantum computing stocks in 1850?",
        expected_results={
            'min_search_results': 0,
            'min_events': 0
        }
    )
    
    # Test 2: Very broad query
    print("\nTest 2: Broad Query")
    results2 = tester.test_workflow(
        "Tell me everything about business",
        expected_results={
            'query_type': 'CONCEPTUAL',
            'min_search_results': 3
        }
    )
    
    return [results1, results2]


def test_performance():
    """Test performance with multiple queries"""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    tester = WorkflowTester()
    
    queries = [
        "Chesapeake Energy merger details",
        "EU regulatory changes in 2024",
        "ESG investment trends"
    ]
    
    total_times = []
    component_times = {
        'query_analysis': [],
        'search': [],
        'extraction': [],
        'timeline': [],
        'report': []
    }
    
    for query in queries:
        print(f"\nProcessing: {query}")
        results = tester.test_workflow(query)
        
        if 'total_time' in results:
            total_times.append(results['total_time'])
            
            # Collect component times
            for component in component_times:
                if component in results and 'time' in results[component]:
                    component_times[component].append(results[component]['time'])
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if total_times:
        print(f"\nTotal Workflow Times:")
        print(f"  - Average: {sum(total_times)/len(total_times):.2f}s")
        print(f"  - Min: {min(total_times):.2f}s")
        print(f"  - Max: {max(total_times):.2f}s")
        
        print(f"\nComponent Average Times:")
        for component, times in component_times.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"  - {component}: {avg_time:.2f}s")


def main():
    """Run all workflow tests"""
    print("\n" + "="*80)
    print("AGENTIC NEWS RAG - FULL WORKFLOW TEST SUITE")
    print("="*80)
    
    print("\nChecking prerequisites...")
    
    # Check if articles are indexed
    articles_dir = Path("text_articles")
    if not articles_dir.exists() or not list(articles_dir.glob("*.txt")):
        print("❌ No articles found in text_articles/")
        print("   Please ensure articles are available")
        return
    
    print("✓ Articles directory found")
    print("\nStarting tests...\n")
    
    # Run test suites
    try:
        # Basic tests
        basic_results = test_basic_queries()
        
        # Complex tests
        complex_results = test_complex_queries()
        
        # Edge cases
        edge_results = test_edge_cases()
        
        # Performance test
        test_performance()
        
        print("\n" + "="*80)
        print("TEST SUITE COMPLETED")
        print("="*80)
        
        # Summary
        total_tests = len(basic_results) + 1 + len(edge_results) + 3  # performance queries
        successful_tests = sum(
            1 for r in basic_results + [complex_results] + edge_results 
            if 'error' not in r
        )
        
        print(f"\nSummary: {successful_tests}/{total_tests} workflows completed successfully")
        
        if successful_tests == total_tests:
            print("\n🎉 All workflow tests passed!")
        else:
            print(f"\n⚠️  {total_tests - successful_tests} workflows had issues")
        
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
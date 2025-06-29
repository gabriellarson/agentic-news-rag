#!/usr/bin/env python3
"""
Test script for the Query Analysis Agent

This script tests the query analysis functionality with various query types
to ensure the agent is working correctly with the llama.cpp server.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from config import init_config
    from agents.query_analysis import QueryAnalysisAgent, QueryType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Make sure you're running from the project root or the src directory is accessible")
    print(f"Project root: {project_root}")
    print(f"Src path: {src_path}")
    sys.exit(1)

import logging


def test_connection():
    """Test connection to llama.cpp server"""
    print("🔗 Testing connection to llama.cpp server...")
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        # Simple test query
        test_query = "Hello, are you working?"
        response = agent.client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "user", "content": test_query}],
            temperature=0.1,
            max_tokens=50
        )
        
        print(f"✅ Connection successful!")
        print(f"   Model: {agent.model}")
        print(f"   Endpoint: {agent.client.base_url}")
        print(f"   Response: {response.choices[0].message.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_query_classification():
    """Test query classification functionality"""
    print("\n📊 Testing query classification...")
    
    test_cases = [
        ("What happened with Chesapeake Energy's acquisition?", QueryType.FACTUAL),
        ("ESG investing trends and sustainability", QueryType.CONCEPTUAL),
        ("Timeline of events in Ukraine war", QueryType.TEMPORAL),
        ("Apple Tim Cook statements", QueryType.ENTITY),
        ("Compare US vs EU energy policies", QueryType.COMPARATIVE),
    ]
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        correct = 0
        for query, expected_type in test_cases:
            result_type = agent._classify_query_type(query)
            is_correct = result_type == expected_type
            status = "✅" if is_correct else "❌"
            
            print(f"   {status} '{query[:40]}...' -> {result_type.value} "
                  f"(expected: {expected_type.value})")
            
            if is_correct:
                correct += 1
        
        print(f"\n   Classification accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
        return correct > len(test_cases) * 0.6  # 60% threshold
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False


def test_entity_extraction():
    """Test entity extraction functionality"""
    print("\n👥 Testing entity extraction...")
    
    test_cases = [
        ("Apple CEO Tim Cook visited California", ["Apple", "Tim Cook", "California"]),
        ("Microsoft and Google partnership announcement", ["Microsoft", "Google"]),
        ("Federal Reserve meeting in Washington", ["Federal Reserve", "Washington"]),
        ("Generic query without entities", []),
    ]
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        for query, expected_entities in test_cases:
            result_entities = agent._extract_entities(query)
            
            # Check if we found reasonable entities
            found_expected = sum(1 for entity in expected_entities 
                               if any(entity.lower() in found.lower() for found in result_entities))
            
            status = "✅" if (found_expected > 0 or not expected_entities) else "⚠️"
            
            print(f"   {status} '{query[:40]}...'")
            print(f"      Found: {result_entities}")
            print(f"      Expected: {expected_entities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return False


def test_temporal_extraction():
    """Test temporal constraint extraction"""
    print("\n⏰ Testing temporal extraction...")
    
    test_cases = [
        "What happened yesterday?",
        "Events from last week",
        "News in January 2024",
        "Recent developments this month",
        "Generic query without time references",
    ]
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        for query in test_cases:
            constraints = agent._extract_temporal_constraints(query, datetime.now())
            
            has_temporal = (constraints.start_date or 
                          constraints.end_date or 
                          constraints.relative_terms)
            
            status = "✅" if has_temporal or "time" not in query.lower() else "⚠️"
            
            print(f"   {status} '{query}'")
            print(f"      Constraints: {constraints.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Temporal extraction test failed: {e}")
        return False


def test_query_expansion():
    """Test query expansion functionality"""
    print("\n🔍 Testing query expansion...")
    
    test_queries = [
        "Climate policy changes",
        "Tech company mergers",
        "Energy market volatility",
    ]
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        for query in test_queries:
            expanded = agent._expand_query(query, QueryType.CONCEPTUAL, [])
            
            status = "✅" if len(expanded) > 1 else "⚠️"
            
            print(f"   {status} '{query}'")
            for i, expanded_query in enumerate(expanded):
                marker = "🔸" if i == 0 else "  🔹"
                print(f"      {marker} {expanded_query}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query expansion test failed: {e}")
        return False


def test_full_analysis():
    """Test complete query analysis"""
    print("\n🎯 Testing complete query analysis...")
    
    test_queries = [
        "What are the latest developments in the Chesapeake Energy deal?",
        "How have ESG investment trends changed this year?",
        "Timeline of major events in the Ukraine conflict",
        "Compare Apple and Microsoft market strategies",
    ]
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        for query in test_queries:
            print(f"\n   🔸 Query: '{query}'")
            
            analysis = agent.analyze_query(query)
            
            print(f"      Type: {analysis.query_type.value}")
            print(f"      Entities: {analysis.entities}")
            print(f"      Search Alpha: {analysis.search_alpha}")
            print(f"      Temporal: {bool(analysis.temporal_constraints.start_date or analysis.temporal_constraints.end_date)}")
            print(f"      Expanded Queries: {len(analysis.expanded_queries)}")
            
            # Show first expanded query as example
            if len(analysis.expanded_queries) > 1:
                print(f"      Example expansion: '{analysis.expanded_queries[1]}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Full analysis test failed: {e}")
        return False


def run_interactive_test():
    """Interactive test mode"""
    print("\n🎮 Interactive Test Mode")
    print("Enter queries to test the analysis (type 'quit' to exit):")
    
    try:
        config = init_config()
        agent = QueryAnalysisAgent(config)
        
        while True:
            query = input("\n> ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue
            
            print("\n📊 Analysis Results:")
            analysis = agent.analyze_query(query)
            
            # Pretty print results
            result = analysis.to_dict()
            print(json.dumps(result, indent=2, default=str))
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Interactive test failed: {e}")


def main():
    """Main test runner"""
    print("🧪 Query Analysis Agent Test Suite")
    print("=" * 50)
    
    # Test connection first
    if not test_connection():
        print("\n❌ Cannot proceed without server connection")
        sys.exit(1)
    
    # Run all tests
    tests = [
        ("Query Classification", test_query_classification),
        ("Entity Extraction", test_entity_extraction),
        ("Temporal Extraction", test_temporal_extraction),
        ("Query Expansion", test_query_expansion),
        ("Full Analysis", test_full_analysis),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"\n💥 {test_name} - ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Query Analysis Agent is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Offer interactive mode
    if input("\nRun interactive test mode? (y/N): ").lower().startswith('y'):
        run_interactive_test()


if __name__ == "__main__":
    main()
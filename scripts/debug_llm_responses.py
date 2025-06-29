#!/usr/bin/env python3
"""
Debug script to see raw LLM responses and optimize prompts for Qwen3-30B
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config import init_config
from openai import OpenAI

def test_classification_prompts():
    """Test different classification prompt formats"""
    config = init_config()
    llm_config = config.get_llm_config()
    
    client = OpenAI(
        base_url=llm_config.get('endpoint', 'http://localhost:8001/v1'),
        api_key="not-needed"
    )
    model = llm_config.get('model')
    
    test_query = "What happened with Chesapeake Energy's acquisition?"
    
    prompts = {
        "Current Prompt": '''Classify this news search query into one category:

Query: "What happened with Chesapeake Energy's acquisition?"

Categories:
- FACTUAL: Looking for specific facts or events
- CONCEPTUAL: Looking for broader concepts, trends, or themes  
- TEMPORAL: Focused on time-based events or chronology
- ENTITY: Focused on specific people, companies, or organizations
- COMPARATIVE: Comparing different events or entities

Answer with only one word: FACTUAL, CONCEPTUAL, TEMPORAL, ENTITY, or COMPARATIVE''',

        "Simple Prompt": '''Classify this query: "What happened with Chesapeake Energy's acquisition?"

Choose: FACTUAL, CONCEPTUAL, TEMPORAL, ENTITY, or COMPARATIVE

Answer:''',

        "Direct Prompt": '''Query: "What happened with Chesapeake Energy's acquisition?"

This is a: FACTUAL query (looking for specific facts about an event)

Classify as: FACTUAL''',

        "System Message": '''You are a news query classifier. Classify queries as: FACTUAL, CONCEPTUAL, TEMPORAL, ENTITY, or COMPARATIVE.

Query: "What happened with Chesapeake Energy's acquisition?"
Classification:'''
    }
    
    for prompt_name, prompt in prompts.items():
        print(f"\n{'='*60}")
        print(f"Testing: {prompt_name}")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}")
        print(f"\n{'-'*40}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            raw_response = response.choices[0].message.content
            print(f"Raw Response: '{raw_response}'")
            print(f"Stripped: '{raw_response.strip()}'")
            print(f"Length: {len(raw_response.strip())}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_entity_extraction_prompts():
    """Test different entity extraction prompt formats"""
    config = init_config()
    llm_config = config.get_llm_config()
    
    client = OpenAI(
        base_url=llm_config.get('endpoint', 'http://localhost:8001/v1'),
        api_key="not-needed"
    )
    model = llm_config.get('model')
    
    test_query = "Apple CEO Tim Cook visited California"
    
    prompts = {
        "Current JSON Prompt": '''Extract named entities from this query:

Query: "Apple CEO Tim Cook visited California"

Find all:
- People names
- Company names  
- Organization names
- Location names

Return as JSON array. Examples:
["Apple Inc.", "Tim Cook", "California"]
[]

Query: "Apple CEO Tim Cook visited California"
JSON array:''',

        "Simple List Prompt": '''Extract entities from: "Apple CEO Tim Cook visited California"

Entities:
- Apple
- Tim Cook
- California

Extract entities from: "Apple CEO Tim Cook visited California"

Entities:''',

        "Direct Format": '''Text: "Apple CEO Tim Cook visited California"
Entities: Apple, Tim Cook, California

Text: "Apple CEO Tim Cook visited California"
Entities:'''
    }
    
    for prompt_name, prompt in prompts.items():
        print(f"\n{'='*60}")
        print(f"Testing: {prompt_name}")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}")
        print(f"\n{'-'*40}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            raw_response = response.choices[0].message.content
            print(f"Raw Response: '{raw_response}'")
            print(f"Length: {len(raw_response.strip())}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_basic_response():
    """Test basic LLM response"""
    config = init_config()
    llm_config = config.get_llm_config()
    
    client = OpenAI(
        base_url=llm_config.get('endpoint', 'http://localhost:8001/v1'),
        api_key="not-needed"
    )
    model = llm_config.get('model')
    
    simple_prompts = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Classify this as positive or negative: 'I love this!'",
        "Extract the name from: 'Hello, my name is John'",
        "Return only the word FACTUAL",
        "Answer with one word: FACTUAL"
    ]
    
    for prompt in simple_prompts:
        print(f"\n{'-'*40}")
        print(f"Prompt: {prompt}")
        print(f"{'-'*40}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            raw_response = response.choices[0].message.content
            print(f"Response: '{raw_response}'")
            
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("🔍 LLM Response Debug Tool")
    print("=" * 60)
    
    print("\n1. Testing Basic Responses...")
    test_basic_response()
    
    print("\n\n2. Testing Classification Prompts...")
    test_classification_prompts()
    
    print("\n\n3. Testing Entity Extraction Prompts...")
    test_entity_extraction_prompts()

if __name__ == "__main__":
    main()
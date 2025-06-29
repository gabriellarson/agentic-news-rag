#!/usr/bin/env python3
"""
Diagnostic script to identify why LLM is returning empty responses
"""

import sys
from pathlib import Path
import json

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config import init_config
from openai import OpenAI

def test_different_parameters():
    """Test different API parameters to identify the issue"""
    config = init_config()
    llm_config = config.get_llm_config()
    
    client = OpenAI(
        base_url=llm_config.get('endpoint', 'http://localhost:8001/v1'),
        api_key="not-needed"
    )
    model = llm_config.get('model', 'qwen3-30b')
    
    test_prompt = "Hello world"
    
    # Test different parameter combinations
    parameter_sets = [
        {"temperature": 0.1, "max_tokens": 50},
        {"temperature": 0.7, "max_tokens": 50},
        {"temperature": 0.1, "max_tokens": 100},
        {"temperature": 0.1, "max_tokens": 10},
        {"temperature": 1.0, "max_tokens": 50},
        {"temperature": 0.0, "max_tokens": 50},
    ]
    
    for i, params in enumerate(parameter_sets, 1):
        print(f"\n{'-'*50}")
        print(f"Test {i}: {params}")
        print(f"{'-'*50}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                **params
            )
            
            content = response.choices[0].message.content
            print(f"Response: '{content}'")
            print(f"Length: {len(content) if content else 0}")
            
            # Print full response object for debugging
            print(f"Finish reason: {response.choices[0].finish_reason}")
            print(f"Model used: {response.model}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_different_roles():
    """Test different message roles"""
    config = init_config()
    llm_config = config.get_llm_config()
    
    client = OpenAI(
        base_url=llm_config.get('endpoint', 'http://localhost:8001/v1'),
        api_key="not-needed"
    )
    model = llm_config.get('model', 'qwen3-30b')
    
    test_prompt = "What is 2 + 2?"
    
    message_formats = [
        [{"role": "user", "content": test_prompt}],
        [{"role": "system", "content": "You are a helpful assistant."}, 
         {"role": "user", "content": test_prompt}],
        [{"role": "assistant", "content": "Hello!"}, 
         {"role": "user", "content": test_prompt}],
    ]
    
    for i, messages in enumerate(message_formats, 1):
        print(f"\n{'-'*50}")
        print(f"Message Format {i}: {messages}")
        print(f"{'-'*50}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            print(f"Response: '{content}'")
            print(f"Length: {len(content) if content else 0}")
            
        except Exception as e:
            print(f"Error: {e}")

def test_raw_request():
    """Test raw HTTP request to see if it's an OpenAI client issue"""
    import requests
    
    config = init_config()
    llm_config = config.get_llm_config()
    
    endpoint = llm_config.get('endpoint', 'http://localhost:8001/v1')
    model = llm_config.get('model', 'qwen3-30b')
    
    url = f"{endpoint}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello world"}],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    
    print(f"\n{'-'*50}")
    print(f"Raw HTTP Request Test")
    print(f"{'-'*50}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"Extracted Content: '{content}'")
            print(f"Length: {len(content)}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_model_name_variations():
    """Test different model name variations"""
    config = init_config()
    llm_config = config.get_llm_config()
    
    client = OpenAI(
        base_url=llm_config.get('endpoint', 'http://localhost:8001/v1'),
        api_key="not-needed"
    )
    
    # Try different model names
    model_names = [
        "qwen3-30b",
        "qwen3-30b-a3b",
        "qwen",
        "default",
        "",  # Empty model name
        "gpt-3.5-turbo",  # Generic name
    ]
    
    test_prompt = "Hello"
    
    for model_name in model_names:
        print(f"\n{'-'*50}")
        print(f"Testing model: '{model_name}'")
        print(f"{'-'*50}")
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            print(f"Response: '{content}'")
            print(f"Length: {len(content) if content else 0}")
            print(f"Model returned: {response.model}")
            
        except Exception as e:
            print(f"Error: {e}")

def check_server_health():
    """Check if the llama.cpp server is healthy"""
    import requests
    
    config = init_config()
    llm_config = config.get_llm_config()
    
    base_url = llm_config.get('endpoint', 'http://localhost:8001/v1')
    
    # Try health/models endpoints
    endpoints_to_test = [
        f"{base_url}/models",
        f"{base_url.replace('/v1', '')}/health",
        f"{base_url.replace('/v1', '')}/v1/models",
    ]
    
    for endpoint in endpoints_to_test:
        print(f"\n{'-'*50}")
        print(f"Testing endpoint: {endpoint}")
        print(f"{'-'*50}")
        
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}...")  # First 500 chars
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("🔍 LLM Issue Diagnostics")
    print("=" * 60)
    
    print("\n1. Checking server health...")
    check_server_health()
    
    print("\n\n2. Testing different model names...")
    test_model_name_variations()
    
    print("\n\n3. Testing different parameters...")
    test_different_parameters()
    
    print("\n\n4. Testing different message roles...")
    test_different_roles()
    
    print("\n\n5. Testing raw HTTP request...")
    test_raw_request()
    
    print("\n" + "=" * 60)
    print("🔧 Recommendations:")
    print("1. Check if llama.cpp server is properly loaded with the model")
    print("2. Verify the correct model name in your server")
    print("3. Check server logs for any errors")
    print("4. Try connecting with curl directly to test the server")

if __name__ == "__main__":
    main()
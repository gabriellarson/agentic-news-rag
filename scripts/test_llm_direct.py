#!/usr/bin/env python3
"""
Direct test of LLM responses to debug empty output issue
"""

import requests
import json

def test_direct_api():
    """Test the LLM API directly with minimal setup"""
    
    # Direct API call without any framework
    url = "http://localhost:8001/v1/chat/completions"
    
    # Test 1: Very simple prompt
    print("Test 1: Simple prompt")
    print("-" * 50)
    
    payload = {
        "model": "D:\\AI\\GGUFs\\Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Full response: {json.dumps(data, indent=2)}")
        content = data['choices'][0]['message']['content']
        print(f"Content: '{content}'")
        print(f"Content length: {len(content)}")
    else:
        print(f"Error: {response.text}")
    
    # Test 2: Classification prompt
    print("\n\nTest 2: Classification prompt")
    print("-" * 50)
    
    payload = {
        "model": "D:\\AI\\GGUFs\\Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
        "messages": [
            {"role": "user", "content": "Answer with one word: FACTUAL"}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(f"Content: '{content}'")
        print(f"Content repr: {repr(content)}")
        print(f"Content bytes: {content.encode('utf-8')}")
    
    # Test 3: With system message
    print("\n\nTest 3: With system message")
    print("-" * 50)
    
    payload = {
        "model": "D:\\AI\\GGUFs\\Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
        "messages": [
            {"role": "system", "content": "You always respond with exactly one word."},
            {"role": "user", "content": "What category is this: FACTUAL"}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        content = data['choices'][0]['message']['content']
        print(f"Content: '{content}'")
    
    # Test 4: Different parameters
    print("\n\nTest 4: Different parameters")
    print("-" * 50)
    
    test_params = [
        {"temperature": 0.0, "max_tokens": 50},
        {"temperature": 1.0, "max_tokens": 50},
        {"temperature": 0.7, "max_tokens": 100, "top_p": 0.9},
        {"temperature": 0.7, "max_tokens": 100, "n": 1}
    ]
    
    for params in test_params:
        payload = {
            "model": "D:\\AI\\GGUFs\\Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            **params
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"Params {params}: Content='{content}' (len={len(content)})")

if __name__ == "__main__":
    test_direct_api()
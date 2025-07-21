#!/usr/bin/env python3
"""
Secure Perplexity API query script.
This script securely retrieves the Perplexity API key from the macOS keychain
and makes a query without exposing the key.
"""

import json
import subprocess
import sys
import requests

def get_api_key():
    """Securely retrieve API key from macOS keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "perplexity-api", "-w"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error: Could not retrieve Perplexity API key from keychain")
        return None

def query_perplexity(query_text):
    """Make a query to Perplexity API."""
    api_key = get_api_key()
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar",
        "messages": [{"role": "user", "content": query_text}]
    }
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error querying Perplexity API: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python secure_perplexity_query.py 'your query here'")
        return 1
    
    query = sys.argv[1]
    result = query_perplexity(query)
    
    if result and "choices" in result:
        print(result["choices"][0]["message"]["content"])
    else:
        print("No results or error occurred")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

import os
import logging
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

log = logging.getLogger("rag")

# Configuration
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-tiny")

def generate_answer(prompt: str) -> str:
    """
    Generate an answer using Mistral API via direct HTTP request.
    This bypasses the SDK import issues.
    """
    
    if not API_KEY:
        log.error("❌ MISTRAL_API_KEY not found in .env file")
        return "❌ No API key found. Please add MISTRAL_API_KEY to .env file"
    
    try:
        start_time = time.time()
        log.info("Generating response with Mistral API (HTTP)...")
        
        # Mistral API endpoint
        url = "https://api.mistral.ai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 8192,
            "top_p": 0.9,
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        elapsed = time.time() - start_time
        log.info(f"✅ Response received in {elapsed:.2f} seconds")
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content'].strip()
            
            if answer:
                return answer
            else:
                return "No response generated."
        else:
            # Handle errors
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            except:
                error_msg = response.text
            
            log.error(f"API error {response.status_code}: {error_msg}")
            
            if response.status_code == 401:
                return "❌ Invalid API key. Please check your MISTRAL_API_KEY"
            elif response.status_code == 429:
                return "⏳ Rate limit exceeded. Please wait a moment and try again."
            else:
                return f"❌ API error ({response.status_code}): {error_msg}"
            
    except requests.exceptions.Timeout:
        return "⏳ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ Connection error. Please check your internet connection."
    except Exception as e:
        log.error(f"Generation error: {e}")
        return f"❌ Error: {str(e)}"

def generate_answer_stream(prompt: str):
    """
    Streaming version for better user experience.
    """
    if not API_KEY:
        yield "❌ No API key found. Please add MISTRAL_API_KEY to .env file"
        return
    
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stream": True,
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = json.loads(line[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"❌ API error: {response.status_code}"
            
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def test_model():
    """
    Quick test to verify the API is working.
    """
    print("=" * 60)
    print("Testing Mistral API (HTTP Direct)")
    print("=" * 60)
    
    # Check API key
    if not API_KEY:
        print("❌ No API key found in .env")
        print("\nPlease create .env file with:")
        print("  MISTRAL_API_KEY=your-api-key-here")
        print("  MISTRAL_MODEL=mistral-tiny")
        return False
    
    print(f"✅ API Key found: {API_KEY[:8]}...")
    print(f"Model: {MODEL_NAME}")
    print(f"API URL: https://api.mistral.ai/v1/chat/completions")
    
    # Test connection first
    print("\nTesting connection...")
    try:
        response = requests.get("https://api.mistral.ai/v1/models", 
                              headers={"Authorization": f"Bearer {API_KEY}"},
                              timeout=10)
        if response.status_code == 200:
            print("✅ Connection successful!")
            models = response.json()
            print(f"Available models: {[m['id'] for m in models.get('data', [])[:5]]}")
        else:
            print(f"⚠️ Connection test returned: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Connection test failed: {e}")
    
    # Test generation
    print("\nTesting generation...")
    test_prompt = "What is 2+2? Answer in one sentence."
    print(f"Prompt: {test_prompt}")
    print("\nGenerating...")
    
    answer = generate_answer(test_prompt)
    print(f"\nAnswer: {answer}")
    
    if answer and len(answer) > 0:
        print("\n✅ Model is working correctly!")
        return True
    else:
        print("\n❌ Model failed to generate")
        return False

def get_model_info():
    """
    Get information about the current configuration.
    """
    return {
        "provider": "Mistral AI",
        "model": MODEL_NAME,
        "api_key_configured": bool(API_KEY),
        "api_key_preview": f"{API_KEY[:8]}..." if API_KEY else None
    }

if __name__ == "__main__":
    test_model()
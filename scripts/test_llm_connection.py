import os, sys
from src.llm_client import LLMClient

def main():
    print("\n" + "="*80)
    print("LLM Connection Test".center(80))
    print("="*80 + "\n")
    
    try:
        client = LLMClient()
        print(f"API Type: {client.api_type}")
        print(f"Endpoint: {client.endpoint}")
        print(f"Model: {client.model}")
        print(f"Mock: {client.mock}\n")
        
        msg = [{"role":"user","content":"Say 'connection ok'."}]
        print("Sending test message...")
        out = client.chat(msg, max_tokens=16, temperature=0.1)
        print(f"\nLLM Reply: {out[:200]}\n")
        print("✅ LLM connection test PASSED\n")
        return 0
    except Exception as e:
        print(f"❌ LLM connection test FAILED: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

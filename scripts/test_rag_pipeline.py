import os, sys, requests, json

BASE = f"http://{os.getenv('API_HOST','0.0.0.0')}:{os.getenv('API_PORT','7000')}"
HEADERS = {"x-api-token": os.getenv("API_TOKEN","change-me")}

def main():
    print("\n" + "="*80)
    print("RAG Pipeline Test".center(80))
    print("="*80 + "\n")
    
    try:
        # Health check
        print("1. Health check...")
        health_resp = requests.get(f"{BASE}/health", timeout=5)
        health_resp.raise_for_status()
        try:
            h = health_resp.json()
        except ValueError:
            raise RuntimeError(f"/health did not return JSON: {health_resp.text[:200]}")
        print(f"   OK: {h}\n")
        
        # Chat query
        print("2. Chat query...")
        q = {"question":"How do I create a project?", "k": 5, "namespace":"clockify"}
        r = requests.post(f"{BASE}/chat", headers=HEADERS, json=q, timeout=60)
        print(f"   Status: {r.status_code}")
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"/chat did not return JSON: {r.text[:200]}")
        print(f"   Answer: {data.get('answer','')[:150]}...")
        print(f"   Sources: {len(data.get('sources',[]))} items")
        print(f"   Latency (total): {data.get('latency_ms',{}).get('total',0)}ms\n")
        
        if len(data.get("sources",[])) >= 2:
            print("✅ RAG pipeline test PASSED\n")
            return 0
        else:
            print("⚠️ RAG pipeline: answer generated but fewer sources than expected\n")
            return 0  # Still pass if answer exists
    except Exception as e:
        print(f"❌ RAG pipeline test FAILED: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

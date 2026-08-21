import requests
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing /")
    try:
        r = requests.get(f"{base_url}/")
        print(f"Root: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Root error: {e}")
        
    print("\nTesting validation with invalid chunking parameters")
    payload = {
        "source": "test_data/sample.txt",
        "metadata": {},
        "chunk_size": 1,
        "chunk_overlap": 1000
    }
    r = requests.post(f"{base_url}/api/v1/documents/ingest", json=payload)
    print(f"Ingest invalid: {r.status_code} - {r.text}")

    print("\nTesting successful ingestion")
    payload = {
        "source": "test_data/sample.txt",
        "metadata": {"test": True},
        "chunk_size": 500,
        "chunk_overlap": 100
    }
    r = requests.post(f"{base_url}/api/v1/documents/ingest", json=payload)
    print(f"Ingest valid: {r.status_code} - {r.text}")

if __name__ == "__main__":
    test_api()

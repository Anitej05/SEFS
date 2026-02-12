"""
Quick test script to verify backend endpoints
"""
import requests

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("Testing SEFS Backend Endpoints\n" + "="*50)
    
    # Test root
    try:
        res = requests.get(f"{BASE_URL}/")
        print(f"✓ Root: {res.json()}")
    except Exception as e:
        print(f"✗ Root failed: {e}")
    
    # Test system-info
    try:
        res = requests.get(f"{BASE_URL}/system-info")
        print(f"✓ System Info: {res.json()}")
    except Exception as e:
        print(f"✗ System Info failed: {e}")
    
    # Test stats
    try:
        res = requests.get(f"{BASE_URL}/stats")
        print(f"✓ Stats: {res.json()}")
    except Exception as e:
        print(f"✗ Stats failed: {e}")
    
    # Test files
    try:
        res = requests.get(f"{BASE_URL}/files")
        data = res.json()
        print(f"✓ Files: {len(data)} items")
        for item in data[:3]:
            print(f"  - {item.get('name')} ({item.get('type')})")
    except Exception as e:
        print(f"✗ Files failed: {e}")

if __name__ == "__main__":
    test_endpoints()

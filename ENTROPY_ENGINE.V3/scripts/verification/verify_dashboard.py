import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
"""
Verify dashboard server connectivity.
"""
import time
import requests
import threading
from entropy.dashboard.server import DashboardServer

def run_server():
    server = DashboardServer(port=8081)
    server.start()
    while True:
        time.sleep(1)

def verify():
    print("Starting server in background...")
    # In a real test we'd use a thread, but for this quick check:
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)
    
    print("Checking status endpoint...")
    try:
        resp = requests.get("http://localhost:8081/api/status")
        print(f"Status Code: {resp.status_code}")
        print(f"Content: {resp.json()}")
        if resp.status_code == 200:
            print("✅ Dashboard Server Verified")
        else:
            print("❌ Dashboard Server Failed")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    verify()

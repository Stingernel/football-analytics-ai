"""
Check API Status
Validates API keys and checks quota/status.
"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load env variables explicitly
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

def check_odds_api():
    print("\n[1] Checking The Odds API...")
    api_key = os.getenv("ODDS_API_KEY")
    
    if not api_key:
        print("❌ ODDS_API_KEY is MISSING in .env")
        return
    
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    print(f"   Key found: {masked_key}")
    
    if api_key == "your-api-key-here":
        print("❌ Key is default placeholder! Please update .env")
        return
        
    url = f"https://api.the-odds-api.com/v4/sports?apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        print(f"   Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            # Check headers for quota
            requests_remaining = resp.headers.get("x-requests-remaining", "Unknown")
            requests_used = resp.headers.get("x-requests-used", "Unknown")
            print(f"✅ API Key is VALID")
            print(f"   Requests Remaining: {requests_remaining}")
            print(f"   Requests Used: {requests_used}")
        elif resp.status_code == 401:
            print("❌ Unauthorized: Invalid API Key")
        elif resp.status_code == 429:
            print("❌ Quota Exceeded (Rate Limited)")
        else:
            print(f"❌ Error: {resp.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def check_football_data_api():
    print("\n[2] Checking Football-Data.org API...")
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    
    if not api_key:
        print("❌ FOOTBALL_DATA_API_KEY is MISSING in .env")
        return

    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    print(f"   Key found: {masked_key}")
    
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=FINISHED&limit=1"
    headers = {'X-Auth-Token': api_key}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"   Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            print(f"✅ API Key is VALID")
        elif resp.status_code == 403:
             print("❌ Restricted: API Key valid but likely Free Tier limited (No historical access or Rate Limited)")
        elif resp.status_code == 400 or resp.status_code == 401:
             print("❌ Invalid API Key")
        elif resp.status_code == 429:
             print("❌ Rate Limit Exceeded")
        else:
             print(f"❌ Error: {resp.text}")

    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("API STATUS CHECK")
    print("="*60)
    check_odds_api()
    check_football_data_api()

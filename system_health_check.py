"""
SYSTEM HEALTH CHECK - FINAL AUDIT
Run by: IT & Betting Consultant Agent
Purpose: Verify operational readiness before "Go Live"
"""
import sys
import os
import requests
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime
import logging

# Setup
sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🕵️  AUDIT: {title}")
    print(f"{'='*60}")

def check_api_keys():
    print_header("API & EXTERNAL CONNECTIONS")
    
    # 1. The Odds API
    key_odds = os.getenv("ODDS_API_KEY")
    if key_odds and len(key_odds) > 10 and key_odds != "your-api-key-here":
        print(f"✅ The Odds API Key: PRESENT ({key_odds[:4]}...)")
        # Quick connectivity check
        try:
            resp = requests.get(f"https://api.the-odds-api.com/v4/sports?apiKey={key_odds}")
            if resp.status_code == 200:
                quota = resp.headers.get('x-requests-remaining', '?')
                print(f"   └── Status: CONNECTED (Quota: {quota} left)")
            else:
                print(f"   └── Status: ⚠️ ERROR {resp.status_code}")
        except:
            print("   └── Status: ❌ CONNECTION FAILED")
    else:
        print("❌ The Odds API Key: MISSING or INVALID")

    # 2. Football-Data.org
    key_fd = os.getenv("FOOTBALL_DATA_API_KEY")
    if key_fd and len(key_fd) > 10:
        print(f"✅ Football-Data API Key: PRESENT ({key_fd[:4]}...)")
    else:
        print("⚠️ Football-Data API Key: MISSING (Validation module might fail)")

def check_database_integrity():
    print_header("DATABASE INTEGRITY")
    db_path = "football_analytics.db"
    
    if not os.path.exists(db_path):
        print("❌ Database File: MISSING")
        return
        
    print("✅ Database File: FOUND")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check Tables
        tables = ["fixtures", "predictions", "cached_predictions", "matches"]
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   └── Table '{table}': OK ({count} rows)")
            else:
                print(f"   └── Table '{table}': ❌ MISSING")
                
        # Check for dummy data
        cursor.execute("SELECT COUNT(*) FROM fixtures WHERE source='demo'")
        dummies = cursor.fetchone()[0]
        if dummies > 0:
            print(f"⚠️ WARNING: Found {dummies} fixtures from 'demo' source.")
        else:
            print("✅ Data Quality: CLEAN (No demo fixtures found)")
            
    except Exception as e:
        print(f"❌ Database Error: {e}")
    finally:
        if conn: conn.close()

def check_model_readiness():
    print_header("AI MODEL ENGINE")
    
    model_path = Path("trained_models/simple_model.pkl")
    if model_path.exists():
        size = model_path.stat().st_size
        created = datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"✅ Model File: FOUND ({size} bytes)")
        print(f"   └── Created: {created}")
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            model = data.get('model')
            print(f"   └── Type: {type(model).__name__}")
            if hasattr(model, 'coef_'):
                print("   └── Logic: WEIGHTED (Coefficients loaded)")
            else:
                print("   └── Logic: UNKNOWN")
                
        except Exception as e:
            print(f"   └── Status: ❌ CORRUPT ({e})")
    else:
        print("❌ Model File: MISSING (System will use random fallback!)")

def check_logic_engine():
    print_header("LOGIC SIMULATION")
    
    try:
        from models.ensemble import get_ensemble
        ensemble = get_ensemble()
        
        # Simulating a clear mismatch
        # Man City (Strong) vs Ipswich (Weak)
        match_data = {
            'home_team': 'Man City', 'away_team': 'Ipswich', 
            'home_odds': 1.10, 'draw_odds': 9.0, 'away_odds': 19.0,
            'home_form': 'WWWWW', 'away_form': 'LLLLL',
            'home_xg': 2.8, 'away_xg': 0.5,
            'league': 'Premier League'
        }
        
        print("Testing Scenario: Man City (1.10) vs Ipswich (19.0)")
        result = ensemble.predict(match_data)
        probs = result['probabilities']
        
        print(f"   └── Prediction: {result['predicted_result']}")
        print(f"   └── Probabilities: Home={probs['home_win']:.1%}, Draw={probs['draw']:.1%}, Away={probs['away_win']:.1%}")
        
        if probs['home_win'] > 0.70:
            print("✅ Logic Check: PASSED (Model respects favorites)")
        else:
            print("❌ Logic Check: FAILED (Model output suspicious for clear favorite)")
            
    except Exception as e:
        print(f"❌ Logic Engine Error: {e}")

if __name__ == "__main__":
    print("\n🚀 STARTING PROFESSIONAL AUDIT...\n")
    check_api_keys()
    check_database_integrity()
    check_model_readiness()
    check_logic_engine()
    print("\n" + "="*60)
    print("AUDIT CONCLUSION: CHECK RESULTS ABOVE")
    print("="*60 + "\n")

"""
Auto-Validate System - Fetch results and validate predictions automatically.
Uses Football-Data.org API to get match results.
"""
import sys
import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from app.database import get_db

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
BASE_URL = 'https://api.football-data.org/v4'


def fetch_recent_results():
    """Fetch recent match results from Football-Data.org."""
    if not API_KEY:
        print("ERROR: No FOOTBALL_DATA_API_KEY")
        return []
    
    headers = {'X-Auth-Token': API_KEY}
    
    # Get matches from last 3 days
    date_from = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    date_to = datetime.now().strftime('%Y-%m-%d')
    
    all_results = []
    leagues = ['PL', 'PD', 'BL1', 'SA', 'FL1']  # Major leagues
    
    for league in leagues:
        url = f"{BASE_URL}/competitions/{league}/matches?status=FINISHED&dateFrom={date_from}&dateTo={date_to}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get('matches', []):
                    winner = m.get('score', {}).get('winner')
                    result = 'D' if winner == 'DRAW' else ('H' if winner == 'HOME_TEAM' else 'A')
                    all_results.append({
                        'home_team': m.get('homeTeam', {}).get('name'),
                        'away_team': m.get('awayTeam', {}).get('name'),
                        'result': result,
                        'home_goals': m.get('score', {}).get('fullTime', {}).get('home'),
                        'away_goals': m.get('score', {}).get('fullTime', {}).get('away'),
                        'date': m.get('utcDate')
                    })
            time.sleep(2)  # Rate limit
        except Exception as e:
            print(f"Error fetching {league}: {e}")
    
    return all_results


def match_team_names(pred_team: str, api_team: str) -> bool:
    """Fuzzy match team names."""
    if not pred_team or not api_team:
        return False
    
    # Normalize
    p = pred_team.lower().replace(' fc', '').replace(' cf', '').strip()
    a = api_team.lower().replace(' fc', '').replace(' cf', '').strip()
    
    # Exact match
    if p == a:
        return True
    
    # Partial match
    if p[:6] in a or a[:6] in p:
        return True
    
    return False


def auto_validate():
    """Automatically validate predictions using API results."""
    print("=" * 60)
    print("AUTO-VALIDATION - Fetching Results & Validating")
    print("=" * 60)
    
    # Fetch results
    print("\n1. Fetching recent results from API...")
    results = fetch_recent_results()
    print(f"   Found {len(results)} finished matches")
    
    if not results:
        print("No results to validate against.")
        return
    
    # Get pending predictions
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fixture_id, home_team, away_team, predicted_result
            FROM cached_predictions 
            WHERE (validated = 0 OR validated IS NULL)
        """)
        pending = cursor.fetchall()
    
    print(f"\n2. Checking {len(pending)} pending predictions...")
    
    validated = 0
    correct = 0
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for pred in pending:
            fixture_id = pred[0]
            pred_home = pred[1]
            pred_away = pred[2]
            predicted = pred[3]
            
            # Find matching result
            for r in results:
                if match_team_names(pred_home, r['home_team']) and match_team_names(pred_away, r['away_team']):
                    actual = r['result']
                    is_correct = predicted == actual
                    
                    if is_correct:
                        correct += 1
                    validated += 1
                    
                    status = "✅" if is_correct else "❌"
                    print(f"   {status} {pred_home} vs {pred_away}: Pred={predicted}, Actual={actual}")
                    
                    # Update database
                    cursor.execute("""
                        UPDATE cached_predictions 
                        SET validated = 1, actual_result = ?, was_correct = ?
                        WHERE fixture_id = ?
                    """, (actual, 1 if is_correct else 0, fixture_id))
                    break
        
        conn.commit()
    
    # Report
    print(f"\n{'=' * 60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Validated: {validated}")
    print(f"  Correct: {correct}")
    if validated > 0:
        print(f"  Accuracy: {correct/validated*100:.1f}%")
    
    # Overall stats
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cached_predictions WHERE validated = 1")
        total_val = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM cached_predictions WHERE was_correct = 1")
        total_correct = cursor.fetchone()[0]
    
    print(f"\n📊 OVERALL ACCURACY:")
    print(f"   Total Validated: {total_val}")
    print(f"   Total Correct: {total_correct}")
    if total_val > 0:
        print(f"   Overall Accuracy: {total_correct/total_val*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    auto_validate()

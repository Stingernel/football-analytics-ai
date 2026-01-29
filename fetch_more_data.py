"""Fetch historical data from previous seasons."""
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import get_db
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('FOOTBALL_DATA_API_KEY')
BASE_URL = 'https://api.football-data.org/v4'

LEAGUES = {'Premier League': 'PL', 'La Liga': 'PD', 'Bundesliga': 'BL1', 'Serie A': 'SA', 'Ligue 1': 'FL1'}

def fetch_season(league_code, season, league_name):
    """Fetch one season of data."""
    headers = {'X-Auth-Token': API_KEY}
    url = f"{BASE_URL}/competitions/{league_code}/matches?status=FINISHED&season={season}"
    
    try:
        print(f"  Fetching {league_name} {season}...")
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"    Error: {resp.status_code}")
            return 0
        
        data = resp.json()
        matches = data.get('matches', [])
        
        count = 0
        with get_db() as conn:
            cursor = conn.cursor()
            for m in matches:
                try:
                    fixture_id = f"hist_{season}_{m.get('id')}"
                    score = m.get('score', {}).get('fullTime', {})
                    winner = m.get('score', {}).get('winner')
                    result = 'D' if winner == 'DRAW' else ('H' if winner == 'HOME_TEAM' else 'A')
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO matches 
                        (fixture_id, home_team, away_team, match_date, league, season, home_goals, away_goals, result)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fixture_id,
                        m.get('homeTeam', {}).get('name'),
                        m.get('awayTeam', {}).get('name'),
                        m.get('utcDate'),
                        league_name,
                        f"{season}-{season+1}",
                        score.get('home'),
                        score.get('away'),
                        result
                    ))
                    count += 1
                except:
                    pass
            conn.commit()
        
        print(f"    Saved {count} matches")
        return count
        
    except Exception as e:
        print(f"    Error: {e}")
        return 0


def main():
    print("=" * 50)
    print("FETCHING HISTORICAL DATA (2022, 2023, 2024)")
    print("=" * 50)
    
    if not API_KEY:
        print("ERROR: No API key!")
        return
    
    total = 0
    for season in [2022, 2023, 2024]:
        print(f"\nSeason {season}/{season+1}:")
        for name, code in LEAGUES.items():
            count = fetch_season(code, season, name)
            total += count
            time.sleep(7)  # Rate limit
    
    print(f"\n\nTotal new matches: {total}")
    
    # Verify
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        print(f"Total matches in DB: {cursor.fetchone()[0]}")


if __name__ == "__main__":
    main()

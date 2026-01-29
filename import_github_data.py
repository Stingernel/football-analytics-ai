"""
Import historical football data from GitHub openfootball project.
Source: https://github.com/openfootball/football.json
"""
import sys
import json
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.database import get_db

# GitHub raw URLs for openfootball data
BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

# Seasons and leagues to fetch
DATA_SOURCES = [
    # Premier League
    ("2019-20/en.1.json", "Premier League", "2019-2020"),
    ("2020-21/en.1.json", "Premier League", "2020-2021"),
    ("2021-22/en.1.json", "Premier League", "2021-2022"),
    ("2022-23/en.1.json", "Premier League", "2022-2023"),
    ("2023-24/en.1.json", "Premier League", "2023-2024"),
    # La Liga
    ("2019-20/es.1.json", "La Liga", "2019-2020"),
    ("2020-21/es.1.json", "La Liga", "2020-2021"),
    ("2021-22/es.1.json", "La Liga", "2021-2022"),
    ("2022-23/es.1.json", "La Liga", "2022-2023"),
    ("2023-24/es.1.json", "La Liga", "2023-2024"),
    # Bundesliga
    ("2019-20/de.1.json", "Bundesliga", "2019-2020"),
    ("2020-21/de.1.json", "Bundesliga", "2020-2021"),
    ("2021-22/de.1.json", "Bundesliga", "2021-2022"),
    ("2022-23/de.1.json", "Bundesliga", "2022-2023"),
    ("2023-24/de.1.json", "Bundesliga", "2023-2024"),
    # Serie A
    ("2019-20/it.1.json", "Serie A", "2019-2020"),
    ("2020-21/it.1.json", "Serie A", "2020-2021"),
    ("2021-22/it.1.json", "Serie A", "2021-2022"),
    ("2022-23/it.1.json", "Serie A", "2022-2023"),
    ("2023-24/it.1.json", "Serie A", "2023-2024"),
    # Ligue 1
    ("2019-20/fr.1.json", "Ligue 1", "2019-2020"),
    ("2020-21/fr.1.json", "Ligue 1", "2020-2021"),
    ("2021-22/fr.1.json", "Ligue 1", "2021-2022"),
    ("2022-23/fr.1.json", "Ligue 1", "2022-2023"),
    ("2023-24/fr.1.json", "Ligue 1", "2023-2024"),
]


def fetch_json(path: str) -> dict:
    """Fetch JSON from GitHub."""
    url = f"{BASE_URL}/{path}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def import_matches(data: dict, league: str, season: str) -> int:
    """Import matches to database."""
    if not data or 'matches' not in data:
        return 0
    
    count = 0
    with get_db() as conn:
        cursor = conn.cursor()
        
        for m in data['matches']:
            try:
                score = m.get('score', {}).get('ft', [None, None])
                if not score or len(score) < 2:
                    continue
                
                home_goals = score[0]
                away_goals = score[1]
                
                if home_goals is None or away_goals is None:
                    continue
                
                # Determine result
                if home_goals > away_goals:
                    result = 'H'
                elif away_goals > home_goals:
                    result = 'A'
                else:
                    result = 'D'
                
                home_team = m.get('team1', '')
                away_team = m.get('team2', '')
                match_date = m.get('date', '')
                
                # Unique fixture ID
                fixture_id = f"gh_{season}_{home_team[:4]}_{away_team[:4]}_{match_date}"
                
                cursor.execute("""
                    INSERT OR IGNORE INTO matches 
                    (fixture_id, home_team, away_team, match_date, league, season,
                     home_goals, away_goals, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fixture_id, home_team, away_team, match_date, league, season,
                    home_goals, away_goals, result
                ))
                count += 1
                
            except:
                pass
        
        conn.commit()
    
    return count


def main():
    print("=" * 60)
    print("IMPORTING HISTORICAL DATA FROM GITHUB")
    print("Source: openfootball/football.json")
    print("=" * 60)
    
    total = 0
    
    for path, league, season in DATA_SOURCES:
        print(f"  {league} {season}...", end=" ")
        
        data = fetch_json(path)
        if data:
            count = import_matches(data, league, season)
            print(f"{count} matches")
            total += count
        else:
            print("not available")
    
    print(f"\n{'=' * 60}")
    print(f"Total imported: {total}")
    
    # Verify
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        total_db = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM matches WHERE home_odds IS NOT NULL")
        with_odds = cursor.fetchone()[0]
    
    print(f"Total in database: {total_db}")
    print(f"Matches with odds: {with_odds}")
    print("=" * 60)


if __name__ == "__main__":
    main()

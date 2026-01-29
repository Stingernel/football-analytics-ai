"""
Download and import historical football data with odds from football-data.co.uk
This provides 5+ seasons of data with real betting odds included!
"""
import sys
import csv
import io
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from app.database import get_db

# Football-data.co.uk CSV URLs
# Format: https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    'Premier League': 'E0',
    'Championship': 'E1',
    'La Liga': 'SP1',
    'Bundesliga': 'D1',
    'Serie A': 'I1',
    'Ligue 1': 'F1'
}

# Seasons to fetch (format: 2324 means 2023-2024)
SEASONS = ['1920', '2021', '2122', '2223', '2324', '2425']


def download_csv(season: str, league_code: str) -> str:
    """Download CSV from football-data.co.uk."""
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        else:
            return None
    except:
        return None


def parse_result(row: dict) -> str:
    """Parse result from CSV row."""
    ftr = row.get('FTR', '')  # Full Time Result
    if ftr == 'H':
        return 'H'
    elif ftr == 'A':
        return 'A'
    else:
        return 'D'


def import_csv_data(csv_text: str, league_name: str, season: str) -> int:
    """Import CSV data into database."""
    if not csv_text:
        return 0
    
    count = 0
    reader = csv.DictReader(io.StringIO(csv_text))
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for row in reader:
            try:
                # Parse date
                date_str = row.get('Date', '')
                if '/' in date_str:
                    # Handle DD/MM/YYYY or DD/MM/YY format
                    parts = date_str.split('/')
                    if len(parts[2]) == 2:
                        date_str = f"{parts[0]}/{parts[1]}/20{parts[2]}"
                    try:
                        match_date = datetime.strptime(date_str, '%d/%m/%Y').isoformat()
                    except:
                        match_date = None
                else:
                    match_date = None
                
                # Get odds (B365 = Bet365, common source)
                home_odds = float(row.get('B365H') or row.get('BWH') or row.get('PSH') or 0) or None
                draw_odds = float(row.get('B365D') or row.get('BWD') or row.get('PSD') or 0) or None
                away_odds = float(row.get('B365A') or row.get('BWA') or row.get('PSA') or 0) or None
                
                # Get goals
                home_goals = int(row.get('FTHG', 0) or 0)
                away_goals = int(row.get('FTAG', 0) or 0)
                
                # Teams
                home_team = row.get('HomeTeam', '')
                away_team = row.get('AwayTeam', '')
                
                if not home_team or not away_team:
                    continue
                
                # Fixture ID
                fixture_id = f"fdc_{season}_{home_team[:3]}_{away_team[:3]}_{date_str}"
                
                cursor.execute("""
                    INSERT OR IGNORE INTO matches 
                    (fixture_id, home_team, away_team, match_date, league, season,
                     home_goals, away_goals, home_odds, draw_odds, away_odds, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fixture_id, home_team, away_team, match_date, league_name, f"20{season[:2]}-20{season[2:]}",
                    home_goals, away_goals, home_odds, draw_odds, away_odds, parse_result(row)
                ))
                count += 1
                
            except Exception as e:
                pass
        
        conn.commit()
    
    return count


def main():
    print("=" * 60)
    print("DOWNLOADING HISTORICAL DATA WITH ODDS")
    print("Source: football-data.co.uk (FREE)")
    print("=" * 60)
    
    total = 0
    
    for season in SEASONS:
        print(f"\nSeason 20{season[:2]}/20{season[2:]}:")
        
        for league_name, league_code in LEAGUES.items():
            print(f"  {league_name}...", end=" ")
            
            csv_data = download_csv(season, league_code)
            if csv_data:
                count = import_csv_data(csv_data, league_name, season)
                print(f"{count} matches")
                total += count
            else:
                print("not available")
    
    print(f"\n{'=' * 60}")
    print(f"Total imported: {total} matches")
    
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

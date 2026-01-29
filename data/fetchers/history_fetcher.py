"""
Historical Match Fetcher
Fetches finished matches from Football-Data.org to build a history dataset.
Prerequisite for:
- Opponent Adjusted Form (calculating strength of opponents)
- Backtesting Engine (simulating past performance)
"""
import os
import time
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Fix python path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.database import get_db

logger = logging.getLogger(__name__)

# Football-Data.org league codes
LEAGUE_CODES = {
    'Premier League': 'PL',
    'Bundesliga': 'BL1', 
    'La Liga': 'PD',
    'Serie A': 'SA',
    'Ligue 1': 'FL1'
}

# Rate limit configuration
RATE_LIMIT_DELAY = 6.5  # seconds

class HistoryFetcher:
    """Fetch and store historical match data."""
    
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_DATA_API_KEY', '')
        self.base_url = 'https://api.football-data.org/v4'
        self.last_call_time = 0
    
    def _rate_limit(self):
        """Enforce rate limit."""
        elapsed = time.time() - self.last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = time.time()
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make API request with rate limiting."""
        if not self.api_key:
            logger.error("API key not configured")
            return None
            
        self._rate_limit()
        
        headers = {'X-Auth-Token': self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit hit, waiting 60s...")
                time.sleep(60)
                return self._make_request(endpoint)
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def fetch_season_matches(self, league_name: str, season: int = 2025) -> int:
        """Fetch all finished matches for a specific league/season."""
        code = LEAGUE_CODES.get(league_name)
        if not code:
            logger.error(f"Unknown league: {league_name}")
            return 0
            
        logger.info(f"Fetching history for {league_name}...")
        
        # Determine season start year (e.g. 2024 for 24/25 season)
        # Football-Data.org uses the start year for 'season'
        # For 2025/2026 season, use 2025? No, usually it's current year dependent.
        # Let's try fetching current 'season' matches.
        
        endpoint = f"/competitions/{code}/matches?status=FINISHED"
        data = self._make_request(endpoint)
        
        if not data or 'matches' not in data:
            return 0
            
        matches = data['matches']
        saved_count = self._save_matches(matches, league_name)
        logger.info(f"Saved {saved_count} historical matches for {league_name}")
        return saved_count
        
    def _save_matches(self, matches: List[Dict], league_name: str) -> int:
        """Save matches to database."""
        count = 0
        with get_db() as conn:
            cursor = conn.cursor()
            
            for m in matches:
                try:
                    fixture_id = str(m.get('id'))
                    match_date = m.get('utcDate')
                    
                    # Score
                    score = m.get('score', {}).get('fullTime', {})
                    home_goals = score.get('home')
                    away_goals = score.get('away')
                    
                    # Result
                    winner = m.get('score', {}).get('winner')
                    result = 'D'
                    if winner == 'HOME_TEAM': result = 'H'
                    elif winner == 'AWAY_TEAM': result = 'A'
                    
                    # Team names
                    home_team = m.get('homeTeam', {}).get('name')
                    away_team = m.get('awayTeam', {}).get('name')
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO matches (
                            fixture_id, home_team, away_team,
                            match_date, league, season,
                            home_goals, away_goals, result
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fixture_id, home_team, away_team,
                        match_date, league_name, '2024-2025', # Assuming current season
                        home_goals, away_goals, result
                    ))
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to save match {fixture_id}: {e}")
            
            conn.commit()
        return count

    def fetch_all_leagues(self):
        """Fetch history for all supported leagues."""
        total = 0
        for league in LEAGUE_CODES.keys():
            total += self.fetch_season_matches(league)
        return total

def fetch_history_now():
    """Start fetching history immediately."""
    fetcher = HistoryFetcher()
    total = fetcher.fetch_all_leagues()
    print(f"Total historical matches saved: {total}")
    return total

if __name__ == "__main__":
    fetch_history_now()

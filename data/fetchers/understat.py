"""
Data fetcher for Understat (xG data).
Uses web scraping as Understat doesn't have a public API.
"""
import json
import re
import requests
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://understat.com"

LEAGUE_MAP = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
    "Russian Premier League": "RFPL"
}


class UnderstatFetcher:
    """Fetches xG data from Understat via web scraping."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_league_data(
        self,
        league: str = "Premier League",
        season: str = "2024"
    ) -> Optional[Dict]:
        """
        Fetch league data including xG for all teams.
        
        Args:
            league: League name
            season: Year (e.g., "2024" for 2024-25 season)
            
        Returns:
            Dictionary with team xG data
        """
        league_code = LEAGUE_MAP.get(league)
        if not league_code:
            logger.error(f"Unknown league: {league}")
            return None
        
        url = f"{BASE_URL}/league/{league_code}/{season}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract JSON data from script tags
            data = self._extract_json(response.text, 'teamsData')
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch Understat data: {e}")
            return None
    
    def get_team_xg(
        self,
        team: str,
        league: str = "Premier League",
        season: str = "2024",
        last_n: int = 5
    ) -> Dict[str, float]:
        """
        Get xG averages for a team.
        
        Returns:
            Dict with 'xg_for' and 'xg_against' averages
        """
        data = self.get_league_data(league, season)
        
        if not data:
            return {'xg_for': 1.5, 'xg_against': 1.0}  # Defaults
        
        for team_id, team_data in data.items():
            if team.lower() in team_data.get('title', '').lower():
                history = team_data.get('history', [])
                
                if history:
                    recent = history[-last_n:]
                    xg_for = sum(float(m.get('xG', 0)) for m in recent) / len(recent)
                    xg_against = sum(float(m.get('xGA', 0)) for m in recent) / len(recent)
                    
                    return {
                        'xg_for': round(xg_for, 2),
                        'xg_against': round(xg_against, 2)
                    }
        
        return {'xg_for': 1.5, 'xg_against': 1.0}
    
    def get_match_xg(
        self,
        match_id: str
    ) -> Optional[Dict]:
        """Get xG for a specific match."""
        url = f"{BASE_URL}/match/{match_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = self._extract_json(response.text, 'shotsData')
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch match xG: {e}")
            return None
    
    def _extract_json(self, html: str, var_name: str) -> Optional[Dict]:
        """Extract JSON data from JavaScript variable in HTML."""
        pattern = rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        
        if match:
            json_str = match.group(1)
            # Unescape
            json_str = json_str.encode().decode('unicode_escape')
            return json.loads(json_str)
        
        return None


# Demo data when scraping is unavailable
DEMO_XG_DATA = {
    "Manchester City": {"xg_for": 2.4, "xg_against": 0.9},
    "Arsenal": {"xg_for": 2.1, "xg_against": 1.0},
    "Liverpool": {"xg_for": 2.3, "xg_against": 1.1},
    "Chelsea": {"xg_for": 1.8, "xg_against": 1.3},
    "Manchester United": {"xg_for": 1.6, "xg_against": 1.4},
    "Tottenham": {"xg_for": 1.9, "xg_against": 1.2},
    "Newcastle": {"xg_for": 1.7, "xg_against": 1.0},
    "Brighton": {"xg_for": 1.5, "xg_against": 1.2},
    "Aston Villa": {"xg_for": 1.8, "xg_against": 1.1},
    "West Ham": {"xg_for": 1.4, "xg_against": 1.5}
}


def get_demo_xg(team: str) -> Dict[str, float]:
    """Get demo xG data for testing."""
    for key, value in DEMO_XG_DATA.items():
        if team.lower() in key.lower():
            return value
    return {"xg_for": 1.5, "xg_against": 1.2}


# Singleton
_fetcher = None

def get_understat() -> UnderstatFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = UnderstatFetcher()
    return _fetcher

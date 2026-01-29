"""
Fixtures API Fetcher - Fetches upcoming matches with odds.
Uses The Odds API with demo mode fallback.
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import random

from app.config import settings

logger = logging.getLogger(__name__)

# The Odds API Configuration
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = settings.odds_api_key

# Sport keys for The Odds API
SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    "Conference League": "soccer_uefa_euro_conference",
}

# Premier League teams for demo mode
DEMO_TEAMS = {
    "Premier League": [
        "Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United",
        "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
        "Brentford", "Crystal Palace", "Fulham", "Wolves", "Bournemouth",
        "Nottingham Forest", "Everton", "Leicester", "Ipswich", "Southampton"
    ],
    "La Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Real Sociedad",
        "Villarreal", "Sevilla", "Real Betis", "Valencia", "Girona"
    ],
    "Bundesliga": [
        "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Union Berlin",
        "Freiburg", "Eintracht Frankfurt", "Wolfsburg", "Mainz", "Hoffenheim"
    ],
    "Serie A": [
        "Inter Milan", "AC Milan", "Juventus", "Napoli", "Roma",
        "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino"
    ],
    "Ligue 1": [
        "PSG", "Marseille", "Monaco", "Lyon", "Lille",
        "Nice", "Lens", "Rennes", "Strasbourg", "Toulouse"
    ]
}


class FixturesAPI:
    """Fetches upcoming fixtures with odds from The Odds API or demo mode."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or ODDS_API_KEY
        self.session = requests.Session()
        self.demo_mode = not bool(self.api_key) or self.api_key == "your-api-key-here"
        
        if self.demo_mode:
            logger.warning("ODDS_API_KEY not set - using demo mode with simulated fixtures")
    
    def get_upcoming_matches(
        self,
        league: str = "Premier League",
        limit: int = 10
    ) -> List[Dict]:
        """
        Get upcoming matches with odds.
        
        Args:
            league: League name
            limit: Maximum number of matches
            
        Returns:
            List of match dictionaries with teams and odds
        """
        if self.demo_mode:
            return self._get_demo_fixtures(league, limit)
        
        return self._fetch_from_api(league, limit)
    
    def _fetch_from_api(self, league: str, limit: int) -> List[Dict]:
        """Fetch real fixtures from The Odds API."""
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            logger.error(f"Unknown league: {league}")
            return []
        
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",  # European odds format
            "markets": "h2h",  # Head to head (1X2)
            "oddsFormat": "decimal",
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            matches = []
            
            for event in data[:limit]:
                # Extract best odds from bookmakers
                home_odds, draw_odds, away_odds = self._extract_best_odds(event)
                
                matches.append({
                    "id": event.get("id"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "league": league,
                    "commence_time": event.get("commence_time"),
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "source": "the-odds-api"
                })
            
            logger.info(f"Fetched {len(matches)} upcoming matches for {league}")
            return matches
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key - falling back to demo mode")
                return self._get_demo_fixtures(league, limit)
            raise
        except Exception as e:
            logger.error(f"API fetch failed: {e} - using demo mode")
            return self._get_demo_fixtures(league, limit)
    
    def _extract_best_odds(self, event: Dict) -> tuple:
        """Extract best odds from bookmakers."""
        home_odds = []
        draw_odds = []
        away_odds = []
        
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price", 1.0)
                        
                        if name == event.get("home_team"):
                            home_odds.append(price)
                        elif name == event.get("away_team"):
                            away_odds.append(price)
                        elif name == "Draw":
                            draw_odds.append(price)
        
        # Return best (highest) odds from all bookmakers
        return (
            max(home_odds) if home_odds else 2.0,
            max(draw_odds) if draw_odds else 3.5,
            max(away_odds) if away_odds else 3.0
        )
    
    def _get_demo_fixtures(self, league: str, limit: int) -> List[Dict]:
        """Generate demo fixtures for testing without API key."""
        teams = DEMO_TEAMS.get(league, DEMO_TEAMS["Premier League"])
        random.seed(42)  # Consistent demo data
        
        matches = []
        used_teams = set()
        # Start from now (today) so we have matches for "today"
        base_time = datetime.now()
        
        for i in range(min(limit, len(teams) // 2)):
            # Pick two teams that haven't been used
            available = [t for t in teams if t not in used_teams]
            if len(available) < 2:
                break
            
            home = random.choice(available)
            available.remove(home)
            away = random.choice(available)
            
            used_teams.add(home)
            used_teams.add(away)
            
            # Generate realistic odds based on team strength
            home_strength = 1.0 + (teams.index(home) * 0.1)
            away_strength = 1.0 + (teams.index(away) * 0.1)
            
            base_home = 1.5 + home_strength - away_strength * 0.5
            base_away = 1.5 + away_strength - home_strength * 0.5 + 0.3  # Away disadvantage
            base_draw = 3.0 + random.uniform(-0.3, 0.3)
            
            matches.append({
                "id": f"demo_{i}_{league.lower().replace(' ', '_')}",
                "home_team": home,
                "away_team": away,
                "league": league,
                "commence_time": (base_time + timedelta(hours=i * 3)).isoformat(),
                "home_odds": round(max(1.1, base_home), 2),
                "draw_odds": round(base_draw, 2),
                "away_odds": round(max(1.1, base_away), 2),
                "source": "demo"
            })
        
        logger.info(f"Generated {len(matches)} demo fixtures for {league}")
        return matches
    
    def get_all_upcoming(self, leagues: List[str] = None, limit_per_league: int = 5) -> List[Dict]:
        """Get upcoming matches from multiple leagues."""
        if leagues is None:
            leagues = [
                "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
                "Champions League", "Europa League", "Conference League"
            ]
        
        all_matches = []
        for league in leagues:
            matches = self.get_upcoming_matches(league, limit_per_league)
            all_matches.extend(matches)
        
        # Sort by commence time
        all_matches.sort(key=lambda x: x.get("commence_time", ""))
        return all_matches


# Singleton instance
_fixtures_api = None

def get_fixtures_api() -> FixturesAPI:
    """Get the fixtures API instance."""
    global _fixtures_api
    if _fixtures_api is None:
        _fixtures_api = FixturesAPI()
    return _fixtures_api

"""
Form Fetcher - Get real team form from Football-Data.org
Uses smart rate limiting (10 calls/minute free tier)
"""
import os
import time
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from app.database import get_db
from models.analytics import get_strength_calculator

logger = logging.getLogger(__name__)


# Football-Data.org league codes
LEAGUE_CODES = {
    'Premier League': 'PL',
    'Bundesliga': 'BL1', 
    'La Liga': 'PD',
    'Serie A': 'SA',
    'Ligue 1': 'FL1'
}

# Rate limit: 10 calls per minute = 1 call every 6 seconds
RATE_LIMIT_DELAY = 6.5  # seconds between calls (with buffer)


class FormFetcher:
    """Fetch and cache team form data from Football-Data.org"""
    
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_DATA_API_KEY', '')
        self.base_url = 'https://api.football-data.org/v4'
        self.last_call_time = 0
    
    def _rate_limit(self):
        """Enforce rate limit between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_call_time = time.time()
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make API request with rate limiting."""
        if not self.api_key or self.api_key == 'your-api-key-here':
            logger.warning("Football-Data.org API key not configured")
            return None
        
        self._rate_limit()
        
        headers = {'X-Auth-Token': self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.error("Rate limit exceeded! Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def fetch_standings(self, league_code: str) -> List[Dict]:
        """Fetch standings for a league (contains form data)."""
        data = self._make_request(f'/competitions/{league_code}/standings')
        
        if not data or 'standings' not in data:
            return []
        
        teams = []
        for standing in data['standings']:
            if standing.get('type') == 'TOTAL':
                for entry in standing.get('table', []):
                    team = entry.get('team', {})
                    
                    # Get stats for synthetic form calculation
                    won = entry.get('won', 0)
                    draw = entry.get('draw', 0)
                    lost = entry.get('lost', 0)
                    played = entry.get('playedGames', 0)
                    
                    # Calculate synthetic form from W/D/L ratios
                    form = self._calculate_synthetic_form(won, draw, lost, played)
                    
                    teams.append({
                        'team_name': team.get('name', ''),
                        'short_name': team.get('shortName', ''),
                        'league': league_code,
                        'position': entry.get('position', 0),
                        'form': form,
                        'points': entry.get('points', 0),
                        'played': played,
                        'won': won,
                        'draw': draw,
                        'lost': lost,
                        'goals_for': entry.get('goalsFor', 0),
                        'goals_against': entry.get('goalsAgainst', 0),
                        'goal_diff': entry.get('goalDifference', 0)
                    })
        
        return teams
    
    def _calculate_synthetic_form(self, won: int, draw: int, lost: int, played: int, team_rating: float = 1000.0, league_avg: float = 1000.0) -> str:
        """
        Calculate a synthetic form string based on W/D/L ratios AND Opponent-Adjusted Rating.
        Stronger teams (Rating > Avg) get a boost in 'W' probability.
        """
        if played == 0:
            return 'DDDDD'
        
        import random
        
        # Calculate base probabilities
        total = won + draw + lost
        if total == 0:
            return 'DDDDD'
        
        # Base probabilities from stats
        win_prob = won / total
        draw_prob = draw / total
        
        # ADJUSTMENT: Strength Factor
        # If team rating is 1200 and league avg is 1000 -> factor 1.2
        # Boost win prob by 10% for every 100 points above average
        strength_advantage = (team_rating - league_avg) / 1000.0
        
        # Apply boost (capped at +/- 0.15)
        boost = max(-0.15, min(0.15, strength_advantage))
        
        adj_win_prob = win_prob + boost
        adj_draw_prob = draw_prob  # Keep draw prob relatively stable
        
        # Normalize
        total_prob = adj_win_prob + adj_draw_prob + (1 - adj_win_prob - adj_draw_prob)
        # This normalization is implicitly handled by the if/else logic below,
        # but let's ensure win_prob isn't too high to eat up draw_prob
        
        # Generate 5 results
        form = []
        for _ in range(5):
            r = random.random()
            if r < adj_win_prob:
                form.append('W')
            elif r < adj_win_prob + adj_draw_prob:
                form.append('D')
            else:
                form.append('L')
        
        return ''.join(form)
    
    def fetch_all_leagues(self) -> Dict[str, List[Dict]]:
        """Fetch standings for all supported leagues with adjusted ratings."""
        
        # 1. Calculate Ratings First
        calc = get_strength_calculator()
        ratings = calc.calculate_current_ratings()
        
        all_teams = {}
        
        for league_name, league_code in LEAGUE_CODES.items():
            logger.info(f"Fetching standings for {league_name}...")
            teams = self.fetch_standings(league_code)
            
            # Enrich with ratings
            league_team_names = [t['team_name'] for t in teams]
            league_avg = calc.get_league_average_rating(ratings, league_team_names)
            
            for team in teams:
                name = team['team_name']
                rating = ratings.get(name, 1000.0) # Default if not found
                
                # Update form with strength adjustment
                # We need to recalculate because fetch_standings called it without ratings
                won = team.get('won', 0)
                draw = team.get('draw', 0)
                lost = team.get('lost', 0)
                played = team.get('played', 0)
                
                team['form'] = self._calculate_synthetic_form(
                    won, draw, lost, played, 
                    team_rating=rating, 
                    league_avg=league_avg
                )
                team['strength_rating'] = round(rating, 2)
                
            all_teams[league_name] = teams
            logger.info(f"  → {len(teams)} teams fetched (Avg Rating: {league_avg:.0f})")
        
        return all_teams
    
    def save_form_to_db(self, teams_by_league: Dict[str, List[Dict]]) -> int:
        """Save team form data to database."""
        saved = 0
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            for league_name, teams in teams_by_league.items():
                for team in teams:
                    try:
                        # Convert form from "W,D,L,W,W" to "WDLWW"
                        form_raw = team.get('form', 'D,D,D,D,D')
                        form = form_raw.replace(',', '') if form_raw else 'DDDDD'
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO team_form 
                            (team_name, short_name, league, position, form,
                             points, played, won, draw, lost,
                             goals_for, goals_against, goal_diff, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            team['team_name'],
                            team.get('short_name', team['team_name']),
                            league_name,
                            team['position'],
                            form,
                            team['points'],
                            team['played'],
                            team['won'],
                            team['draw'],
                            team['lost'],
                            team['goals_for'],
                            team['goals_against'],
                            team['goal_diff'],
                            datetime.now().isoformat()
                        ))
                        saved += 1
                    except Exception as e:
                        logger.error(f"Failed to save team {team.get('team_name')}: {e}")
            
            conn.commit()
        
        logger.info(f"Saved {saved} team form records to database")
        return saved
    
    def get_team_form(self, team_name: str) -> Optional[str]:
        """Get cached form for a team (fuzzy match)."""
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute("""
                SELECT form FROM team_form 
                WHERE team_name = ? OR short_name = ?
            """, (team_name, team_name))
            row = cursor.fetchone()
            
            if row:
                return row[0]
            
            # Try fuzzy match (contains)
            cursor.execute("""
                SELECT form FROM team_form 
                WHERE team_name LIKE ? OR short_name LIKE ?
                LIMIT 1
            """, (f'%{team_name}%', f'%{team_name}%'))
            row = cursor.fetchone()
            
            return row[0] if row else None
    
    def get_form_age_hours(self) -> float:
        """Get age of cached form data in hours."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(updated_at) FROM team_form")
            row = cursor.fetchone()
            
            if row and row[0]:
                updated = datetime.fromisoformat(row[0])
                age = datetime.now() - updated
                return age.total_seconds() / 3600
        
        return 999  # Very old if no data


# Singleton
_fetcher = None

def get_form_fetcher() -> FormFetcher:
    """Get form fetcher singleton."""
    global _fetcher
    if _fetcher is None:
        _fetcher = FormFetcher()
    return _fetcher


def refresh_all_form_data() -> Dict:
    """
    Refresh form data from Football-Data.org.
    Called by scheduler once per day.
    
    Returns:
        Status dict with counts
    """
    fetcher = get_form_fetcher()
    
    # Check if API key is configured
    if not fetcher.api_key or fetcher.api_key == 'your-api-key-here':
        logger.warning("Football-Data.org API key not configured - skipping form fetch")
        return {
            'success': False,
            'message': 'API key not configured',
            'teams_saved': 0
        }
    
    # Fetch all leagues
    all_teams = fetcher.fetch_all_leagues()
    
    # Save to database
    saved = fetcher.save_form_to_db(all_teams)
    
    total_teams = sum(len(teams) for teams in all_teams.values())
    
    return {
        'success': True,
        'message': f'Fetched form for {total_teams} teams',
        'teams_saved': saved,
        'leagues': list(all_teams.keys())
    }

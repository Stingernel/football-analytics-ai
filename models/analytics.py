"""
Analytics Engine
Handles advanced calculations for team strength and form analysis.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import statistics

from app.database import get_db

logger = logging.getLogger(__name__)

class StrengthCalculator:
    """
    Calculates team strength ratings based on historical match results.
    Uses a simplified Elo-like rating system or relative strength index.
    """
    
    def __init__(self):
        self.base_rating = 1000.0
        self.k_factor = 32  # Impact of a single match
    
    def calculate_current_ratings(self) -> Dict[str, float]:
        """
        Replay history to calculate current team ratings.
        Returns: Dict {team_name: rating}
        """
        ratings = {}
        
        with get_db() as conn:
            cursor = conn.cursor()
            # Fetch all finished matches ordered by date
            cursor.execute("""
                SELECT home_team, away_team, result, match_date 
                FROM matches 
                ORDER BY match_date ASC
            """)
            matches = cursor.fetchall()
            
            # Initialize ratings for all teams seen
            for m in matches:
                home, away = m['home_team'], m['away_team']
                if home not in ratings: ratings[home] = self.base_rating
                if away not in ratings: ratings[away] = self.base_rating
                
                # Calculate expected score (Elo formula)
                # P(A) = 1 / (1 + 10^((RatingB - RatingA) / 400))
                ra = ratings[home]
                rb = ratings[away]
                
                expected_home = 1 / (1 + 10 ** ((rb - ra) / 400))
                expected_away = 1 / (1 + 10 ** ((ra - rb) / 400))
                
                # Actual score (1=Win, 0.5=Draw, 0=Loss)
                result = m['result']
                if result == 'H':
                    actual_home, actual_away = 1.0, 0.0
                elif result == 'D':
                    actual_home, actual_away = 0.5, 0.5
                else: # Away win
                    actual_home, actual_away = 0.0, 1.0
                
                # Update ratings
                ratings[home] = ra + self.k_factor * (actual_home - expected_home)
                ratings[away] = rb + self.k_factor * (actual_away - expected_away)
                
        return ratings
    
    def get_league_average_rating(self, ratings: Dict[str, float], league_teams: List[str]) -> float:
        """Calculate average rating for specific league teams."""
        if not league_teams:
            return self.base_rating
            
        league_ratings = [ratings.get(t, self.base_rating) for t in league_teams]
        return statistics.mean(league_ratings)


# Singleton
_calculator = None

def get_strength_calculator() -> StrengthCalculator:
    global _calculator
    if _calculator is None:
        _calculator = StrengthCalculator()
    return _calculator


class MarketAnalyzer:
    """
    Analyzes odds movement to detect sharp money and market sentiment.
    """
    
    def detect_dropping_odds(self, threshold_pct: float = 0.05) -> List[Dict]:
        """
        Detect matches where odds have dropped significantly (> 5%).
        This often indicates 'Sharp Money' or insider information.
        
        Args:
            threshold_pct: Drop threshold (0.05 = 5%)
            
        Returns:
            List of matches with significant movement
        """
        results = []
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get latest odds for upcoming matches
            cursor.execute("""
                SELECT id, home_team, away_team, home_odds, draw_odds, away_odds, fetched_at 
                FROM fixtures 
                WHERE DATE(match_date) >= DATE('now')
            """)
            fixtures = cursor.fetchall()
            
            for fix in fixtures:
                fix_id = fix['id']
                
                # Get opening odds (first recorded for this fixture)
                cursor.execute("""
                    SELECT home_odds, draw_odds, away_odds, recorded_at 
                    FROM odds_history 
                    WHERE fixture_id = ? 
                    ORDER BY recorded_at ASC 
                    LIMIT 1
                """, (fix_id,))
                opening = cursor.fetchone()
                
                if not opening:
                    continue
                    
                current_home = fix['home_odds'] or 2.0
                opening_home = opening['home_odds'] or 2.0
                
                current_away = fix['away_odds'] or 2.0
                opening_away = opening['away_odds'] or 2.0
                
                # Calculate movement
                # Drop = (Opening - Current) / Opening
                # Example: Open 2.0 -> Current 1.8 = (2.0 - 1.8) / 2.0 = 0.10 (10% drop, Strong Signal)
                
                home_drop = (opening_home - current_home) / opening_home
                away_drop = (opening_away - current_away) / opening_away
                
                alert_type = None
                drop_val = 0
                
                if home_drop > threshold_pct:
                    alert_type = 'HOME_DROP'
                    drop_val = home_drop
                elif away_drop > threshold_pct:
                    alert_type = 'AWAY_DROP'
                    drop_val = away_drop
                    
                if alert_type:
                    results.append({
                        'fixture_id': fix_id,
                        'home_team': fix['home_team'],
                        'away_team': fix['away_team'],
                        'alert_type': alert_type,
                        'drop_percent': round(drop_val * 100, 1),
                        'opening_odds': opening_home if alert_type == 'HOME_DROP' else opening_away,
                        'current_odds': current_home if alert_type == 'HOME_DROP' else current_away,
                        'market_signal': 'SHARP_MONEY' if drop_val > 0.10 else 'MODERATE_MOVE'
                    })
                    
        return sorted(results, key=lambda x: x['drop_percent'], reverse=True)

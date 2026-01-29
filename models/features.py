"""
Professional Feature Engineering for Football Match Prediction.
Features used by professional bookmakers and betting syndicates.
"""
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Professional-grade feature extraction for match prediction."""
    
    # Home advantage factors by league (based on historical data)
    HOME_ADVANTAGE = {
        "Premier League": 0.10,
        "La Liga": 0.12,
        "Bundesliga": 0.08,
        "Serie A": 0.11,
        "Ligue 1": 0.09,
        "default": 0.10
    }
    
    @staticmethod
    def form_to_weighted_score(form: str, weights: List[float] = None) -> float:
        """
        Convert form string to weighted score.
        Recent matches weighted more heavily (professional approach).
        
        Args:
            form: Form string e.g., "WWDLW" (newest first)
            weights: Custom weights, default is exponential decay
            
        Returns:
            Normalized score 0-1
        """
        if not form:
            return 0.5  # Neutral
        
        # Default weights: exponential decay (recent matters more)
        if weights is None:
            weights = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        score = 0.0
        for i, char in enumerate(form.upper()[:5]):
            weight = weights[i] if i < len(weights) else 0.05
            if char == 'W':
                score += weight * 1.0
            elif char == 'D':
                score += weight * 0.4
            # 'L' adds 0
        
        return min(1.0, score)
    
    @staticmethod
    def calculate_streak(form: str, streak_type: str = 'win') -> int:
        """
        Calculate current streak from form.
        
        Args:
            form: Form string (newest first)
            streak_type: 'win', 'unbeaten', or 'losing'
        """
        if not form:
            return 0
        
        streak = 0
        for char in form.upper():
            if streak_type == 'win':
                if char == 'W':
                    streak += 1
                else:
                    break
            elif streak_type == 'unbeaten':
                if char in ('W', 'D'):
                    streak += 1
                else:
                    break
            elif streak_type == 'losing':
                if char == 'L':
                    streak += 1
                else:
                    break
        
        return streak
    
    @staticmethod
    def odds_to_probabilities(home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        """
        Convert odds to true probabilities (remove overround/margin).
        
        Returns fair probabilities without bookmaker margin.
        """
        if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
            return {'home': 0.40, 'draw': 0.28, 'away': 0.32}
        
        # Implied probabilities
        home_implied = 1 / home_odds
        draw_implied = 1 / draw_odds
        away_implied = 1 / away_odds
        
        # Total (overround, usually ~105-110%)
        total = home_implied + draw_implied + away_implied
        overround = total - 1.0  # Bookmaker margin
        
        # Fair probabilities (remove margin proportionally)
        return {
            'home': home_implied / total,
            'draw': draw_implied / total,
            'away': away_implied / total,
            'overround': overround
        }
    
    @staticmethod
    def calculate_xg_advantage(home_xg_for: float, home_xg_against: float,
                               away_xg_for: float, away_xg_against: float) -> Dict[str, float]:
        """
        Calculate xG-based advantages.
        
        Returns normalized advantages for prediction.
        """
        home_attack = home_xg_for / 3.0  # Normalize to 0-1 range
        home_defense = 1 - (home_xg_against / 3.0)  # Lower is better
        away_attack = away_xg_for / 3.0
        away_defense = 1 - (away_xg_against / 3.0)
        
        return {
            'home_attack_rating': min(1.0, home_attack),
            'home_defense_rating': min(1.0, max(0, home_defense)),
            'away_attack_rating': min(1.0, away_attack),
            'away_defense_rating': min(1.0, max(0, away_defense)),
            'home_net_xg': (home_xg_for - home_xg_against) / 3.0,
            'away_net_xg': (away_xg_for - away_xg_against) / 3.0
        }
    
    def extract_features(self, match_data: Dict) -> Dict[str, float]:
        """
        Extract all professional features from match data.
        
        Args:
            match_data: Dictionary with match information
            
        Returns:
            Dictionary of normalized features (0-1 range)
        """
        features = {}
        
        # === Form Features ===
        home_form = match_data.get('home_form', 'DDD')
        away_form = match_data.get('away_form', 'DDD')
        
        features['home_form_weighted'] = self.form_to_weighted_score(home_form)
        features['away_form_weighted'] = self.form_to_weighted_score(away_form)
        features['form_diff'] = features['home_form_weighted'] - features['away_form_weighted']
        
        # === Streak Features ===
        features['home_win_streak'] = min(5, self.calculate_streak(home_form, 'win')) / 5.0
        features['away_win_streak'] = min(5, self.calculate_streak(away_form, 'win')) / 5.0
        features['home_unbeaten_streak'] = min(10, self.calculate_streak(home_form, 'unbeaten')) / 10.0
        features['away_unbeaten_streak'] = min(10, self.calculate_streak(away_form, 'unbeaten')) / 10.0
        
        # === Market/Odds Features ===
        home_odds = match_data.get('home_odds', 2.0)
        draw_odds = match_data.get('draw_odds', 3.5)
        away_odds = match_data.get('away_odds', 4.0)
        
        market_probs = self.odds_to_probabilities(home_odds, draw_odds, away_odds)
        features['market_home_prob'] = market_probs['home']
        features['market_draw_prob'] = market_probs['draw']
        features['market_away_prob'] = market_probs['away']
        features['market_overround'] = min(0.2, market_probs.get('overround', 0.05))
        
        # === xG Features ===
        home_xg = match_data.get('home_xg', 1.5)
        away_xg = match_data.get('away_xg', 1.2)
        home_xg_against = match_data.get('home_xg_against', 1.0)
        away_xg_against = match_data.get('away_xg_against', 1.3)
        
        xg_features = self.calculate_xg_advantage(
            home_xg, home_xg_against, away_xg, away_xg_against
        )
        features.update(xg_features)
        
        # === Home Advantage ===
        league = match_data.get('league', 'default')
        features['home_advantage'] = self.HOME_ADVANTAGE.get(league, 0.10)
        
        # === Head-to-Head ===
        features['h2h_home_wins'] = match_data.get('h2h_home_wins', 0.4)
        features['h2h_draws'] = match_data.get('h2h_draws', 0.25)
        features['h2h_away_wins'] = match_data.get('h2h_away_wins', 0.35)
        
        # === Momentum Score (composite) ===
        home_momentum = (
            features['home_form_weighted'] * 0.4 +
            features['home_win_streak'] * 0.3 +
            features['home_unbeaten_streak'] * 0.3
        )
        away_momentum = (
            features['away_form_weighted'] * 0.4 +
            features['away_win_streak'] * 0.3 +
            features['away_unbeaten_streak'] * 0.3
        )
        features['home_momentum'] = home_momentum
        features['away_momentum'] = away_momentum
        features['momentum_diff'] = home_momentum - away_momentum
        
        return features
    
    def get_feature_vector(self, match_data: Dict) -> np.ndarray:
        """Convert features to numpy array for model input."""
        features = self.extract_features(match_data)
        
        # Define feature order
        feature_order = [
            'home_form_weighted', 'away_form_weighted', 'form_diff',
            'home_win_streak', 'away_win_streak',
            'home_unbeaten_streak', 'away_unbeaten_streak',
            'market_home_prob', 'market_draw_prob', 'market_away_prob',
            'home_attack_rating', 'home_defense_rating',
            'away_attack_rating', 'away_defense_rating',
            'home_net_xg', 'away_net_xg',
            'home_advantage',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
            'home_momentum', 'away_momentum', 'momentum_diff'
        ]
        
        return np.array([features.get(f, 0.5) for f in feature_order])


# Singleton
_engineer = None

def get_feature_engineer() -> FeatureEngineer:
    global _engineer
    if _engineer is None:
        _engineer = FeatureEngineer()
    return _engineer

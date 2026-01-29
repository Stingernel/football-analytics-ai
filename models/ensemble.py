"""
Ensemble predictor combining XGBoost and LSTM models.
"""
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import logging

from app.config import settings
from models.xgboost_model import XGBoostPredictor, DemoXGBoostPredictor
from models.lstm_model import LSTMPredictor, DemoLSTMPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Hybrid ensemble combining XGBoost and LSTM predictions.
    
    Final probabilities: 0.6 * XGBoost + 0.4 * LSTM (configurable)
    """
    
    def __init__(
        self, 
        xgb_weight: float = None,
        lstm_weight: float = None,
        use_demo_models: bool = True
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            xgb_weight: Weight for XGBoost predictions
            lstm_weight: Weight for LSTM predictions
            use_demo_models: Use demo models if trained models not available
        """
        self.xgb_weight = xgb_weight or settings.ensemble_xgb_weight
        self.lstm_weight = lstm_weight or settings.ensemble_lstm_weight
        
        # Normalize weights
        total_weight = self.xgb_weight + self.lstm_weight
        self.xgb_weight /= total_weight
        self.lstm_weight /= total_weight
        
        # Initialize models
        self.xgb_model = None
        self.lstm_model = None
        
        self._load_models(use_demo_models)
    
    def _load_models(self, use_demo: bool = True):
        """Load trained models or fall back to demo."""
        models_dir = settings.models_dir
        
        # Try loading XGBoost
        xgb_path = models_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            try:
                self.xgb_model = XGBoostPredictor(xgb_path)
                logger.info("Loaded trained XGBoost model")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost: {e}")
        
        if self.xgb_model is None and use_demo:
            self.xgb_model = DemoXGBoostPredictor()
            logger.info("Using demo XGBoost model")
        
        # Try loading LSTM
        lstm_path = models_dir / "lstm_model.h5"
        if lstm_path.exists():
            try:
                self.lstm_model = LSTMPredictor(lstm_path)
                logger.info("Loaded trained LSTM model")
            except Exception as e:
                logger.warning(f"Failed to load LSTM: {e}")
        
        if self.lstm_model is None and use_demo:
            self.lstm_model = DemoLSTMPredictor()
            logger.info("Using demo LSTM model")
    
    def predict(self, match_data: Dict) -> Dict:
        """
        Generate ensemble prediction.
        
        Args:
            match_data: Match information dict
            
        Returns:
            Dictionary with:
            - probabilities: Combined probabilities
            - xgb_probs: XGBoost probabilities
            - lstm_probs: LSTM probabilities
            - predicted_result: Most likely outcome
            - confidence: Confidence score
        """
        # Get individual predictions
        xgb_probs = self.xgb_model.predict(match_data) if self.xgb_model else {'H': 0.33, 'D': 0.34, 'A': 0.33}
        lstm_probs = self.lstm_model.predict(match_data) if self.lstm_model else {'H': 0.33, 'D': 0.34, 'A': 0.33}
        
        # Combine with weights
        ensemble_probs = {}
        for outcome in ['H', 'D', 'A']:
            ensemble_probs[outcome] = (
                self.xgb_weight * xgb_probs[outcome] +
                self.lstm_weight * lstm_probs[outcome]
            )
        
        # Normalize (should already be close to 1, but ensure)
        total = sum(ensemble_probs.values())
        ensemble_probs = {k: v/total for k, v in ensemble_probs.items()}
        
        # Determine predicted result
        predicted_result = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[predicted_result]
        
        # Calculate agreement (higher if models agree)
        xgb_pred = max(xgb_probs, key=xgb_probs.get)
        lstm_pred = max(lstm_probs, key=lstm_probs.get)
        models_agree = xgb_pred == lstm_pred == predicted_result
        
        return {
            'probabilities': {
                'home_win': round(ensemble_probs['H'], 4),
                'draw': round(ensemble_probs['D'], 4),
                'away_win': round(ensemble_probs['A'], 4)
            },
            'xgb_probs': {
                'home_win': round(xgb_probs['H'], 4),
                'draw': round(xgb_probs['D'], 4),
                'away_win': round(xgb_probs['A'], 4)
            },
            'lstm_probs': {
                'home_win': round(lstm_probs['H'], 4),
                'draw': round(lstm_probs['D'], 4),
                'away_win': round(lstm_probs['A'], 4)
            },
            'predicted_result': predicted_result,
            'confidence': round(confidence, 4),
            'models_agree': models_agree
        }
    
    def analyze_value_bets(
        self, 
        probabilities: Dict[str, float],
        odds: Dict[str, float],
        confidence: float = 0.5,
        models_agree: bool = True,
        bankroll: float = 1000.0
    ) -> list:
        """
        Professional value bet analysis with Kelly Criterion.
        
        Args:
            probabilities: Model probabilities {home_win, draw, away_win}
            odds: Bookmaker odds {home, draw, away}
            confidence: Model confidence score
            models_agree: Whether models agree on prediction
            bankroll: Total bankroll for stake calculation
            
        Returns:
            List of value bet recommendations with ratings
        """
        from models.stake_calculator import analyze_value_bet_professional
        
        value_bets = []
        
        mapping = [
            ('HOME', 'home_win', 'home'),
            ('DRAW', 'draw', 'draw'),
            ('AWAY', 'away_win', 'away')
        ]
        
        for bet_type, prob_key, odds_key in mapping:
            model_prob = probabilities[prob_key]
            bookmaker_odds = odds.get(odds_key, 0)
            
            if bookmaker_odds <= 1:
                continue
            
            # Use professional analysis
            analysis = analyze_value_bet_professional(
                model_prob=model_prob,
                odds=bookmaker_odds,
                confidence=confidence,
                models_agree=models_agree,
                bankroll=bankroll
            )
            
            value_bets.append({
                'bet_type': bet_type,
                'odds': bookmaker_odds,
                'implied_prob': analysis['implied_prob'],
                'model_prob': analysis['model_prob'],
                'edge': analysis['edge'],
                'edge_pct': analysis['edge_pct'],
                'recommended': analysis['recommended'],
                'stake': analysis['stake'],
                'stake_pct': analysis['stake_pct'],
                'expected_value': analysis['expected_value'],
                'rating': analysis['rating'],
                'reason': analysis['reason']
            })
        
        # Sort by rating stars then edge (best first)
        value_bets.sort(
            key=lambda x: (x['rating']['stars'], x['edge']), 
            reverse=True
        )
        
        return value_bets


# Singleton instance
_ensemble = None


def get_ensemble() -> EnsemblePredictor:
    """Get or create ensemble predictor singleton."""
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsemblePredictor()
    return _ensemble

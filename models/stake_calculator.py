"""
Kelly Criterion Stake Calculator.
Professional stake sizing for optimal bankroll growth.
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Configuration for value bet detection
VALUE_BET_CONFIG = {
    'min_edge': 0.07,           # 7% minimum edge (was 5%)
    'min_confidence': 0.55,     # 55% min model confidence
    'min_odds': 1.30,           # Avoid very low odds
    'max_odds': 5.00,           # Avoid very high odds
    'kelly_fraction': 0.25,     # Quarter Kelly for safety
    'max_stake_pct': 0.10,      # Never bet more than 10% of bankroll
}


def kelly_stake(
    odds: float, 
    win_prob: float, 
    bankroll: float = 1000.0,
    fraction: float = None
) -> Dict:
    """
    Calculate optimal stake using Kelly Criterion.
    
    Kelly % = (bp - q) / b
    where:
        b = odds - 1 (net odds, decimal to fractional)
        p = probability of winning
        q = probability of losing (1-p)
    
    Args:
        odds: Decimal odds (e.g., 2.50)
        win_prob: Model's probability of winning (0-1)
        bankroll: Total bankroll amount
        fraction: Kelly fraction (0.25 = quarter Kelly, safer)
        
    Returns:
        Dictionary with stake recommendation
    """
    fraction = fraction or VALUE_BET_CONFIG['kelly_fraction']
    
    # Validate inputs
    if odds <= 1:
        return {'stake': 0, 'kelly_pct': 0, 'reason': 'Invalid odds'}
    if not 0 < win_prob < 1:
        return {'stake': 0, 'kelly_pct': 0, 'reason': 'Invalid probability'}
    
    # Calculate Kelly percentage
    b = odds - 1  # Net odds (fractional)
    p = win_prob
    q = 1 - p
    
    kelly_pct = (b * p - q) / b
    
    # Apply fraction for safety
    adjusted_kelly = kelly_pct * fraction
    
    # Apply maximum stake limit
    max_stake_pct = VALUE_BET_CONFIG['max_stake_pct']
    final_pct = max(0, min(adjusted_kelly, max_stake_pct))
    
    # Calculate stake amount
    stake = round(final_pct * bankroll, 2)
    
    return {
        'stake': stake,
        'kelly_pct': round(kelly_pct * 100, 2),
        'adjusted_pct': round(final_pct * 100, 2),
        'expected_value': round((win_prob * odds - 1) * 100, 2),  # EV%
        'recommended': kelly_pct > 0
    }


def calculate_confidence_rating(
    confidence: float,
    edge: float,
    models_agree: bool
) -> Dict:
    """
    Calculate prediction quality rating (1-5 stars).
    
    Rating criteria:
    - ⭐⭐⭐⭐⭐ STRONG: Confidence >65%, Edge >10%, Models agree
    - ⭐⭐⭐⭐ GOOD: Confidence >55%, Edge >7%
    - ⭐⭐⭐ FAIR: Confidence >45%
    - ⭐⭐ WEAK: Confidence 35-45%
    - ⭐ SKIP: Too uncertain (<35%)
    
    Args:
        confidence: Model confidence (0-1)
        edge: Value bet edge (0-1)
        models_agree: Whether XGBoost and LSTM agree
        
    Returns:
        Dictionary with rating info
    """
    stars = 1
    label = "SKIP"
    color = "#d63031"  # Red
    
    conf_pct = confidence * 100
    edge_pct = edge * 100
    
    if confidence >= 0.65 and edge >= 0.10 and models_agree:
        stars = 5
        label = "STRONG"
        color = "#00b894"  # Green
    elif confidence >= 0.55 and edge >= 0.07:
        stars = 4
        label = "GOOD"
        color = "#00cec9"  # Teal
    elif confidence >= 0.45:
        stars = 3
        label = "FAIR"
        color = "#fdcb6e"  # Yellow
    elif confidence >= 0.35:
        stars = 2
        label = "WEAK"
        color = "#e17055"  # Orange
    else:
        stars = 1
        label = "SKIP"
        color = "#d63031"  # Red
    
    return {
        'stars': stars,
        'label': label,
        'color': color,
        'stars_display': "⭐" * stars,
        'should_bet': stars >= 3,
        'is_value_bet': edge >= VALUE_BET_CONFIG['min_edge']
    }


def analyze_value_bet_professional(
    model_prob: float,
    odds: float,
    confidence: float,
    models_agree: bool = True,
    bankroll: float = 1000.0
) -> Dict:
    """
    Professional value bet analysis with all metrics.
    
    Args:
        model_prob: Model's predicted probability (0-1)
        odds: Decimal odds
        confidence: Model confidence
        models_agree: Whether models agree on prediction
        bankroll: Total bankroll
        
    Returns:
        Complete value bet analysis
    """
    config = VALUE_BET_CONFIG
    
    # Calculate implied probability
    implied_prob = 1 / odds if odds > 1 else 0.5
    
    # Calculate edge
    edge = model_prob - implied_prob
    
    # Check if meets value bet criteria
    is_value = (
        edge >= config['min_edge'] and
        confidence >= config['min_confidence'] and
        config['min_odds'] <= odds <= config['max_odds']
    )
    
    # Calculate Kelly stake
    kelly_result = kelly_stake(odds, model_prob, bankroll)
    
    # Calculate rating
    rating = calculate_confidence_rating(confidence, edge, models_agree)
    
    # Reason if not a value bet
    reason = None
    if not is_value:
        if edge < config['min_edge']:
            reason = f"Edge too low ({edge*100:.1f}% < {config['min_edge']*100}%)"
        elif confidence < config['min_confidence']:
            reason = f"Low confidence ({confidence*100:.0f}% < {config['min_confidence']*100}%)"
        elif odds < config['min_odds']:
            reason = f"Odds too low ({odds:.2f} < {config['min_odds']})"
        elif odds > config['max_odds']:
            reason = f"Odds too high ({odds:.2f} > {config['max_odds']})"
    
    return {
        'is_value_bet': is_value,
        'model_prob': round(model_prob, 4),
        'implied_prob': round(implied_prob, 4),
        'edge': round(edge, 4),
        'edge_pct': round(edge * 100, 2),
        'odds': odds,
        'stake': kelly_result['stake'],
        'stake_pct': kelly_result['adjusted_pct'],
        'expected_value': kelly_result['expected_value'],
        'rating': rating,
        'reason': reason,
        'recommended': is_value and rating['stars'] >= 3
    }


def get_bet_recommendation(value_bets: list) -> Optional[Dict]:
    """
    Get the best bet recommendation from list of value bets.
    
    Prioritizes by: rating stars > edge > confidence
    """
    valid_bets = [vb for vb in value_bets if vb.get('recommended')]
    
    if not valid_bets:
        return None
    
    # Sort by composite score
    def score(vb):
        return (
            vb['rating']['stars'] * 100 +
            vb['edge'] * 50 +
            vb['model_prob'] * 10
        )
    
    valid_bets.sort(key=score, reverse=True)
    return valid_bets[0]

"""
Fixture Congestion Detection - Adjusts predictions for teams with heavy schedules.
Critical factor for European competition teams (UCL/UEL/UECL).
"""
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Teams in European competitions 2024/25 season
# Update this at start of each season
EUROPEAN_TEAMS = {
    "Champions League": {
        "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Girona"],
        "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Stuttgart"],
        "Serie A": ["Inter Milan", "AC Milan", "Juventus", "Bologna"],
        "Ligue 1": ["PSG", "Monaco", "Brest", "Lille"]
    },
    "Europa League": {
        "Premier League": ["Manchester United", "Tottenham", "West Ham"],
        "La Liga": ["Real Sociedad", "Athletic Bilbao", "Real Betis"],
        "Bundesliga": ["Eintracht Frankfurt", "Hoffenheim", "Freiburg"],
        "Serie A": ["Roma", "Lazio", "Atalanta"],
        "Ligue 1": ["Lyon", "Nice", "Marseille"]
    },
    "Conference League": {
        "Premier League": ["Chelsea", "Brighton"],
        "La Liga": ["Villarreal", "Real Betis"],
        "Bundesliga": ["Union Berlin", "Werder Bremen"],
        "Serie A": ["Fiorentina", "Napoli"],
        "Ligue 1": ["Lens", "Rennes"]
    }
}

# Congestion penalty factors
CONGESTION_FACTORS = {
    # Days since last match -> penalty multiplier
    1: 0.80,  # Played yesterday = -20% performance
    2: 0.85,  # 2 days ago = -15%
    3: 0.92,  # 3 days ago = -8% (typical midweek)
    4: 0.97,  # 4 days ago = -3%
    5: 1.00,  # 5+ days = no penalty
}

# European competition priority (affects domestic focus)
COMPETITION_PRIORITY = {
    "Champions League": 0.95,  # -5% focus on league when in UCL
    "Europa League": 0.97,    # -3% 
    "Conference League": 0.99  # -1%
}


def get_team_european_status(team: str, league: str) -> Optional[str]:
    """
    Check if team is in European competition.
    
    Returns:
        Competition name or None
    """
    for comp, leagues in EUROPEAN_TEAMS.items():
        if league in leagues:
            if team in leagues[league]:
                return comp
    return None


def calculate_congestion_factor(
    team: str,
    league: str,
    days_since_last_match: int = None,
    is_home: bool = True
) -> Dict:
    """
    Calculate congestion adjustment factor for a team.
    
    Args:
        team: Team name
        league: Team's league
        days_since_last_match: Days since last match (None = assume fresh)
        is_home: Whether playing at home
        
    Returns:
        Dict with factor and explanation
    """
    total_factor = 1.0
    reasons = []
    
    # Check European competition
    euro_comp = get_team_european_status(team, league)
    if euro_comp:
        comp_factor = COMPETITION_PRIORITY.get(euro_comp, 1.0)
        total_factor *= comp_factor
        reasons.append(f"{euro_comp} participant ({(1-comp_factor)*100:.0f}% focus penalty)")
    
    # Check fixture congestion
    if days_since_last_match is not None:
        if days_since_last_match <= 4:
            cong_factor = CONGESTION_FACTORS.get(days_since_last_match, 1.0)
            total_factor *= cong_factor
            reasons.append(f"Played {days_since_last_match} days ago ({(1-cong_factor)*100:.0f}% fatigue)")
    
    # Away penalty for European teams (travel fatigue)
    if euro_comp and not is_home and days_since_last_match and days_since_last_match <= 4:
        travel_penalty = 0.97  # Extra -3% for away after European travel
        total_factor *= travel_penalty
        reasons.append("Away after European match (-3%)")
    
    return {
        'team': team,
        'factor': round(total_factor, 3),
        'european_competition': euro_comp,
        'penalty_percent': round((1 - total_factor) * 100, 1),
        'reasons': reasons,
        'should_warn': total_factor < 0.95
    }


def adjust_prediction_for_congestion(
    home_team: str,
    away_team: str,
    league: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_days_rest: int = None,
    away_days_rest: int = None
) -> Dict:
    """
    Adjust prediction probabilities based on fixture congestion.
    
    Returns:
        Adjusted probabilities and explanation
    """
    # Get congestion factors
    home_cong = calculate_congestion_factor(home_team, league, home_days_rest, is_home=True)
    away_cong = calculate_congestion_factor(away_team, league, away_days_rest, is_home=False)
    
    # Calculate relative advantage change
    # If home team more congested, shift probability towards draw/away
    home_factor = home_cong['factor']
    away_factor = away_cong['factor']
    
    # Net adjustment (positive = favors home, negative = favors away)
    relative_diff = home_factor - away_factor
    
    # Adjust probabilities
    adjustment = relative_diff * 0.15  # Max 15% swing based on congestion
    
    adjusted_home = max(0.05, min(0.90, home_prob + adjustment))
    adjusted_away = max(0.05, min(0.90, away_prob - adjustment))
    adjusted_draw = 1.0 - adjusted_home - adjusted_away
    adjusted_draw = max(0.10, min(0.45, adjusted_draw))
    
    # Normalize
    total = adjusted_home + adjusted_draw + adjusted_away
    adjusted_home /= total
    adjusted_draw /= total  
    adjusted_away /= total
    
    # Build warnings
    warnings = []
    if home_cong['should_warn']:
        warnings.append(f"⚠️ {home_team}: {', '.join(home_cong['reasons'])}")
    if away_cong['should_warn']:
        warnings.append(f"⚠️ {away_team}: {', '.join(away_cong['reasons'])}")
    
    return {
        'original': {
            'home': round(home_prob, 4),
            'draw': round(draw_prob, 4),
            'away': round(away_prob, 4)
        },
        'adjusted': {
            'home': round(adjusted_home, 4),
            'draw': round(adjusted_draw, 4),
            'away': round(adjusted_away, 4)
        },
        'home_congestion': home_cong,
        'away_congestion': away_cong,
        'warnings': warnings,
        'has_congestion_impact': len(warnings) > 0
    }


def get_congestion_badge(team: str, league: str) -> Optional[Dict]:
    """
    Get a badge/indicator for team congestion status.
    For display in dashboard.
    """
    euro_comp = get_team_european_status(team, league)
    
    if euro_comp:
        badges = {
            "Champions League": {"emoji": "🏆", "color": "#0a3d62", "text": "UCL"},
            "Europa League": {"emoji": "🌍", "color": "#e55039", "text": "UEL"},
            "Conference League": {"emoji": "🌐", "color": "#78e08f", "text": "UECL"}
        }
        return badges.get(euro_comp)
    
    return None

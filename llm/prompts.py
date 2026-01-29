"""
Prompt templates for LLM-based football analysis.
"""

SYSTEM_PROMPT = """You are FOOTBALL ANALYTICS GPT — an expert in football match prediction and betting analysis.
You combine statistical reasoning with deep football knowledge to provide insightful predictions.
Your analysis is based on hybrid machine learning models (XGBoost + LSTM) that process historical data, 
expected goals (xG), team form, and bookmaker odds."""

CONTEXT_TEMPLATE = """## Current Match Context

**Match:** {home_team} vs {away_team}
**League:** {league}
**Date:** {match_date}

### Team Form (Last 5 Matches)
- **{home_team}:** {home_form} 
- **{away_team}:** {away_form}

### Expected Goals (xG) Average
- **{home_team}:** {home_xg:.2f}
- **{away_team}:** {away_xg:.2f}

### Bookmaker Odds
- Home Win: {home_odds}
- Draw: {draw_odds}
- Away Win: {away_odds}
"""

PREDICTION_TEMPLATE = """## ML Model Predictions

### Ensemble Prediction (XGBoost 60% + LSTM 40%)
- **Home Win:** {home_prob:.1%}
- **Draw:** {draw_prob:.1%}
- **Away Win:** {away_prob:.1%}
- **Predicted Result:** {predicted_result}
- **Confidence:** {confidence:.1%}

### Individual Model Outputs
| Model | Home Win | Draw | Away Win |
|-------|----------|------|----------|
| XGBoost | {xgb_home:.1%} | {xgb_draw:.1%} | {xgb_away:.1%} |
| LSTM | {lstm_home:.1%} | {lstm_draw:.1%} | {lstm_away:.1%} |

### Value Bet Analysis
{value_bet_analysis}
"""

ANALYSIS_REQUEST = """Based on the above data and predictions, provide a comprehensive match analysis including:

1. **Match Preview** (2-3 sentences): Brief overview of the fixture and its context
2. **Key Factors** (bullet points): What's driving the prediction
3. **Risk Assessment**: Confidence level explanation
4. **Betting Recommendation**: Clear recommendation with reasoning

Keep your response professional, data-driven, and concise. Avoid excessive hedging language.
Focus on actionable insights."""

# Value bet formatting
VALUE_BET_TEMPLATE = """- **{bet_type}** @ {odds:.2f}
  - Implied Probability: {implied_prob:.1%}
  - Model Probability: {model_prob:.1%}
  - Edge: {edge:+.1%} {"✅ VALUE BET" if edge >= 0.05 else ""}
"""


def build_analysis_prompt(
    match_data: dict,
    prediction: dict,
    value_bets: list
) -> str:
    """
    Build complete prompt for LLM analysis.
    
    Args:
        match_data: Match information
        prediction: Model predictions
        value_bets: Value bet analysis
        
    Returns:
        Complete prompt string
    """
    # Format value bets
    value_bet_lines = []
    for vb in value_bets:
        line = f"- **{vb['bet_type']}** @ {vb['odds']:.2f}: "
        line += f"Implied {vb['implied_prob']:.1%} | Model {vb['model_prob']:.1%} | "
        edge_pct = vb['edge'] * 100
        line += f"Edge {edge_pct:+.1f}%"
        if vb['recommended']:
            line += " ✅"
        value_bet_lines.append(line)
    
    value_bet_text = "\n".join(value_bet_lines) if value_bet_lines else "No significant value bets identified."
    
    # Build context
    context = CONTEXT_TEMPLATE.format(
        home_team=match_data.get('home_team', 'Home Team'),
        away_team=match_data.get('away_team', 'Away Team'),
        league=match_data.get('league', 'Unknown League'),
        match_date=match_data.get('match_date', 'TBD'),
        home_form=match_data.get('home_form', 'N/A'),
        away_form=match_data.get('away_form', 'N/A'),
        home_xg=match_data.get('home_xg', 0),
        away_xg=match_data.get('away_xg', 0),
        home_odds=match_data.get('home_odds', 0),
        draw_odds=match_data.get('draw_odds', 0),
        away_odds=match_data.get('away_odds', 0)
    )
    
    probs = prediction.get('probabilities', {})
    xgb = prediction.get('xgb_probs', {})
    lstm = prediction.get('lstm_probs', {})
    
    result_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    
    # Build prediction section
    pred_section = PREDICTION_TEMPLATE.format(
        home_prob=probs.get('home_win', 0),
        draw_prob=probs.get('draw', 0),
        away_prob=probs.get('away_win', 0),
        predicted_result=result_map.get(prediction.get('predicted_result', 'H'), 'Home Win'),
        confidence=prediction.get('confidence', 0),
        xgb_home=xgb.get('home_win', 0),
        xgb_draw=xgb.get('draw', 0),
        xgb_away=xgb.get('away_win', 0),
        lstm_home=lstm.get('home_win', 0),
        lstm_draw=lstm.get('draw', 0),
        lstm_away=lstm.get('away_win', 0),
        value_bet_analysis=value_bet_text
    )
    
    # Combine all parts
    full_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n{pred_section}\n{ANALYSIS_REQUEST}"
    
    return full_prompt


def build_quick_summary_prompt(
    home_team: str,
    away_team: str,
    prediction: dict
) -> str:
    """Build a shorter prompt for quick summaries."""
    probs = prediction.get('probabilities', {})
    result_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    
    return f"""Provide a one-paragraph prediction summary for {home_team} vs {away_team}.

Probabilities: Home {probs.get('home_win', 0):.0%} | Draw {probs.get('draw', 0):.0%} | Away {probs.get('away_win', 0):.0%}
Predicted: {result_map.get(prediction.get('predicted_result', 'H'))} (Confidence: {prediction.get('confidence', 0):.0%})

Keep it to 3-4 sentences. Be confident and direct."""

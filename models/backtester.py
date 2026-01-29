"""
Backtesting Engine
Simulates predictions on historical data to evaluate model performance and detailed ROI.
"""
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from app.database import get_db
from models.ensemble import get_ensemble

logger = logging.getLogger(__name__)

class Backtester:
    """
    Simulates betting strategies on historical matches.
    """
    
    def __init__(self):
        self.ensemble = get_ensemble()
        self.initial_bankroll = 1000.0
        
    def _get_historical_matches(self, league: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Fetch finished matches from database."""
        query = """
            SELECT * FROM matches 
            WHERE result IS NOT NULL 
        """
        params = []
        
        if league:
            query += " AND league = ?"
            params.append(league)
            
        query += " ORDER BY match_date DESC LIMIT ?"
        params.append(limit)
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def _get_form_at_date(self, team: str, match_date: str, conn) -> str:
        """
        Reconstruct team form (last 5 games) BEFORE a specific date.
        CRITICAL for realistic backtesting (no look-ahead bias).
        """
        cursor = conn.cursor()
        cursor.execute("""
            SELECT result, home_team, away_team 
            FROM matches 
            WHERE (home_team = ? OR away_team = ?)
            AND match_date < ?
            AND result IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 5
        """, (team, team, match_date))
        
        rows = cursor.fetchall()
        if not rows:
            return "DDDDD"
            
        form = []
        for r in rows:
            # Determine if W/D/L for this team
            res = r['result']
            is_home = (r['home_team'] == team)
            
            if res == 'D':
                form.append('D')
            elif (is_home and res == 'H') or (not is_home and res == 'A'):
                form.append('W')
            else:
                form.append('L')
                
        # Pad with D if less than 5
        while len(form) < 5:
            form.append('D')
            
        return "".join(form)

    def run_backtest(self, league: Optional[str] = None, limit: int = 50) -> Dict:
        """
        Run backtest simulation.
        
        Algorithm:
        1. Fetch historical matches
        2. For each match:
           a. Reconstruct Home/Away form at that date
           b. Generate prediction using Ensemble
           c. Compare with actual result
           d. Calculate PnL (Profit and Loss)
        """
        matches = self._get_historical_matches(league, limit)
        results = []
        
        bankroll = self.initial_bankroll
        wins = 0
        total_bets = 0
        
        logger.info(f"Starting backtest on {len(matches)} matches...")
        
        with get_db() as conn:
            for match in matches:
                # 1. Reconstruct Context
                home_form = self._get_form_at_date(match['home_team'], match['match_date'], conn)
                away_form = self._get_form_at_date(match['away_team'], match['match_date'], conn)
                
                # 2. Build Match Data
                match_data = {
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'home_form': home_form,
                    'away_form': away_form,
                    'home_odds': match['home_odds'] or 2.0,
                    'draw_odds': match['draw_odds'] or 3.0,
                    'away_odds': match['away_odds'] or 3.0,
                    'league': match['league'],
                    'match_date': match['match_date']
                }
                
                # 3. Predict
                try:
                    pred = self.ensemble.predict(match_data)
                    
                    # 4. Simulate Stattegy: Bet on Highest Probability
                    # Simple Strategy: Bet 1 unit on the predicted outcome
                    predicted_res = pred['predicted_result'] # H, D, A
                    confidence = pred['confidence']
                    
                    # Only bet if confidence > 50% (More active strategy)
                    if confidence > 0.50:
                        # Safe Odds Retrieval
                        odds = None
                        if predicted_res == 'H': odds = match['home_odds']
                        elif predicted_res == 'D': odds = match['draw_odds']
                        elif predicted_res == 'A': odds = match['away_odds']
                        
                        # CRITICAL: Skip bet if odds are missing (prevents calculation errors)
                        if not odds:
                            continue
                            
                        is_win = (predicted_res == match['result'])
                        
                        stake = 100 # Fixed stake $100
                        pnl = 0
                        if is_win:
                            wins += 1
                            pnl = stake * (odds - 1)
                        else:
                            pnl = -stake
                            
                        bankroll += pnl
                        total_bets += 1
                        
                        results.append({
                            'match': f"{match['home_team']} vs {match['away_team']}",
                            'date': match['match_date'],
                            'prediction': predicted_res,
                            'actual': match['result'],
                            'confidence': round(confidence, 2),
                            'odds': odds,
                            'result': 'WIN' if is_win else 'LOSS',
                            'pnl': round(pnl, 2)
                        })
                        
                except Exception as e:
                    logger.error(f"Backtest error for match {match['id']}: {e}")
                    continue
                    
        # Metrics
        roi = ((bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        
        return {
            'total_matches_analyzed': len(matches),
            'total_bets_placed': total_bets,
            'wins': wins,
            'win_rate': round(win_rate, 2),
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': round(bankroll, 2),
            'roi_percent': round(roi, 2),
            'history': results[:20] # Return last 20 bets details
        }

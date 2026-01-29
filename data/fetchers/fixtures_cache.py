"""
Fixtures Cache Service - Manages cached fixtures in database.
Reduces API calls by storing fixtures and serving from cache.
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from app.database import get_db

logger = logging.getLogger(__name__)


class FixturesCache:
    """Service for caching and retrieving fixtures from database."""
    
    def save_fixtures(self, fixtures: List[Dict]) -> int:
        """
        Save fixtures to database cache.
        Updates existing fixtures or inserts new ones.
        Also records odds history for market analysis.
        
        Returns:
            Number of fixtures saved
        """
        if not fixtures:
            return 0
        
        saved = 0
        with get_db() as conn:
            cursor = conn.cursor()
            
            for fix in fixtures:
                try:
                    # Parse match date
                    match_date = fix.get('commence_time', '')
                    if isinstance(match_date, str) and match_date:
                        # Handle ISO format
                        match_date = match_date.replace('Z', '+00:00')
                        if 'T' in match_date:
                            match_date = match_date.split('T')[0] + ' ' + match_date.split('T')[1][:8]
                    
                    # 1. Update Fixtures Table
                    cursor.execute("""
                        INSERT OR REPLACE INTO fixtures 
                        (id, home_team, away_team, league, match_date, 
                         home_odds, draw_odds, away_odds, source, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fix.get('id'),
                        fix.get('home_team'),
                        fix.get('away_team'),
                        fix.get('league'),
                        match_date,
                        fix.get('home_odds'),
                        fix.get('draw_odds'),
                        fix.get('away_odds'),
                        fix.get('source', 'the-odds-api'),
                        datetime.now().isoformat()
                    ))
                    
                    # 2. Market Movement: Insert into odds_history
                    # We log every update to track movement
                    cursor.execute("""
                        INSERT INTO odds_history 
                        (fixture_id, home_odds, draw_odds, away_odds, source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        fix.get('id'),
                        fix.get('home_odds'),
                        fix.get('draw_odds'),
                        fix.get('away_odds'),
                        fix.get('source', 'the-odds-api')
                    ))
                    
                    saved += 1
                    
                except Exception as e:
                    logger.error(f"Failed to save fixture: {e}")
            
            conn.commit()
        
        logger.info(f"Saved {saved} fixtures to cache")
        return saved
    
    def get_fixtures_by_date(self, target_date: datetime) -> List[Dict]:
        """Get fixtures for a specific date."""
        date_str = target_date.strftime('%Y-%m-%d')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, home_team, away_team, league, match_date,
                       home_odds, draw_odds, away_odds, source, fetched_at
                FROM fixtures
                WHERE DATE(match_date) = DATE(?)
                ORDER BY match_date ASC, league ASC
            """, (date_str,))
            
            rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_today_fixtures(self) -> List[Dict]:
        """Get today's fixtures."""
        return self.get_fixtures_by_date(datetime.now())
    
    def get_tomorrow_fixtures(self) -> List[Dict]:
        """Get tomorrow's fixtures."""
        return self.get_fixtures_by_date(datetime.now() + timedelta(days=1))
    
    def get_week_fixtures(self) -> List[Dict]:
        """Get fixtures for the next 7 days."""
        today = datetime.now()
        week_later = today + timedelta(days=7)
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, home_team, away_team, league, match_date,
                       home_odds, draw_odds, away_odds, source, fetched_at
                FROM fixtures
                WHERE DATE(match_date) >= DATE(?) AND DATE(match_date) <= DATE(?)
                ORDER BY match_date ASC, league ASC
            """, (today.strftime('%Y-%m-%d'), week_later.strftime('%Y-%m-%d')))
            
            rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_fixtures_by_league(self, league: str) -> List[Dict]:
        """Get upcoming fixtures for a specific league."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, home_team, away_team, league, match_date,
                       home_odds, draw_odds, away_odds, source, fetched_at
                FROM fixtures
                WHERE league = ? AND DATE(match_date) >= DATE(?)
                ORDER BY match_date ASC
                LIMIT 20
            """, (league, today))
            
            rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_all_upcoming(self, limit: int = 50) -> List[Dict]:
        """Get all upcoming fixtures."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, home_team, away_team, league, match_date,
                       home_odds, draw_odds, away_odds, source, fetched_at
                FROM fixtures
                WHERE DATE(match_date) >= DATE(?)
                ORDER BY match_date ASC, league ASC
                LIMIT ?
            """, (today, limit))
            
            rows = cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    def get_last_fetch_time(self) -> Optional[datetime]:
        """Get when fixtures were last fetched."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(fetched_at) FROM fixtures
            """)
            row = cursor.fetchone()
            
            if row and row[0]:
                return datetime.fromisoformat(row[0])
        
        return None
    
    def clear_old_fixtures(self, days_old: int = 2) -> int:
        """Remove fixtures older than X days."""
        cutoff = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM fixtures WHERE DATE(match_date) < DATE(?)
            """, (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleared {deleted} old fixtures")
        return deleted
    
    def get_fixture_count(self) -> int:
        """Get total cached fixtures count."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fixtures")
            return cursor.fetchone()[0]
    
    def log_api_call(self, api_name: str, endpoint: str = None):
        """Log an API call for usage tracking."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_usage (api_name, endpoint) VALUES (?, ?)
            """, (api_name, endpoint))
            conn.commit()
    
    def get_api_usage_today(self) -> int:
        """Get API call count for today."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(calls_count) FROM api_usage 
                WHERE DATE(logged_at) = DATE(?)
            """, (today,))
            result = cursor.fetchone()[0]
        
        return result or 0
    
    def get_api_usage_month(self) -> int:
        """Get API call count for current month."""
        month_start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(calls_count) FROM api_usage 
                WHERE DATE(logged_at) >= DATE(?)
            """, (month_start,))
            result = cursor.fetchone()[0]
        
        return result or 0
    
    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        return {
            'id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'league': row[3],
            'commence_time': row[4],
            'home_odds': row[5] or 2.0,
            'draw_odds': row[6] or 3.5,
            'away_odds': row[7] or 3.0,
            'source': row[8] or 'cached'
        }


# Singleton
_cache = None

def get_fixtures_cache() -> FixturesCache:
    """Get fixtures cache singleton."""
    global _cache
    if _cache is None:
        _cache = FixturesCache()
    return _cache


def refresh_fixtures_from_api() -> Dict:
    """
    Refresh fixtures cache from The Odds API.
    Called by scheduler or manually.
    
    Returns:
        Status dict with counts
    """
    from data.fetchers.fixtures_api import get_fixtures_api
    
    cache = get_fixtures_cache()
    api = get_fixtures_api()
    
    # Track API usage
    cache.log_api_call('the-odds-api', '/sports/*/odds')
    
    # Fetch from all leagues
    all_fixtures = api.get_all_upcoming(limit_per_league=10)
    
    # Save to cache
    saved = cache.save_fixtures(all_fixtures)
    
    # Clear old fixtures
    cleared = cache.clear_old_fixtures(days_old=2)
    
    # Auto-predict all fixtures
    predicted = auto_predict_all_fixtures(all_fixtures)
    
    return {
        'fetched': len(all_fixtures),
        'saved': saved,
        'cleared': cleared,
        'predicted': predicted,
        'api_calls_today': cache.get_api_usage_today(),
        'api_calls_month': cache.get_api_usage_month()
    }


def auto_predict_all_fixtures(fixtures: List[Dict]) -> int:
    """
    Auto-predict all fixtures and save to cached_predictions table.
    
    Args:
        fixtures: List of fixture dicts with odds
        
    Returns:
        Number of predictions generated
    """
    from models.ensemble import get_ensemble
    from models.stake_calculator import calculate_confidence_rating
    from models.congestion import adjust_prediction_for_congestion
    from data.fetchers.understat import get_demo_xg
    
    ensemble = get_ensemble()
    predicted = 0
    
    # Try to get form fetcher for real form lookup
    try:
        from data.fetchers.form_fetcher import get_form_fetcher
        form_fetcher = get_form_fetcher()
    except:
        form_fetcher = None
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # NOTE: We no longer clear all predictions to prevent live flipping
        # cursor.execute("DELETE FROM cached_predictions")
        
        for fix in fixtures:
            try:
                fixture_id = fix.get('id')
                commence_time_str = fix.get('commence_time')
                
                # Check if match has started
                is_started = False
                if commence_time_str:
                    try:
                        # Handle ISO format (e.g. 2023-10-25T14:00:00Z)
                        dt_str = commence_time_str.replace('Z', '+00:00')
                        commence_dt = datetime.fromisoformat(dt_str)
                        # Ensure we compare timezone-aware or naive correctly
                        if commence_dt.tzinfo:
                            now = datetime.now(commence_dt.tzinfo)
                        else:
                            now = datetime.now()
                            
                        # Add a small buffer (e.g. 5 mins before kickoff we lock it)
                        if now > (commence_dt - timedelta(minutes=5)):
                            is_started = True
                    except Exception as e:
                        logger.warning(f"Error parsing date {commence_time_str}: {e}")
                
                # Check if prediction already exists
                cursor.execute("SELECT fixture_id FROM cached_predictions WHERE fixture_id = ?", (fixture_id,))
                exists = cursor.fetchone()
                
                # CRITICAL FIX: If match started and prediction exists, DO NOT UPDATE
                # This prevents "flipping" based on live score/odds
                if is_started and exists:
                    continue

                # Try to get real form from database
                home_form = None
                away_form = None
                
                if form_fetcher:
                    home_form = form_fetcher.get_team_form(fix.get('home_team', ''))
                    away_form = form_fetcher.get_team_form(fix.get('away_team', ''))
                
                # Smart form fallback based on odds (if form not found)
                if not home_form:
                    home_odds = fix.get('home_odds', 2.0)
                    if home_odds < 1.5:
                        home_form = 'WWWWW'  # Heavy favorite
                    elif home_odds < 2.0:
                        home_form = 'WWWDW'  # Favorite
                    elif home_odds < 3.0:
                        home_form = 'WDWDW'  # Even
                    else:
                        home_form = 'DWDLD'  # Underdog
                
                if not away_form:
                    away_odds = fix.get('away_odds', 3.0)
                    if away_odds < 1.5:
                        away_form = 'WWWWW'
                    elif away_odds < 2.0:
                        away_form = 'WWWDW'
                    elif away_odds < 3.0:
                        away_form = 'WDWDW'
                    else:
                        away_form = 'DWDLD'
                
                # Prepare match data with real or default form
                match_data = {
                    'home_team': fix.get('home_team'),
                    'away_team': fix.get('away_team'),
                    'home_form': home_form,
                    'away_form': away_form,
                    'home_odds': fix.get('home_odds', 2.0),
                    'draw_odds': fix.get('draw_odds', 3.5),
                    'away_odds': fix.get('away_odds', 3.0),
                    'home_xg': get_demo_xg(fix.get('home_team', '')).get('xg_for', 1.5),
                    'away_xg': get_demo_xg(fix.get('away_team', '')).get('xg_for', 1.2),
                    'league': fix.get('league')
                }
                
                # Get prediction
                pred = ensemble.predict(match_data)
                
                # Apply congestion adjustment for European teams
                league = fix.get('league', '')
                if league:
                    congestion_result = adjust_prediction_for_congestion(
                        home_team=fix.get('home_team', ''),
                        away_team=fix.get('away_team', ''),
                        league=league,
                        home_prob=pred['probabilities']['home_win'],
                        draw_prob=pred['probabilities']['draw'],
                        away_prob=pred['probabilities']['away_win']
                    )
                    # Update probabilities with congestion-adjusted values
                    if congestion_result.get('has_congestion_impact'):
                        pred['probabilities']['home_win'] = congestion_result['adjusted']['home']
                        pred['probabilities']['draw'] = congestion_result['adjusted']['draw']
                        pred['probabilities']['away_win'] = congestion_result['adjusted']['away']
                        # Recalculate predicted result based on adjusted probs
                        probs = pred['probabilities']
                        if probs['home_win'] >= probs['draw'] and probs['home_win'] >= probs['away_win']:
                            pred['predicted_result'] = 'H'
                        elif probs['away_win'] >= probs['home_win'] and probs['away_win'] >= probs['draw']:
                            pred['predicted_result'] = 'A'
                        else:
                            pred['predicted_result'] = 'D'
                        pred['confidence'] = max(probs['home_win'], probs['draw'], probs['away_win'])
                
                # Get value bets
                odds = {
                    'home': fix.get('home_odds', 2.0),
                    'draw': fix.get('draw_odds', 3.5),
                    'away': fix.get('away_odds', 3.0)
                }
                value_bets = ensemble.analyze_value_bets(
                    pred['probabilities'],
                    odds,
                    confidence=pred['confidence'],
                    models_agree=pred['models_agree']
                )
                
                # Find best bet
                best_bet = None
                best_edge = 0
                best_odds = 0
                recommended = 0
                
                for vb in value_bets:
                    if vb.get('recommended'):
                        if vb['edge'] > best_edge:
                            best_bet = vb['bet_type']
                            best_edge = vb['edge']
                            best_odds = vb['odds']
                            recommended = 1
                
                # Get rating
                rating = calculate_confidence_rating(
                    pred['confidence'],
                    best_edge if best_edge > 0 else 0,
                    pred['models_agree']
                )
                
                # Get stake from best value bet
                stake = 0
                ev = 0
                for vb in value_bets:
                    if vb.get('bet_type') == best_bet:
                        stake = vb.get('stake', 0)
                        ev = vb.get('expected_value', 0)
                        break
                
                # Save to cached_predictions
                # Save to cached_predictions
                # Use INSERT OR REPLACE because we checked "is_started" above.
                # If match NOT started, we want to update with latest odds/form.
                cursor.execute("""
                    INSERT OR REPLACE INTO cached_predictions 
                    (fixture_id, home_team, away_team, league, match_date,
                     home_odds, draw_odds, away_odds,
                     home_form, away_form,
                     home_prob, draw_prob, away_prob,
                     predicted_result, confidence, models_agree,
                     best_bet, best_bet_edge, best_bet_odds, recommended,
                     rating_stars, rating_label, stake_amount, expected_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fix.get('id'),
                    fix.get('home_team'),
                    fix.get('away_team'),
                    fix.get('league'),
                    fix.get('commence_time'),
                    fix.get('home_odds'),
                    fix.get('draw_odds'),
                    fix.get('away_odds'),
                    home_form,
                    away_form,
                    pred['probabilities']['home_win'],
                    pred['probabilities']['draw'],
                    pred['probabilities']['away_win'],
                    pred['predicted_result'],
                    pred['confidence'],
                    1 if pred['models_agree'] else 0,
                    best_bet,
                    best_edge,
                    best_odds,
                    recommended,
                    rating['stars'],
                    rating['label'],
                    stake,
                    ev
                ))
                
                # Check if prediction already exists for today to prevent duplicates
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE home_team = ? AND away_team = ? AND DATE(created_at) = DATE(?)
                """, (fix.get('home_team'), fix.get('away_team'), today))
                
                exists = cursor.fetchone()[0] > 0
                
                if not exists:
                    # Save to predictions table for history
                    cursor.execute("""
                        INSERT INTO predictions 
                        (home_team, away_team, league, home_odds, draw_odds, away_odds,
                         home_form, away_form, home_win_prob, draw_prob, away_win_prob,
                         xgb_home_prob, xgb_draw_prob, xgb_away_prob,
                         lstm_home_prob, lstm_draw_prob, lstm_away_prob,
                         predicted_result, confidence,
                         vb_home_implied_prob, vb_home_edge, vb_home_recommended,
                         vb_draw_implied_prob, vb_draw_edge, vb_draw_recommended,
                         vb_away_implied_prob, vb_away_edge, vb_away_recommended)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                    fix.get('home_team'),
                    fix.get('away_team'),
                    fix.get('league'),
                    fix.get('home_odds'),
                    fix.get('draw_odds'),
                    fix.get('away_odds'),
                    home_form,  # Real form or default
                    away_form,
                    pred['probabilities']['home_win'],
                    pred['probabilities']['draw'],
                    pred['probabilities']['away_win'],
                    pred.get('xgb_probs', {}).get('home_win', pred['probabilities']['home_win']),
                    pred.get('xgb_probs', {}).get('draw', pred['probabilities']['draw']),
                    pred.get('xgb_probs', {}).get('away_win', pred['probabilities']['away_win']),
                    pred.get('lstm_probs', {}).get('home_win', pred['probabilities']['home_win']),
                    pred.get('lstm_probs', {}).get('draw', pred['probabilities']['draw']),
                    pred.get('lstm_probs', {}).get('away_win', pred['probabilities']['away_win']),
                    pred['predicted_result'],
                    pred['confidence'],
                    1/fix.get('home_odds', 2.0) if fix.get('home_odds') else 0.5,
                    value_bets[0]['edge'] if value_bets else 0,
                    1 if value_bets and value_bets[0].get('recommended') else 0,
                    1/fix.get('draw_odds', 3.5) if fix.get('draw_odds') else 0.28,
                    value_bets[1]['edge'] if len(value_bets) > 1 else 0,
                    1 if len(value_bets) > 1 and value_bets[1].get('recommended') else 0,
                    1/fix.get('away_odds', 3.0) if fix.get('away_odds') else 0.33,
                    value_bets[2]['edge'] if len(value_bets) > 2 else 0,
                    1 if len(value_bets) > 2 and value_bets[2].get('recommended') else 0
                    ))
                
                predicted += 1
                
            except Exception as e:
                logger.error(f"Failed to predict fixture {fix.get('id')}: {e}")
        
        conn.commit()
    
    logger.info(f"Auto-predicted {predicted} fixtures")
    return predicted


def get_today_with_predictions() -> List[Dict]:
    """Get today's fixtures with pre-computed predictions."""
    today = datetime.now().strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fixture_id, home_team, away_team, league, match_date,
                   home_odds, draw_odds, away_odds,
                   home_prob, draw_prob, away_prob,
                   predicted_result, confidence, models_agree,
                   best_bet, best_bet_edge, best_bet_odds, recommended,
                   rating_stars, rating_label, stake_amount, expected_value
            FROM cached_predictions
            WHERE DATE(match_date) = DATE(?)
            ORDER BY match_date ASC, league ASC
        """, (today,))
        
        rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'league': row[3],
            'commence_time': row[4],
            'home_odds': row[5],
            'draw_odds': row[6],
            'away_odds': row[7],
            'home_prob': row[8],
            'draw_prob': row[9],
            'away_prob': row[10],
            'predicted_result': row[11],
            'confidence': row[12],
            'models_agree': bool(row[13]),
            'best_bet': row[14],
            'best_bet_edge': row[15],
            'best_bet_odds': row[16],
            'recommended': bool(row[17]),
            'rating_stars': row[18],
            'rating_label': row[19],
            'stake': row[20],
            'expected_value': row[21]
        })
    
    return results


def get_tomorrow_with_predictions() -> List[Dict]:
    """Get tomorrow's fixtures with pre-computed predictions."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fixture_id, home_team, away_team, league, match_date,
                   home_odds, draw_odds, away_odds,
                   home_prob, draw_prob, away_prob,
                   predicted_result, confidence, models_agree,
                   best_bet, best_bet_edge, best_bet_odds, recommended,
                   rating_stars, rating_label, stake_amount, expected_value
            FROM cached_predictions
            WHERE DATE(match_date) = DATE(?)
            ORDER BY match_date ASC, league ASC
        """, (tomorrow,))
        
        rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'league': row[3],
            'commence_time': row[4],
            'home_odds': row[5],
            'draw_odds': row[6],
            'away_odds': row[7],
            'home_prob': row[8],
            'draw_prob': row[9],
            'away_prob': row[10],
            'predicted_result': row[11],
            'confidence': row[12],
            'models_agree': bool(row[13]),
            'best_bet': row[14],
            'best_bet_edge': row[15],
            'best_bet_odds': row[16],
            'recommended': bool(row[17]),
            'rating_stars': row[18],
            'rating_label': row[19],
            'stake': row[20],
            'expected_value': row[21]
        })
    
    return results


def get_week_with_predictions() -> List[Dict]:
    """Get fixtures for the next 7 days with pre-computed predictions."""
    today = datetime.now().strftime('%Y-%m-%d')
    week_later = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fixture_id, home_team, away_team, league, match_date,
                   home_odds, draw_odds, away_odds,
                   home_prob, draw_prob, away_prob,
                   predicted_result, confidence, models_agree,
                   best_bet, best_bet_edge, best_bet_odds, recommended,
                   rating_stars, rating_label, stake_amount, expected_value
            FROM cached_predictions
            WHERE DATE(match_date) >= DATE(?) AND DATE(match_date) <= DATE(?)
            ORDER BY match_date ASC, league ASC
        """, (today, week_later))
        
        rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'league': row[3],
            'commence_time': row[4],
            'home_odds': row[5],
            'draw_odds': row[6],
            'away_odds': row[7],
            'home_prob': row[8],
            'draw_prob': row[9],
            'away_prob': row[10],
            'predicted_result': row[11],
            'confidence': row[12],
            'models_agree': bool(row[13]),
            'best_bet': row[14],
            'best_bet_edge': row[15],
            'best_bet_odds': row[16],
            'recommended': bool(row[17]),
            'rating_stars': row[18],
            'rating_label': row[19],
            'stake': row[20],
            'expected_value': row[21]
        })
    
    return results


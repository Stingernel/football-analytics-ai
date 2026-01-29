"""
Prediction API endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging

from llm.analyzer import get_analyzer
from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class PredictRequest(BaseModel):
    """Request model for match prediction."""
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    home_form: Optional[str] = Field(None, description="Home team form (e.g., WWDLW)")
    away_form: Optional[str] = Field(None, description="Away team form")
    home_odds: float = Field(..., ge=1.0, description="Home win odds")
    draw_odds: float = Field(..., ge=1.0, description="Draw odds")
    away_odds: float = Field(..., ge=1.0, description="Away win odds")
    home_xg: Optional[float] = Field(1.5, ge=0, description="Home xG average")
    away_xg: Optional[float] = Field(1.2, ge=0, description="Away xG average")
    league: Optional[str] = Field(None, description="League name")
    match_date: Optional[str] = Field(None, description="Match date (YYYY-MM-DD)")
    include_llm_analysis: bool = Field(True, description="Include LLM analysis")


class RatingResponse(BaseModel):
    """Rating info for value bet."""
    stars: int
    label: str
    color: str
    stars_display: str
    should_bet: bool


class ValueBetResponse(BaseModel):
    """Value bet in response with professional metrics."""
    bet_type: str
    odds: float
    implied_prob: float
    model_prob: float
    edge: float
    edge_pct: Optional[float] = None
    recommended: bool
    stake: Optional[float] = None
    stake_pct: Optional[float] = None
    expected_value: Optional[float] = None
    rating: Optional[RatingResponse] = None
    reason: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    id: Optional[int] = None
    home_team: str
    away_team: str
    league: Optional[str]
    
    # Probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    # Prediction
    predicted_result: str
    predicted_result_full: str
    confidence: float
    models_agree: bool
    
    # Model details
    xgb_probs: dict
    lstm_probs: dict
    
    # Value bets
    value_bets: List[ValueBetResponse]
    recommended_bet: Optional[ValueBetResponse]
    
    # Analysis
    analysis: str
    
    created_at: str


@router.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictRequest):
    """
    Generate prediction for a match.
    
    Takes match data and returns:
    - Win probabilities from ensemble model
    - Predicted outcome with confidence
    - Value bet analysis
    - LLM-generated insights
    """
    try:
        analyzer = get_analyzer()
        
        match_data = {
            'home_team': request.home_team,
            'away_team': request.away_team,
            'home_form': request.home_form or 'DDD',
            'away_form': request.away_form or 'DDD',
            'home_odds': request.home_odds,
            'draw_odds': request.draw_odds,
            'away_odds': request.away_odds,
            'home_xg': request.home_xg or 1.5,
            'away_xg': request.away_xg or 1.2,
            'league': request.league or 'Unknown',
            'match_date': request.match_date or datetime.now().strftime('%Y-%m-%d')
        }
        
        result = await analyzer.analyze_match(
            match_data, 
            include_llm=request.include_llm_analysis
        )
        
        # Save to database
        prediction_id = _save_prediction(result)
        
        # Build response
        probs = result['prediction']['probabilities']
        
        return PredictionResponse(
            id=prediction_id,
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            home_win_prob=probs['home_win'],
            draw_prob=probs['draw'],
            away_win_prob=probs['away_win'],
            predicted_result=result['prediction']['predicted_result'],
            predicted_result_full=result['prediction']['predicted_result_full'],
            confidence=result['prediction']['confidence'],
            models_agree=result['prediction']['models_agree'],
            xgb_probs=result['model_details']['xgb'],
            lstm_probs=result['model_details']['lstm'],
            value_bets=[ValueBetResponse(**vb) for vb in result['value_bets']],
            recommended_bet=ValueBetResponse(**result['recommended_bet']) if result['recommended_bet'] else None,
            analysis=result['analysis'],
            created_at=result['generated_at']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick")
async def quick_prediction(request: PredictRequest):
    """
    Get a quick one-paragraph prediction without full analysis.
    """
    try:
        analyzer = get_analyzer()
        
        match_data = {
            'home_team': request.home_team,
            'away_team': request.away_team,
            'home_form': request.home_form or 'DDD',
            'away_form': request.away_form or 'DDD',
            'home_odds': request.home_odds,
            'draw_odds': request.draw_odds,
            'away_odds': request.away_odds,
            'home_xg': request.home_xg or 1.5,
            'away_xg': request.away_xg or 1.2,
        }
        
        summary = await analyzer.quick_prediction(match_data)
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Quick prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _save_prediction(result: dict) -> int:
    """Save prediction to database and return ID."""
    try:
        match = result.get('match', {})
        recommended = result.get('recommended_bet')
        value_bets = result.get('value_bets', [])
        
        # Extract value bets by type
        vb_home = next((vb for vb in value_bets if vb['bet_type'] == 'HOME'), None)
        vb_draw = next((vb for vb in value_bets if vb['bet_type'] == 'DRAW'), None)
        vb_away = next((vb for vb in value_bets if vb['bet_type'] == 'AWAY'), None)
        
        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO predictions (
                    home_team, away_team, league,
                    home_odds, draw_odds, away_odds,
                    home_form, away_form,
                    created_at,
                    home_win_prob, draw_prob, away_win_prob,
                    xgb_home_prob, xgb_draw_prob, xgb_away_prob,
                    lstm_home_prob, lstm_draw_prob, lstm_away_prob,
                    predicted_result, confidence,
                    vb_home_implied_prob, vb_home_edge, vb_home_recommended,
                    vb_draw_implied_prob, vb_draw_edge, vb_draw_recommended,
                    vb_away_implied_prob, vb_away_edge, vb_away_recommended,
                    value_bet, value_bet_odds, value_bet_edge,
                    llm_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match.get('home_team'),
                match.get('away_team'),
                match.get('league'),
                match.get('home_odds'),
                match.get('draw_odds'),
                match.get('away_odds'),
                match.get('home_form'),
                match.get('away_form'),
                datetime.now().isoformat(),
                result['prediction']['probabilities']['home_win'],
                result['prediction']['probabilities']['draw'],
                result['prediction']['probabilities']['away_win'],
                result['model_details']['xgb']['home_win'],
                result['model_details']['xgb']['draw'],
                result['model_details']['xgb']['away_win'],
                result['model_details']['lstm']['home_win'],
                result['model_details']['lstm']['draw'],
                result['model_details']['lstm']['away_win'],
                result['prediction']['predicted_result'],
                result['prediction']['confidence'],
                # Value bet HOME
                vb_home['implied_prob'] if vb_home else None,
                vb_home['edge'] if vb_home else None,
                1 if vb_home and vb_home.get('recommended') else 0,
                # Value bet DRAW
                vb_draw['implied_prob'] if vb_draw else None,
                vb_draw['edge'] if vb_draw else None,
                1 if vb_draw and vb_draw.get('recommended') else 0,
                # Value bet AWAY
                vb_away['implied_prob'] if vb_away else None,
                vb_away['edge'] if vb_away else None,
                1 if vb_away and vb_away.get('recommended') else 0,
                # Legacy fields
                recommended['bet_type'] if recommended else None,
                recommended['odds'] if recommended else None,
                recommended['edge'] if recommended else None,
                result['analysis']
            ))
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        return None


# ============================================================================
# AUTO-PREDICTION ENDPOINTS
# ============================================================================

class UpcomingMatch(BaseModel):
    """Upcoming match from fixtures API."""
    id: str
    home_team: str
    away_team: str
    league: str
    commence_time: str
    home_odds: float
    draw_odds: float
    away_odds: float
    source: str


class AutoPredictRequest(BaseModel):
    """Request to auto-predict specific matches."""
    match_ids: List[str] = Field(..., description="List of match IDs to predict")
    include_llm: bool = Field(True, description="Include LLM analysis")


class AutoPredictAllRequest(BaseModel):
    """Request to auto-predict all upcoming matches."""
    leagues: Optional[List[str]] = Field(None, description="Leagues to include")
    limit_per_league: int = Field(5, ge=1, le=20, description="Max matches per league")
    include_llm: bool = Field(True, description="Include LLM analysis")


@router.get("/upcoming", response_model=List[UpcomingMatch])
async def get_upcoming_matches(
    league: str = "Premier League",
    limit: int = 10
):
    """
    Get upcoming matches with odds from fixtures API.
    
    Uses The Odds API if ODDS_API_KEY is configured, otherwise demo mode.
    """
    try:
        from data.fetchers.fixtures_api import get_fixtures_api
        
        api = get_fixtures_api()
        matches = api.get_upcoming_matches(league, limit)
        
        return [UpcomingMatch(**m) for m in matches]
        
    except Exception as e:
        logger.error(f"Error fetching upcoming matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming/all", response_model=List[UpcomingMatch])
async def get_all_upcoming_matches(
    limit_per_league: int = 5
):
    """Get upcoming matches from all major leagues."""
    try:
        from data.fetchers.fixtures_api import get_fixtures_api
        
        api = get_fixtures_api()
        matches = api.get_all_upcoming(limit_per_league=limit_per_league)
        
        return [UpcomingMatch(**m) for m in matches]
        
    except Exception as e:
        logger.error(f"Error fetching all upcoming matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto")
async def auto_predict_matches(request: AutoPredictRequest):
    """
    Auto-predict selected upcoming matches.
    
    Takes match IDs and generates predictions for each.
    """
    try:
        from data.fetchers.fixtures_api import get_fixtures_api
        from data.fetchers.football_data import get_fetcher
        
        fixtures_api = get_fixtures_api()
        data_fetcher = get_fetcher()
        analyzer = get_analyzer()
        
        # Get all upcoming matches
        all_matches = fixtures_api.get_all_upcoming(limit_per_league=20)
        match_map = {m['id']: m for m in all_matches}
        
        results = []
        errors = []
        
        for match_id in request.match_ids:
            match = match_map.get(match_id)
            if not match:
                errors.append({"match_id": match_id, "error": "Match not found"})
                continue
            
            try:
                # Try to get form from historical data
                home_form = "DDD"
                away_form = "DDD"
                
                try:
                    df = data_fetcher.fetch_season(match['league'])
                    if df is not None and not df.empty:
                        home_form = data_fetcher.get_team_form(df, match['home_team']) or "DDD"
                        away_form = data_fetcher.get_team_form(df, match['away_team']) or "DDD"
                except:
                    pass
                
                # Build match data
                match_data = {
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'home_form': home_form,
                    'away_form': away_form,
                    'home_odds': match['home_odds'],
                    'draw_odds': match['draw_odds'],
                    'away_odds': match['away_odds'],
                    'home_xg': 1.5,
                    'away_xg': 1.2,
                    'league': match['league'],
                    'match_date': match['commence_time'][:10] if match.get('commence_time') else datetime.now().strftime('%Y-%m-%d')
                }
                
                # Generate prediction
                result = await analyzer.analyze_match(match_data, include_llm=request.include_llm)
                
                # Save to database
                prediction_id = _save_prediction(result)
                
                results.append({
                    "match_id": match_id,
                    "prediction_id": prediction_id,
                    "home_team": match['home_team'],
                    "away_team": match['away_team'],
                    "predicted_result": result['prediction']['predicted_result'],
                    "confidence": result['prediction']['confidence'],
                    "recommended_bet": result['recommended_bet']['bet_type'] if result.get('recommended_bet') else None
                })
                
            except Exception as e:
                errors.append({"match_id": match_id, "error": str(e)})
        
        return {
            "success": len(results),
            "failed": len(errors),
            "predictions": results,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Auto-predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-all")
async def auto_predict_all_matches(request: AutoPredictAllRequest):
    """
    Auto-predict all upcoming matches from specified leagues.
    
    Convenience endpoint that fetches all matches and predicts them.
    """
    try:
        from data.fetchers.fixtures_api import get_fixtures_api
        
        api = get_fixtures_api()
        matches = api.get_all_upcoming(
            leagues=request.leagues,
            limit_per_league=request.limit_per_league
        )
        
        if not matches:
            return {"success": 0, "failed": 0, "predictions": [], "errors": []}
        
        # Call auto_predict with all match IDs
        match_ids = [m['id'] for m in matches]
        
        auto_request = AutoPredictRequest(
            match_ids=match_ids,
            include_llm=request.include_llm
        )
        
        return await auto_predict_matches(auto_request)
        
    except Exception as e:
        logger.error(f"Auto-predict all error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CACHED FIXTURES ENDPOINTS ====================

@router.get("/fixtures/today")
async def get_today_fixtures():
    """Get today's cached fixtures (no new API call)."""
    try:
        from data.fetchers.fixtures_cache import get_fixtures_cache
        
        cache = get_fixtures_cache()
        fixtures = cache.get_today_fixtures()
        
        return fixtures
        
    except Exception as e:
        logger.error(f"Get today fixtures error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fixtures/tomorrow")
async def get_tomorrow_fixtures():
    """Get tomorrow's cached fixtures (no new API call)."""
    try:
        from data.fetchers.fixtures_cache import get_fixtures_cache
        
        cache = get_fixtures_cache()
        fixtures = cache.get_tomorrow_fixtures()
        
        return fixtures
        
    except Exception as e:
        logger.error(f"Get tomorrow fixtures error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fixtures/week")
async def get_week_fixtures():
    """Get this week's cached fixtures (no new API call)."""
    try:
        from data.fetchers.fixtures_cache import get_fixtures_cache
        
        cache = get_fixtures_cache()
        fixtures = cache.get_week_fixtures()
        
        return fixtures
        
    except Exception as e:
        logger.error(f"Get week fixtures error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fixtures/status")
async def get_fixtures_status():
    """Get cache status info."""
    try:
        from data.fetchers.fixtures_cache import get_fixtures_cache
        
        cache = get_fixtures_cache()
        last_update = cache.get_last_fetch_time()
        
        return {
            'last_update': last_update.strftime('%Y-%m-%d %H:%M') if last_update else 'Never',
            'total_fixtures': cache.get_fixture_count(),
            'api_calls_today': cache.get_api_usage_today(),
            'api_calls_month': cache.get_api_usage_month()
        }
        
    except Exception as e:
        logger.error(f"Get fixtures status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fixtures/refresh")
async def refresh_fixtures():
    """
    Manually refresh fixtures from API.
    Admin endpoint - use sparingly to conserve API calls.
    """
    try:
        from data.fetchers.fixtures_cache import refresh_fixtures_from_api
        
        result = refresh_fixtures_from_api()
        
        return {
            'success': True,
            'message': f"Refreshed {result['saved']} fixtures, predicted {result.get('predicted', 0)}",
            **result
        }
        
    except Exception as e:
        logger.error(f"Refresh fixtures error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fixtures/predictions/today")
async def get_today_predictions():
    """Get today's fixtures with pre-computed predictions."""
    try:
        from data.fetchers.fixtures_cache import get_today_with_predictions
        
        return get_today_with_predictions()
        
    except Exception as e:
        logger.error(f"Get today predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fixtures/predictions/tomorrow")
async def get_tomorrow_predictions():
    """Get tomorrow's fixtures with pre-computed predictions."""
    try:
        from data.fetchers.fixtures_cache import get_tomorrow_with_predictions
        
        return get_tomorrow_with_predictions()
        
    except Exception as e:
        logger.error(f"Get tomorrow predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fixtures/predictions/week")
async def get_week_predictions():
    """Get this week's fixtures with pre-computed predictions."""
    try:
        from data.fetchers.fixtures_cache import get_week_with_predictions
        
        return get_week_with_predictions()
        
    except Exception as e:
        logger.error(f"Get week predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

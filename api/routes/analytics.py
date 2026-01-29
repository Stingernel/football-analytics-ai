"""
Analytics Routes
Endpoints for backtesting and system performance analysis.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from models.backtester import Backtester
from models.analytics import MarketAnalyzer, get_strength_calculator

router = APIRouter()

class BacktestRequest(BaseModel):
    league: Optional[str] = None
    limit: int = 50

@router.get("/ratings")
async def get_team_ratings():
    """
    Get current Elo ratings for all teams.
    """
    try:
        calc = get_strength_calculator()
        ratings = calc.calculate_current_ratings()
        # Sort by rating descending
        sorted_ratings = [{"team": k, "rating": int(v)} for k, v in sorted(ratings.items(), key=lambda item: item[1], reverse=True)]
        return sorted_ratings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-movement")
async def get_market_movement(threshold: float = 0.05):
    """
    Get matches with significant odds drops (> 5%).
    """
    try:
        analyzer = MarketAnalyzer()
        results = analyzer.detect_dropping_odds(threshold_pct=threshold)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run a simulation on historical data.
    """
    try:
        tester = Backtester()
        result = tester.run_backtest(
            league=request.league,
            limit=request.limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

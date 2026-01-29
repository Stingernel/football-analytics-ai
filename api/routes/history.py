"""
Prediction history API endpoints - Enhanced with filtering and team data.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, date
import logging

from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class HistoryItem(BaseModel):
    """Prediction history item with team details."""
    id: int
    home_team: Optional[str]
    away_team: Optional[str]
    league: Optional[str]
    home_odds: Optional[float]
    draw_odds: Optional[float]
    away_odds: Optional[float]
    created_at: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_result: str
    confidence: float
    value_bet: Optional[str]
    value_bet_odds: Optional[float]
    value_bet_edge: Optional[float]
    actual_result: Optional[str]
    correct: Optional[bool]


class HistoryDetail(HistoryItem):
    """Detailed history item with full model breakdown and value bet analysis."""
    home_form: Optional[str]
    away_form: Optional[str]
    xgb_home_prob: Optional[float]
    xgb_draw_prob: Optional[float]
    xgb_away_prob: Optional[float]
    lstm_home_prob: Optional[float]
    lstm_draw_prob: Optional[float]
    lstm_away_prob: Optional[float]
    # Value bet details for all outcomes
    vb_home_implied_prob: Optional[float]
    vb_home_edge: Optional[float]
    vb_home_recommended: Optional[bool]
    vb_draw_implied_prob: Optional[float]
    vb_draw_edge: Optional[float]
    vb_draw_recommended: Optional[bool]
    vb_away_implied_prob: Optional[float]
    vb_away_edge: Optional[float]
    vb_away_recommended: Optional[bool]
    llm_analysis: Optional[str]


class PerformanceStats(BaseModel):
    """Model performance statistics."""
    total_predictions: int
    predictions_with_result: int
    correct_predictions: int
    accuracy: float
    home_accuracy: float
    draw_accuracy: float
    away_accuracy: float
    profit_if_bet_predicted: float
    roi: float
    avg_confidence: float
    value_bets_count: int
    value_bets_correct: int


class BreakdownStats(BaseModel):
    """Accuracy breakdown by predicted outcome."""
    predicted_result: str
    total: int
    verified: int
    correct: int
    accuracy: float


@router.get("/", response_model=List[HistoryItem])
async def get_prediction_history(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    only_verified: bool = Query(False, description="Only show predictions with actual results"),
    team_name: Optional[str] = Query(None, description="Filter by team name (home or away)"),
    predicted_result: Optional[str] = Query(None, description="Filter by predicted result (H/D/A)"),
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """Get prediction history with filtering options."""
    try:
        with get_db() as conn:
            query = "SELECT * FROM predictions WHERE 1=1"
            params = []
            
            if only_verified:
                query += " AND actual_result IS NOT NULL"
            
            if team_name:
                query += " AND (home_team LIKE ? OR away_team LIKE ?)"
                search_term = f"%{team_name}%"
                params.extend([search_term, search_term])
            
            if predicted_result and predicted_result in ['H', 'D', 'A']:
                query += " AND predicted_result = ?"
                params.append(predicted_result)
            
            if from_date:
                query += " AND date(created_at) >= ?"
                params.append(from_date)
            
            if to_date:
                query += " AND date(created_at) <= ?"
                params.append(to_date)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            predictions = cursor.fetchall()
            
            result = []
            for p in predictions:
                correct = None
                if p['actual_result']:
                    correct = p['predicted_result'] == p['actual_result']
                
                result.append(HistoryItem(
                    id=p['id'],
                    home_team=p['home_team'],
                    away_team=p['away_team'],
                    league=p['league'],
                    home_odds=p['home_odds'],
                    draw_odds=p['draw_odds'],
                    away_odds=p['away_odds'],
                    created_at=p['created_at'],
                    home_win_prob=p['home_win_prob'],
                    draw_prob=p['draw_prob'],
                    away_win_prob=p['away_win_prob'],
                    predicted_result=p['predicted_result'],
                    confidence=p['confidence'],
                    value_bet=p['value_bet'],
                    value_bet_odds=p['value_bet_odds'],
                    value_bet_edge=p['value_bet_edge'],
                    actual_result=p['actual_result'],
                    correct=correct
                ))
            
            return result
            
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/performance", response_model=PerformanceStats)
async def get_performance_stats():
    """Get comprehensive model performance statistics."""
    try:
        with get_db() as conn:
            # Total predictions
            total = conn.execute(
                "SELECT COUNT(*) as count FROM predictions"
            ).fetchone()['count']
            
            # Predictions with results
            verified = conn.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE actual_result IS NOT NULL"
            ).fetchone()['count']
            
            # Correct predictions
            correct = conn.execute("""
                SELECT COUNT(*) as count FROM predictions 
                WHERE actual_result IS NOT NULL 
                AND predicted_result = actual_result
            """).fetchone()['count']
            
            # Accuracy by result type
            home_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_result = actual_result THEN 1 ELSE 0 END) as correct
                FROM predictions 
                WHERE actual_result IS NOT NULL AND predicted_result = 'H'
            """).fetchone()
            
            draw_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_result = actual_result THEN 1 ELSE 0 END) as correct
                FROM predictions 
                WHERE actual_result IS NOT NULL AND predicted_result = 'D'
            """).fetchone()
            
            away_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_result = actual_result THEN 1 ELSE 0 END) as correct
                FROM predictions 
                WHERE actual_result IS NOT NULL AND predicted_result = 'A'
            """).fetchone()
            
            # Average confidence
            avg_conf = conn.execute(
                "SELECT AVG(confidence) as avg FROM predictions WHERE confidence IS NOT NULL"
            ).fetchone()['avg'] or 0
            
            # Value bets stats
            value_bets = conn.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE value_bet IS NOT NULL"
            ).fetchone()['count']
            
            value_bets_correct = conn.execute("""
                SELECT COUNT(*) as count FROM predictions 
                WHERE value_bet IS NOT NULL 
                AND actual_result IS NOT NULL
                AND value_bet = actual_result
            """).fetchone()['count']
            
            accuracy = correct / verified if verified > 0 else 0
            home_acc = (home_stats['correct'] or 0) / home_stats['total'] if home_stats['total'] > 0 else 0
            draw_acc = (draw_stats['correct'] or 0) / draw_stats['total'] if draw_stats['total'] > 0 else 0
            away_acc = (away_stats['correct'] or 0) / away_stats['total'] if away_stats['total'] > 0 else 0
            
            # Calculate profit (simplified: assuming flat 2.0 odds, 1 unit stake)
            profit = correct * 1.0 - verified * 1.0
            roi = profit / verified if verified > 0 else 0
            
            return PerformanceStats(
                total_predictions=total,
                predictions_with_result=verified,
                correct_predictions=correct,
                accuracy=round(accuracy, 4),
                home_accuracy=round(home_acc, 4),
                draw_accuracy=round(draw_acc, 4),
                away_accuracy=round(away_acc, 4),
                profit_if_bet_predicted=round(profit, 2),
                roi=round(roi, 4),
                avg_confidence=round(avg_conf, 4),
                value_bets_count=value_bets,
                value_bets_correct=value_bets_correct
            )
            
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/breakdown", response_model=List[BreakdownStats])
async def get_accuracy_breakdown():
    """Get accuracy breakdown by predicted outcome."""
    try:
        with get_db() as conn:
            results = []
            
            for result_type in ['H', 'D', 'A']:
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN actual_result IS NOT NULL THEN 1 ELSE 0 END) as verified,
                        SUM(CASE WHEN predicted_result = actual_result THEN 1 ELSE 0 END) as correct
                    FROM predictions 
                    WHERE predicted_result = ?
                """, (result_type,)).fetchone()
                
                accuracy = (stats['correct'] or 0) / stats['verified'] if stats['verified'] > 0 else 0
                
                results.append(BreakdownStats(
                    predicted_result=result_type,
                    total=stats['total'],
                    verified=stats['verified'],
                    correct=stats['correct'] or 0,
                    accuracy=round(accuracy, 4)
                ))
            
            return results
            
    except Exception as e:
        logger.error(f"Error calculating breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{prediction_id}", response_model=HistoryDetail)
async def get_prediction_detail(prediction_id: int):
    """Get detailed prediction with full model breakdown."""
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM predictions WHERE id = ?",
                (prediction_id,)
            )
            p = cursor.fetchone()
            
            if not p:
                raise HTTPException(status_code=404, detail="Prediction not found")
            
            correct = None
            if p['actual_result']:
                correct = p['predicted_result'] == p['actual_result']
            
            return HistoryDetail(
                id=p['id'],
                home_team=p['home_team'],
                away_team=p['away_team'],
                league=p['league'],
                home_odds=p['home_odds'],
                draw_odds=p['draw_odds'],
                away_odds=p['away_odds'],
                home_form=p['home_form'],
                away_form=p['away_form'],
                created_at=p['created_at'],
                home_win_prob=p['home_win_prob'],
                draw_prob=p['draw_prob'],
                away_win_prob=p['away_win_prob'],
                predicted_result=p['predicted_result'],
                confidence=p['confidence'],
                value_bet=p['value_bet'],
                value_bet_odds=p['value_bet_odds'],
                value_bet_edge=p['value_bet_edge'],
                actual_result=p['actual_result'],
                correct=correct,
                xgb_home_prob=p['xgb_home_prob'],
                xgb_draw_prob=p['xgb_draw_prob'],
                xgb_away_prob=p['xgb_away_prob'],
                lstm_home_prob=p['lstm_home_prob'],
                lstm_draw_prob=p['lstm_draw_prob'],
                lstm_away_prob=p['lstm_away_prob'],
                # Value bet details for all outcomes
                vb_home_implied_prob=p['vb_home_implied_prob'],
                vb_home_edge=p['vb_home_edge'],
                vb_home_recommended=bool(p['vb_home_recommended']) if p['vb_home_recommended'] is not None else None,
                vb_draw_implied_prob=p['vb_draw_implied_prob'],
                vb_draw_edge=p['vb_draw_edge'],
                vb_draw_recommended=bool(p['vb_draw_recommended']) if p['vb_draw_recommended'] is not None else None,
                vb_away_implied_prob=p['vb_away_implied_prob'],
                vb_away_edge=p['vb_away_edge'],
                vb_away_recommended=bool(p['vb_away_recommended']) if p['vb_away_recommended'] is not None else None,
                llm_analysis=p['llm_analysis']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{prediction_id}/result")
async def update_prediction_result(prediction_id: int, actual_result: str):
    """Update a prediction with the actual result."""
    if actual_result not in ['H', 'D', 'A']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid result. Must be 'H', 'D', or 'A'"
        )
    
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "UPDATE predictions SET actual_result = ? WHERE id = ?",
                (actual_result, prediction_id)
            )
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Prediction not found")
            
            return {"message": "Result updated successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """Delete a prediction."""
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM predictions WHERE id = ?",
                (prediction_id,)
            )
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Prediction not found")
            
            return {"message": "Prediction deleted successfully"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

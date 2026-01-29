"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class MatchResult(str, Enum):
    HOME = "H"
    DRAW = "D"
    AWAY = "A"


class TeamBase(BaseModel):
    """Base team model."""
    name: str
    short_name: Optional[str] = None
    league: Optional[str] = None
    country: Optional[str] = None


class Team(TeamBase):
    """Team with ID."""
    id: int
    
    class Config:
        from_attributes = True


class MatchInput(BaseModel):
    """Input for match prediction."""
    home_team: str
    away_team: str
    home_form: Optional[str] = None  # e.g., "WWDLW"
    away_form: Optional[str] = None
    home_odds: float = Field(ge=1.0, description="Bookmaker odds for home win")
    draw_odds: float = Field(ge=1.0, description="Bookmaker odds for draw")
    away_odds: float = Field(ge=1.0, description="Bookmaker odds for away win")
    home_xg: Optional[float] = Field(None, ge=0, description="Expected goals home (last 5 avg)")
    away_xg: Optional[float] = Field(None, ge=0, description="Expected goals away (last 5 avg)")
    league: Optional[str] = None


class ProbabilityOutput(BaseModel):
    """Probability distribution for match outcomes."""
    home_win: float = Field(ge=0, le=1)
    draw: float = Field(ge=0, le=1)
    away_win: float = Field(ge=0, le=1)


class ModelPrediction(BaseModel):
    """Individual model prediction."""
    model_name: str
    probabilities: ProbabilityOutput
    confidence: float


class ValueBet(BaseModel):
    """Value bet recommendation."""
    bet_type: str  # "HOME", "DRAW", "AWAY", "NONE"
    odds: float
    implied_prob: float
    model_prob: float
    edge: float  # model_prob - implied_prob
    recommended: bool


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    id: Optional[int] = None
    match: MatchInput
    
    # Ensemble probabilities
    probabilities: ProbabilityOutput
    predicted_result: MatchResult
    confidence: float
    
    # Individual model outputs
    xgb_prediction: ModelPrediction
    lstm_prediction: ModelPrediction
    
    # Value analysis
    value_bets: List[ValueBet]
    
    # LLM analysis
    analysis: str
    reasoning: str
    
    created_at: datetime = Field(default_factory=datetime.now)


class PredictionHistory(BaseModel):
    """Prediction history item."""
    id: int
    home_team: str
    away_team: str
    predicted_result: MatchResult
    confidence: float
    actual_result: Optional[MatchResult] = None
    correct: Optional[bool] = None
    created_at: datetime


class ModelPerformance(BaseModel):
    """Model performance metrics."""
    model_type: str
    accuracy: float
    log_loss: float
    total_predictions: int
    correct_predictions: int
    precision: dict  # {"H": 0.65, "D": 0.45, "A": 0.60}
    recall: dict


class MatchData(BaseModel):
    """Historical match data."""
    id: int
    home_team: str
    away_team: str
    match_date: datetime
    season: str
    league: str
    home_goals: int
    away_goals: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    result: MatchResult
    
    class Config:
        from_attributes = True

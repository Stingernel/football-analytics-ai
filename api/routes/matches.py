"""
Match data API endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging

from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class TeamResponse(BaseModel):
    """Team response model."""
    id: int
    name: str
    short_name: Optional[str]
    league: Optional[str]
    country: Optional[str]


class MatchResponse(BaseModel):
    """Match response model."""
    id: int
    home_team: str
    away_team: str
    match_date: str
    season: Optional[str]
    league: Optional[str]
    home_goals: Optional[int]
    away_goals: Optional[int]
    home_xg: Optional[float]
    away_xg: Optional[float]
    home_odds: Optional[float]
    draw_odds: Optional[float]
    away_odds: Optional[float]
    result: Optional[str]


class AddMatchRequest(BaseModel):
    """Request to add a match."""
    home_team: str
    away_team: str
    match_date: str
    season: Optional[str] = None
    league: Optional[str] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None


@router.get("/teams", response_model=List[TeamResponse])
async def get_teams(
    league: Optional[str] = None,
    limit: int = Query(100, le=500)
):
    """Get list of teams."""
    try:
        with get_db() as conn:
            if league:
                cursor = conn.execute(
                    "SELECT * FROM teams WHERE league = ? LIMIT ?",
                    (league, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM teams LIMIT ?",
                    (limit,)
                )
            
            teams = cursor.fetchall()
            return [TeamResponse(
                id=t['id'],
                name=t['name'],
                short_name=t['short_name'],
                league=t['league'],
                country=t['country']
            ) for t in teams]
            
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[MatchResponse])
async def get_matches(
    league: Optional[str] = None,
    season: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0)
):
    """Get list of matches with optional filters."""
    try:
        with get_db() as conn:
            query = """
                SELECT m.*, 
                       ht.name as home_team_name,
                       at.name as away_team_name
                FROM matches m
                LEFT JOIN teams ht ON m.home_team_id = ht.id
                LEFT JOIN teams at ON m.away_team_id = at.id
                WHERE 1=1
            """
            params = []
            
            if league:
                query += " AND m.league = ?"
                params.append(league)
            
            if season:
                query += " AND m.season = ?"
                params.append(season)
            
            if team:
                query += " AND (ht.name LIKE ? OR at.name LIKE ?)"
                params.extend([f"%{team}%", f"%{team}%"])
            
            query += " ORDER BY m.match_date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            matches = cursor.fetchall()
            
            return [MatchResponse(
                id=m['id'],
                home_team=m['home_team_name'] or 'Unknown',
                away_team=m['away_team_name'] or 'Unknown',
                match_date=m['match_date'],
                season=m['season'],
                league=m['league'],
                home_goals=m['home_goals'],
                away_goals=m['away_goals'],
                home_xg=m['home_xg'],
                away_xg=m['away_xg'],
                home_odds=m['home_odds'],
                draw_odds=m['draw_odds'],
                away_odds=m['away_odds'],
                result=m['result']
            ) for m in matches]
            
    except Exception as e:
        logger.error(f"Error fetching matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(match_id: int):
    """Get a specific match by ID."""
    try:
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT m.*, 
                       ht.name as home_team_name,
                       at.name as away_team_name
                FROM matches m
                LEFT JOIN teams ht ON m.home_team_id = ht.id
                LEFT JOIN teams at ON m.away_team_id = at.id
                WHERE m.id = ?
            """, (match_id,))
            
            m = cursor.fetchone()
            
            if not m:
                raise HTTPException(status_code=404, detail="Match not found")
            
            return MatchResponse(
                id=m['id'],
                home_team=m['home_team_name'] or 'Unknown',
                away_team=m['away_team_name'] or 'Unknown',
                match_date=m['match_date'],
                season=m['season'],
                league=m['league'],
                home_goals=m['home_goals'],
                away_goals=m['away_goals'],
                home_xg=m['home_xg'],
                away_xg=m['away_xg'],
                home_odds=m['home_odds'],
                draw_odds=m['draw_odds'],
                away_odds=m['away_odds'],
                result=m['result']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching match: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=MatchResponse)
async def add_match(request: AddMatchRequest):
    """Add a new match to the database."""
    try:
        with get_db() as conn:
            # Get or create teams
            def get_or_create_team(name: str) -> int:
                cursor = conn.execute(
                    "SELECT id FROM teams WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()
                if row:
                    return row['id']
                
                cursor = conn.execute(
                    "INSERT INTO teams (name, league) VALUES (?, ?)",
                    (name, request.league)
                )
                return cursor.lastrowid
            
            home_team_id = get_or_create_team(request.home_team)
            away_team_id = get_or_create_team(request.away_team)
            
            # Determine result
            result = None
            if request.home_goals is not None and request.away_goals is not None:
                if request.home_goals > request.away_goals:
                    result = 'H'
                elif request.home_goals < request.away_goals:
                    result = 'A'
                else:
                    result = 'D'
            
            cursor = conn.execute("""
                INSERT INTO matches (
                    home_team_id, away_team_id, match_date, season, league,
                    home_goals, away_goals, home_xg, away_xg,
                    home_odds, draw_odds, away_odds, result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                home_team_id, away_team_id, request.match_date,
                request.season, request.league,
                request.home_goals, request.away_goals,
                request.home_xg, request.away_xg,
                request.home_odds, request.draw_odds, request.away_odds,
                result
            ))
            
            match_id = cursor.lastrowid
            
            return MatchResponse(
                id=match_id,
                home_team=request.home_team,
                away_team=request.away_team,
                match_date=request.match_date,
                season=request.season,
                league=request.league,
                home_goals=request.home_goals,
                away_goals=request.away_goals,
                home_xg=request.home_xg,
                away_xg=request.away_xg,
                home_odds=request.home_odds,
                draw_odds=request.draw_odds,
                away_odds=request.away_odds,
                result=result
            )
            
    except Exception as e:
        logger.error(f"Error adding match: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leagues/list")
async def get_leagues():
    """Get list of available leagues."""
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT league FROM matches WHERE league IS NOT NULL ORDER BY league"
            )
            leagues = [row['league'] for row in cursor.fetchall()]
            return {"leagues": leagues}
            
    except Exception as e:
        logger.error(f"Error fetching leagues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

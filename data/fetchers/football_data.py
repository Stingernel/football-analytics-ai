"""
Data fetcher for Football-data.co.uk
Provides historical match data with odds.
"""
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Football-data.co.uk URLs
BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUE_CODES = {
    "Premier League": "E0",
    "Championship": "E1",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
    "Eredivisie": "N1",
    "Primeira Liga": "P1"
}


class FootballDataFetcher:
    """Fetches historical match data from football-data.co.uk"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_season(
        self, 
        league: str = "Premier League",
        season: str = "2324"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a season's data.
        
        Args:
            league: League name
            season: Season code (e.g., "2324" for 2023-24)
            
        Returns:
            DataFrame with match data
        """
        league_code = LEAGUE_CODES.get(league)
        if not league_code:
            logger.error(f"Unknown league: {league}")
            return None
        
        # Check cache first
        cache_file = self.cache_dir / f"{league_code}_{season}.csv"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Fetch from source
        url = f"{BASE_URL}/{season}/{league_code}.csv"
        logger.info(f"Fetching: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            cache_file.write_bytes(response.content)
            
            # Parse CSV
            df = pd.read_csv(cache_file)
            return self._process_dataframe(df, league)
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return None
    
    def _process_dataframe(self, df: pd.DataFrame, league: str) -> pd.DataFrame:
        """Process and clean the dataframe."""
        # Standard column mapping
        column_map = {
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',
            'B365H': 'home_odds',
            'B365D': 'draw_odds',
            'B365A': 'away_odds',
            'Date': 'date'
        }
        
        # Select relevant columns
        available_cols = [c for c in column_map.keys() if c in df.columns]
        df = df[available_cols].rename(columns=column_map)
        
        # Add league
        df['league'] = league
        
        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        
        # Clean missing odds
        for col in ['home_odds', 'draw_odds', 'away_odds']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['home_team', 'away_team'])
    
    def fetch_multiple_seasons(
        self,
        league: str = "Premier League",
        seasons: List[str] = None
    ) -> pd.DataFrame:
        """Fetch multiple seasons and combine."""
        if seasons is None:
            seasons = ["2122", "2223", "2324"]
        
        dfs = []
        for season in seasons:
            df = self.fetch_season(league, season)
            if df is not None:
                df['season'] = season
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_team_form(
        self, 
        df: pd.DataFrame, 
        team: str, 
        before_date: datetime = None,
        n_matches: int = 5
    ) -> str:
        """
        Calculate team form from historical data.
        
        Returns form string like "WWDLW"
        """
        if before_date is None:
            before_date = datetime.now()
        
        # Filter matches involving team
        mask = (
            ((df['home_team'] == team) | (df['away_team'] == team)) &
            (df['date'] < before_date)
        )
        
        matches = df[mask].sort_values('date', ascending=False).head(n_matches)
        
        form = []
        for _, match in matches.iterrows():
            if match['home_team'] == team:
                if match['result'] == 'H':
                    form.append('W')
                elif match['result'] == 'A':
                    form.append('L')
                else:
                    form.append('D')
            else:  # Away team
                if match['result'] == 'A':
                    form.append('W')
                elif match['result'] == 'H':
                    form.append('L')
                else:
                    form.append('D')
        
        return ''.join(form)


# Singleton
_fetcher = None

def get_fetcher() -> FootballDataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = FootballDataFetcher()
    return _fetcher

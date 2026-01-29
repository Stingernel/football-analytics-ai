"""
Sample Premier League data for testing and demos.
"""

SAMPLE_TEAMS = [
    {"name": "Manchester City", "short_name": "MCI", "league": "Premier League", "country": "England"},
    {"name": "Arsenal", "short_name": "ARS", "league": "Premier League", "country": "England"},
    {"name": "Liverpool", "short_name": "LIV", "league": "Premier League", "country": "England"},
    {"name": "Chelsea", "short_name": "CHE", "league": "Premier League", "country": "England"},
    {"name": "Manchester United", "short_name": "MUN", "league": "Premier League", "country": "England"},
    {"name": "Tottenham", "short_name": "TOT", "league": "Premier League", "country": "England"},
    {"name": "Newcastle", "short_name": "NEW", "league": "Premier League", "country": "England"},
    {"name": "Brighton", "short_name": "BHA", "league": "Premier League", "country": "England"},
    {"name": "Aston Villa", "short_name": "AVL", "league": "Premier League", "country": "England"},
    {"name": "West Ham", "short_name": "WHU", "league": "Premier League", "country": "England"},
]

SAMPLE_MATCHES = [
    {
        "home_team": "Manchester City",
        "away_team": "Arsenal",
        "match_date": "2024-09-22",
        "season": "2425",
        "league": "Premier League",
        "home_goals": 2,
        "away_goals": 2,
        "home_xg": 1.8,
        "away_xg": 1.9,
        "home_odds": 1.65,
        "draw_odds": 4.00,
        "away_odds": 4.50,
        "result": "D"
    },
    {
        "home_team": "Liverpool",
        "away_team": "Chelsea",
        "match_date": "2024-10-20",
        "season": "2425",
        "league": "Premier League",
        "home_goals": 2,
        "away_goals": 1,
        "home_xg": 2.3,
        "away_xg": 0.9,
        "home_odds": 1.55,
        "draw_odds": 4.33,
        "away_odds": 5.25,
        "result": "H"
    },
    {
        "home_team": "Arsenal",
        "away_team": "Manchester United",
        "match_date": "2024-12-04",
        "season": "2425",
        "league": "Premier League",
        "home_goals": 3,
        "away_goals": 1,
        "home_xg": 2.5,
        "away_xg": 1.1,
        "home_odds": 1.70,
        "draw_odds": 3.80,
        "away_odds": 4.50,
        "result": "H"
    },
    {
        "home_team": "Tottenham",
        "away_team": "Liverpool",
        "match_date": "2024-12-22",
        "season": "2425",
        "league": "Premier League",
        "home_goals": 3,
        "away_goals": 6,
        "home_xg": 2.1,
        "away_xg": 4.2,
        "home_odds": 3.10,
        "draw_odds": 3.60,
        "away_odds": 2.25,
        "result": "A"
    },
    {
        "home_team": "Newcastle",
        "away_team": "Manchester City",
        "match_date": "2024-12-28",
        "season": "2425",
        "league": "Premier League",
        "home_goals": 1,
        "away_goals": 1,
        "home_xg": 1.2,
        "away_xg": 1.5,
        "home_odds": 3.40,
        "draw_odds": 3.50,
        "away_odds": 2.10,
        "result": "D"
    }
]

TEAM_FORM = {
    "Manchester City": "DWWWW",
    "Arsenal": "WWWDW",
    "Liverpool": "WWWWW",
    "Chelsea": "WDWLW",
    "Manchester United": "LDWDL",
    "Tottenham": "WLWWL",
    "Newcastle": "DWDWW",
    "Brighton": "WDDWL",
    "Aston Villa": "WDWWD",
    "West Ham": "LLDWL"
}

TEAM_XG = {
    "Manchester City": {"xg_for": 2.4, "xg_against": 0.9},
    "Arsenal": {"xg_for": 2.1, "xg_against": 1.0},
    "Liverpool": {"xg_for": 2.8, "xg_against": 1.1},
    "Chelsea": {"xg_for": 1.8, "xg_against": 1.3},
    "Manchester United": {"xg_for": 1.4, "xg_against": 1.5},
    "Tottenham": {"xg_for": 2.0, "xg_against": 1.8},
    "Newcastle": {"xg_for": 1.6, "xg_against": 1.0},
    "Brighton": {"xg_for": 1.5, "xg_against": 1.2},
    "Aston Villa": {"xg_for": 1.7, "xg_against": 1.1},
    "West Ham": {"xg_for": 1.2, "xg_against": 1.6}
}


def get_team_form(team: str) -> str:
    """Get form string for a team."""
    return TEAM_FORM.get(team, "DDDDD")


def get_team_xg(team: str) -> dict:
    """Get xG data for a team."""
    return TEAM_XG.get(team, {"xg_for": 1.5, "xg_against": 1.2})


def populate_database():
    """Populate database with sample data."""
    import sys
    sys.path.insert(0, str(__file__).rsplit('\\data\\', 1)[0])
    
    from app.database import get_db
    
    with get_db() as conn:
        # Insert teams
        for team in SAMPLE_TEAMS:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO teams (name, short_name, league, country)
                    VALUES (?, ?, ?, ?)
                """, (team['name'], team['short_name'], team['league'], team['country']))
            except Exception as e:
                print(f"Error inserting team: {e}")
        
        # Insert matches
        for match in SAMPLE_MATCHES:
            try:
                # Get team IDs
                home_id = conn.execute(
                    "SELECT id FROM teams WHERE name = ?",
                    (match['home_team'],)
                ).fetchone()
                
                away_id = conn.execute(
                    "SELECT id FROM teams WHERE name = ?",
                    (match['away_team'],)
                ).fetchone()
                
                if home_id and away_id:
                    conn.execute("""
                        INSERT INTO matches (
                            home_team_id, away_team_id, match_date, season, league,
                            home_goals, away_goals, home_xg, away_xg,
                            home_odds, draw_odds, away_odds, result
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        home_id[0], away_id[0], match['match_date'],
                        match['season'], match['league'],
                        match['home_goals'], match['away_goals'],
                        match['home_xg'], match['away_xg'],
                        match['home_odds'], match['draw_odds'], match['away_odds'],
                        match['result']
                    ))
            except Exception as e:
                print(f"Error inserting match: {e}")
        
        print("Sample data populated successfully!")


if __name__ == "__main__":
    populate_database()

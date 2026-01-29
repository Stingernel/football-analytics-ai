"""Check data quality in matches table."""
from app.database import get_db

with get_db() as conn:
    cursor = conn.cursor()
    
    # Total matches
    cursor.execute("SELECT COUNT(*) FROM matches")
    total = cursor.fetchone()[0]
    print(f"Total matches: {total}")
    
    # Matches with odds
    cursor.execute("SELECT COUNT(*) FROM matches WHERE home_odds IS NOT NULL")
    with_odds = cursor.fetchone()[0]
    print(f"Matches with odds: {with_odds}")
    
    # Matches with result
    cursor.execute("SELECT COUNT(*) FROM matches WHERE result IS NOT NULL")
    with_result = cursor.fetchone()[0]
    print(f"Matches with result: {with_result}")
    
    # Sample data
    print("\nSample matches:")
    cursor.execute("SELECT home_team, away_team, home_odds, draw_odds, away_odds, result FROM matches LIMIT 5")
    for row in cursor.fetchall():
        print(f"  {row['home_team']} vs {row['away_team']}: odds={row['home_odds']}/{row['draw_odds']}/{row['away_odds']}, result={row['result']}")

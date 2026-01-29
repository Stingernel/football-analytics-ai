"""Check odds data in database."""
from app.database import get_db

with get_db() as conn:
    cursor = conn.cursor()
    
    # Check odds_history
    cursor.execute("SELECT COUNT(*) FROM odds_history")
    print(f"Odds history records: {cursor.fetchone()[0]}")
    
    # Check fixtures with odds
    cursor.execute("SELECT COUNT(*) FROM fixtures WHERE home_odds IS NOT NULL")
    print(f"Fixtures with odds: {cursor.fetchone()[0]}")
    
    # Check cached_predictions with odds
    cursor.execute("SELECT COUNT(*) FROM cached_predictions WHERE home_odds IS NOT NULL")
    print(f"Predictions with odds: {cursor.fetchone()[0]}")
    
    # Sample odds data
    cursor.execute("SELECT home_team, away_team, home_odds, draw_odds, away_odds FROM fixtures WHERE home_odds IS NOT NULL LIMIT 5")
    rows = cursor.fetchall()
    if rows:
        print("\nSample fixtures with odds:")
        for r in rows:
            print(f"  {r['home_team']} vs {r['away_team']}: {r['home_odds']}/{r['draw_odds']}/{r['away_odds']}")
    
    # Check matches table for odds
    cursor.execute("SELECT COUNT(*) FROM matches WHERE home_odds IS NOT NULL")
    print(f"\nMatches with odds: {cursor.fetchone()[0]}")

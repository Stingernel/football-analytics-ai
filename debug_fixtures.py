"""
Debug Fixtures Cache
Check what get_week_fixtures returns.
"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

from data.fetchers.fixtures_cache import get_fixtures_cache

def debug_fixtures():
    cache = get_fixtures_cache()
    
    print(f"Current Time: {datetime.now()}")
    
    print("\n--- Getting Today ---")
    today = cache.get_today_fixtures()
    print(f"Count: {len(today)}")
    
    print("\n--- Getting Tomorrow ---")
    tomorrow = cache.get_tomorrow_fixtures()
    print(f"Count: {len(tomorrow)}")
    
    print("\n--- Getting This Week ---")
    week = cache.get_week_fixtures()
    print(f"Count: {len(week)}")
    
    if len(week) > 0:
        print("\nSample Week Fixtures:")
        for f in week[:5]:
            print(f"  {f['match_date']} | {f['home_team']} vs {f['away_team']}")
            
    # Check specifically for the Oviedo game we saw
    from app.database import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT match_date FROM fixtures WHERE home_team LIKE '%Oviedo%'")
        rows = cursor.fetchall()
        print(f"\nOviedo Matches in DB: {[r[0] for r in rows]}")

if __name__ == "__main__":
    debug_fixtures()

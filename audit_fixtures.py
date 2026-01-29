"""
Audit Fixtures Data - Final
Check DB Schema and Fixtures data.
"""
from app.database import get_db
import datetime

def audit_fixtures():
    print("="*60)
    print("AUDIT FIXTURES DATA (FINAL)")
    print("="*60)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Check total count
        cursor.execute("SELECT COUNT(*) FROM fixtures")
        total = cursor.fetchone()[0]
        print(f"\nTotal Fixtures in DB: {total}")
        
        if total == 0:
            print("❌ Database Fixtures KOSONG.")
        
        # 2. Check Predictions count (Cached Predictions)
        cursor.execute("SELECT COUNT(*) FROM cached_predictions")
        total_pred = cursor.fetchone()[0]
        print(f"Total Cached Predictions: {total_pred}")

        if total > 0:
            # 3. Check date range
            cursor.execute(f"SELECT MIN(match_date), MAX(match_date) FROM fixtures")
            min_date, max_date = cursor.fetchone()
            print(f"Fixtures Date Range: {min_date} to {max_date}")
            
            # 4. Check sample fixtures (Top 5 upcoming)
            cursor.execute(f"""
                SELECT id, home_team, away_team, match_date, league 
                FROM fixtures 
                ORDER BY match_date DESC
                LIMIT 5
            """)
            recent = cursor.fetchall()
            
            print("\nLast 5 Fixtures Added:")
            for f in recent:
                 print(f"  [{f['match_date']}] {f['home_team']} vs {f['away_team']} ({f['league']})")
                 
            # 5. Check UPCOMING fixtures (Real-time check)
            cursor.execute(f"SELECT COUNT(*) FROM fixtures WHERE date(match_date) >= date('now')")
            upcoming_count = cursor.fetchone()[0]
            print(f"\nUpcoming fixtures count (Today/Future): {upcoming_count}")
            
            if upcoming_count == 0:
                print("⚠️ WARNING: No upcoming fixtures found. Data might be stale!")
            else:
                cursor.execute(f"""
                    SELECT id, home_team, away_team, match_date, league 
                    FROM fixtures 
                    WHERE date(match_date) >= date('now')
                    ORDER BY match_date ASC
                    LIMIT 5
                """)
                upcoming = cursor.fetchall()
                print("Next 5 Upcoming Fixtures:")
                for f in upcoming:
                    print(f"  [{f['match_date']}] {f['home_team']} vs {f['away_team']} ({f['league']})")

        # 6. Check for dummy patterns
        cursor.execute("SELECT COUNT(*) FROM fixtures WHERE home_team LIKE 'Team %' OR home_team = 'Home Team'")
        dummy_count = cursor.fetchone()[0]
        if dummy_count > 0:
             print(f"\n⚠️ WARNING: Found {dummy_count} potential DUMMY fixtures (Team X vs Team Y)")

if __name__ == "__main__":
    audit_fixtures()

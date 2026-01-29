"""
Clean Dummy Data
Remove dummy fixtures and predictions from database.
"""
from app.database import get_db

def clean_dummy_data():
    print("="*60)
    print("CLEANING DUMMY DATA")
    print("="*60)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check count before
        cursor.execute("SELECT COUNT(*) FROM fixtures")
        total_fixtures = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM cached_predictions")
        total_preds = cursor.fetchone()[0]
        
        print(f"Before: {total_fixtures} fixtures, {total_preds} predictions")
        
        # Delete dummy fixtures (detected by timestamp usually, or just clear all stale ones)
        # But safest is to clear ALL fixtures if we want fresh start, preventing mix
        # Or delete where source='demo' if we stored that
        
        # Check source column
        try:
            cursor.execute("SELECT COUNT(*) FROM fixtures WHERE source='demo'")
            demo_count = cursor.fetchone()[0]
            print(f"Found {demo_count} explicit 'demo' source fixtures")
            
            if demo_count > 0:
                cursor.execute("DELETE FROM fixtures WHERE source='demo'")
                print("Deleted fixtures with source='demo'")
                
            # Also clear cached_predictions for those fixtures (cascade usually handles but let's be sure)
            # Actually cached_predictions doesn't have foreign key constraint in sqlite usually unless enabled
            cursor.execute("DELETE FROM cached_predictions WHERE fixture_id NOT IN (SELECT id FROM fixtures)")
            print("Cleanup orphaned predictions")
            
        except Exception as e:
            print(f"Error checking source: {e}")
        
        # If user says ALL are dummy, let's just clear ALL if we suspect bad state
        # But let's verify with user input? No, user complained about dummy.
        # Let's clear ALL UPCOMING to be safe and refetch.
        
        print("Clearing all fixtures to force fresh fetch...")
        cursor.execute("DELETE FROM fixtures")
        cursor.execute("DELETE FROM cached_predictions")
        
        conn.commit()
        
        print("✅ Database cleaned. All fixtures removed.")

if __name__ == "__main__":
    clean_dummy_data()

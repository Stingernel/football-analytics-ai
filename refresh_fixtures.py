"""
Refresh Fixtures Only
Directly calls fixtures cache refresh logic.
"""
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)

from data.fetchers.fixtures_cache import refresh_fixtures_from_api

def main():
    print("="*60)
    print("REFRESHING FIXTURES FROM THE ODDS API")
    print("="*60)
    
    try:
        stats = refresh_fixtures_from_api()
        print("\nRefresh Result:")
        print(f"  Processed: {stats.get('processed', 0)}")
        print(f"  Added/Updated: {stats.get('added', 0)}")
        print(f"  Skipped: {stats.get('skipped', 0)}")
        print(f"  Errors: {stats.get('errors', 0)}")
        
        # Check if actual fixtures added
        if stats.get('added', 0) == 0 and stats.get('processed', 0) == 0:
            print("⚠️ No fixtures added. Check API quota or if season ended.")
            
    except Exception as e:
        print(f"❌ Error refreshing fixtures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

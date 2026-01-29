"""
System Strengthening Script
Fetches real historical data, trains model, and runs backtest.
Uses Football-Data.org API for historical match data.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def step1_fetch_historical_data():
    """Fetch historical match data from Football-Data.org."""
    print("\n" + "=" * 60)
    print("[STEP 1] Fetching Historical Data from Football-Data.org")
    print("=" * 60)
    
    from data.fetchers.history_fetcher import HistoryFetcher
    
    fetcher = HistoryFetcher()
    if not fetcher.api_key:
        print("ERROR: FOOTBALL_DATA_API_KEY not set in .env!")
        return 0
    
    leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
    total_matches = 0
    
    for league in leagues:
        print(f"\n  Fetching {league}...")
        try:
            count = fetcher.fetch_season_matches(league)
            total_matches += count
            print(f"    -> {count} matches saved")
        except Exception as e:
            print(f"    -> Error: {e}")
    
    print(f"\n  Total matches fetched: {total_matches}")
    return total_matches


def step2_train_with_real_data():
    """Train model using fetched historical data."""
    print("\n" + "=" * 60)
    print("[STEP 2] Training Model with Real Data")
    print("=" * 60)
    
    from app.database import get_db
    
    # Check how much data we have
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches WHERE result IS NOT NULL")
        count = cursor.fetchone()[0]
    
    print(f"\n  Matches available for training: {count}")
    
    if count < 50:
        print("  WARNING: Not enough data for good training (need 50+ matches)")
        print("  Training with available data anyway...")
    
    # Run training
    from train_simple import train_and_save
    accuracy = train_and_save()
    
    return accuracy


def step3_run_backtest():
    """Run backtest to validate model performance."""
    print("\n" + "=" * 60)
    print("[STEP 3] Running Backtest Validation")
    print("=" * 60)
    
    from models.backtester import Backtester
    
    tester = Backtester()
    
    print("\n  Running backtest on historical data...")
    result = tester.run_backtest(limit=100)
    
    print(f"\n  Backtest Results:")
    print(f"  ------------------")
    print(f"  Total matches analyzed: {result['total_matches_analyzed']}")
    print(f"  Total bets placed: {result['total_bets_placed']}")
    print(f"  Wins: {result['wins']}")
    print(f"  Win rate: {result['win_rate']:.1f}%")
    print(f"  Initial bankroll: ${result['initial_bankroll']:.2f}")
    print(f"  Final bankroll: ${result['final_bankroll']:.2f}")
    print(f"  ROI: {result['roi_percent']:.1f}%")
    
    return result


def step4_update_predictions():
    """Generate fresh predictions for upcoming fixtures."""
    print("\n" + "=" * 60)
    print("[STEP 4] Updating Predictions with Trained Model")
    print("=" * 60)
    
    from data.fetchers.fixtures_cache import refresh_fixtures_from_api
    
    print("\n  Refreshing fixtures and generating predictions...")
    result = refresh_fixtures_from_api()
    
    print(f"  Result: {result}")
    return result


def main():
    print("=" * 60)
    print("[*] FOOTBALL ANALYTICS - SYSTEM STRENGTHENING")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Fetch historical data from Football-Data.org")
    print("  2. Train ML model with real data")
    print("  3. Run backtest to validate performance")
    print("  4. Update predictions with trained model")
    
    # Step 1: Fetch data
    matches = step1_fetch_historical_data()
    
    if matches > 0:
        # Step 2: Train
        accuracy = step2_train_with_real_data()
        
        # Step 3: Backtest
        backtest = step3_run_backtest()
        
        # Step 4: Update predictions
        step4_update_predictions()
    else:
        print("\nNo data fetched. Please check:")
        print("  1. FOOTBALL_DATA_API_KEY is set in .env")
        print("  2. Internet connection is working")
        print("  3. API rate limits (10 calls/min for free tier)")
    
    print("\n" + "=" * 60)
    print("[OK] SYSTEM STRENGTHENING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

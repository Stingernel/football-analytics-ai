"""Quick script to check model accuracy."""
import pickle

# Load model
with open('trained_models/simple_model.pkl', 'rb') as f:
    data = pickle.load(f)

print("=" * 50)
print("MODEL ACCURACY REPORT")
print("=" * 50)
print(f"Test Accuracy: {data['accuracy']*100:.1f}%")
print(f"Trained at: {data['trained_at']}")
print(f"Features: {len(data['feature_names'])}")
print()

# Run quick backtest
print("Running backtest...")
from models.backtester import Backtester
t = Backtester()
r = t.run_backtest(limit=100)
print()
print("=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"Matches analyzed: {r['total_matches_analyzed']}")
print(f"Bets placed: {r['total_bets_placed']}")
print(f"Wins: {r['wins']}")
print(f"Win Rate: {r['win_rate']:.1f}%")
print(f"ROI: {r['roi_percent']:.1f}%")

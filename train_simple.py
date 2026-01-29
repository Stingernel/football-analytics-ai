"""
Simple ML Model Training - Fixed version with proper features.
Uses team strength calculated from historical performance.
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from app.database import get_db
from models.simple_classifier import SimpleLogisticClassifier

MODELS_DIR = Path(__file__).parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True)


def calculate_team_stats(matches, up_to_index: int) -> Dict[str, Dict]:
    """Calculate team strength from previous matches."""
    team_stats = defaultdict(lambda: {'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0})
    
    for i, m in enumerate(matches):
        if i >= up_to_index:
            break
        
        home = m['home_team']
        away = m['away_team']
        result = m['result']
        home_g = int(m['home_goals']) if m['home_goals'] else 0
        away_g = int(m['away_goals']) if m['away_goals'] else 0
        
        team_stats[home]['matches'] += 1
        team_stats[home]['goals_for'] += home_g
        team_stats[home]['goals_against'] += away_g
        
        team_stats[away]['matches'] += 1
        team_stats[away]['goals_for'] += away_g
        team_stats[away]['goals_against'] += home_g
        
        if result == 'H':
            team_stats[home]['wins'] += 1
            team_stats[away]['losses'] += 1
        elif result == 'A':
            team_stats[home]['losses'] += 1
            team_stats[away]['wins'] += 1
        else:
            team_stats[home]['draws'] += 1
            team_stats[away]['draws'] += 1
    
    return team_stats


def get_team_strength(stats: Dict) -> float:
    """Calculate team strength from stats."""
    if stats['matches'] == 0:
        return 0.5
    
    win_rate = stats['wins'] / stats['matches']
    goal_diff = (stats['goals_for'] - stats['goals_against']) / (stats['matches'] * 3)
    
    return min(1.0, max(0.0, 0.5 + win_rate * 0.3 + goal_diff * 0.2))


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load match data using rolling team strength."""
    print("Loading training data...")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT home_team, away_team, home_goals, away_goals, result
            FROM matches 
            WHERE result IS NOT NULL
            ORDER BY match_date ASC
        """)
        matches = cursor.fetchall()
    
    if len(matches) < 100:
        print("Not enough data! Using synthetic...")
        return _generate_sample_data()
    
    print(f"Loaded {len(matches)} matches")
    
    X = []
    y = []
    skipped = 0
    
    # Start from match 50 to have some history
    for i in range(50, len(matches)):
        m = matches[i]
        
        # Calculate team stats from PREVIOUS matches only
        stats = calculate_team_stats(matches, i)
        
        home = m['home_team']
        away = m['away_team']
        
        if stats[home]['matches'] < 3 or stats[away]['matches'] < 3:
            skipped += 1
            continue
        
        home_strength = get_team_strength(stats[home])
        away_strength = get_team_strength(stats[away])
        
        # Features based on historical performance
        features = [
            home_strength,
            away_strength,
            home_strength - away_strength,
            home_strength * 1.1,  # Home advantage
            stats[home]['wins'] / max(stats[home]['matches'], 1),
            stats[away]['wins'] / max(stats[away]['matches'], 1),
            (stats[home]['goals_for'] / max(stats[home]['matches'], 1)) / 3.0,
            (stats[away]['goals_for'] / max(stats[away]['matches'], 1)) / 3.0,
            0.45,  # Historical home win rate baseline
            home_strength + 0.05  # Slight home advantage
        ]
        
        X.append(features)
        y.append(m['result'])
    
    print(f"Using {len(X)} matches (skipped {skipped} with insufficient history)")
    
    feature_names = [
        'home_strength', 'away_strength', 'strength_diff', 'home_adj_strength',
        'home_win_rate', 'away_win_rate', 'home_attack', 'away_attack',
        'baseline', 'home_advantage'
    ]
    
    return np.array(X), np.array(y), feature_names


def _generate_sample_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate sample data."""
    np.random.seed(42)
    X = []
    y = []
    
    for _ in range(500):
        home_str = np.random.uniform(0.3, 0.8)
        away_str = np.random.uniform(0.3, 0.8)
        
        features = [
            home_str, away_str, home_str - away_str, home_str * 1.1,
            np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6),
            np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7),
            0.45, home_str + 0.05
        ]
        X.append(features)
        
        # Probability based on strength
        home_prob = 0.35 + (home_str - away_str) * 0.3
        draw_prob = 0.28
        
        rand = np.random.random()
        if rand < home_prob:
            y.append('H')
        elif rand < home_prob + draw_prob:
            y.append('D')
        else:
            y.append('A')
    
    return np.array(X), np.array(y), ['f'+ str(i) for i in range(10)]


def train_and_save():
    """Train and save model."""
    print("=" * 50)
    print("TRAINING ML MODEL")
    print("=" * 50)
    
    X, y, feature_names = load_training_data()
    print(f"\nSamples: {len(X)}, Features: {len(feature_names)}")
    
    for label in ['H', 'D', 'A']:
        count = np.sum(y == label)
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    # Split (chronological - last 20% for test)
    n_train = int(0.8 * len(X))
    X_train, X_test = X_norm[:n_train], X_norm[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    model = SimpleLogisticClassifier(learning_rate=0.1, n_iterations=1500)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\n*** Train Accuracy: {train_acc*100:.1f}% ***")
    print(f"*** Test Accuracy:  {test_acc*100:.1f}% ***")
    
    # Per-class accuracy
    y_pred = model.predict(X_test)
    print("\nPer-class (test):")
    for label in ['H', 'D', 'A']:
        mask = y_test == label
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == label)
            print(f"  {label}: {acc*100:.1f}% ({mask.sum()} samples)")
    
    # Save
    with open(MODELS_DIR / "simple_model.pkl", 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'normalization': {'mean': X_mean, 'std': X_std},
            'accuracy': test_acc,
            'trained_at': datetime.now().isoformat()
        }, f)
    
    print(f"\nModel saved to trained_models/simple_model.pkl")
    
    # Log
    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO model_logs (model_type, accuracy, training_samples, notes) VALUES (?, ?, ?, ?)",
            ('SimpleLogistic', test_acc, len(X), f"Rolling strength features")
        )
    
    return test_acc


if __name__ == "__main__":
    train_and_save()

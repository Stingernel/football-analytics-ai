"""
Advanced ML Training with Elo Ratings and Better Features.
Implements multiple accuracy improvements.
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

# Starting Elo rating
INITIAL_ELO = 1500
K_FACTOR = 32  # How much Elo changes per match


class EloSystem:
    """Elo rating system for football teams."""
    
    def __init__(self):
        self.ratings = defaultdict(lambda: INITIAL_ELO)
        self.history = defaultdict(list)  # Track rating history
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update(self, home_team: str, away_team: str, result: str):
        """Update Elo ratings after a match."""
        home_elo = self.ratings[home_team]
        away_elo = self.ratings[away_team]
        
        # Actual scores (W=1, D=0.5, L=0)
        if result == 'H':
            home_actual, away_actual = 1.0, 0.0
        elif result == 'A':
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Expected scores (with home advantage +50)
        home_expected = self.expected_score(home_elo + 50, away_elo)
        away_expected = 1 - home_expected
        
        # Update ratings
        self.ratings[home_team] = home_elo + K_FACTOR * (home_actual - home_expected)
        self.ratings[away_team] = away_elo + K_FACTOR * (away_actual - away_expected)
        
        # Track history
        self.history[home_team].append(self.ratings[home_team])
        self.history[away_team].append(self.ratings[away_team])
    
    def get_rating(self, team: str) -> float:
        return self.ratings[team]
    
    def get_momentum(self, team: str, n: int = 5) -> float:
        """Calculate rating momentum (trend)."""
        hist = self.history[team]
        if len(hist) < n:
            return 0.0
        
        recent = hist[-n:]
        return (recent[-1] - recent[0]) / 100  # Normalized


class FormTracker:
    """Track team form with goals and performance."""
    
    def __init__(self):
        self.recent_results = defaultdict(list)  # List of (result, goals_for, goals_against)
    
    def update(self, team: str, result: str, goals_for: int, goals_against: int, is_home: bool):
        self.recent_results[team].append({
            'result': result,
            'gf': goals_for,
            'ga': goals_against,
            'home': is_home
        })
        # Keep only last 10
        if len(self.recent_results[team]) > 10:
            self.recent_results[team] = self.recent_results[team][-10:]
    
    def get_form_score(self, team: str, n: int = 5) -> float:
        """Calculate weighted form score (recent games weighted more)."""
        results = self.recent_results[team][-n:]
        if not results:
            return 0.5
        
        score = 0
        weights = [i + 1 for i in range(len(results))]  # More recent = higher weight
        total_weight = sum(weights)
        
        for i, r in enumerate(results):
            if r['result'] == 'H' or r['result'] == 'W':
                match_score = 1.0
            elif r['result'] == 'A' or r['result'] == 'L':
                match_score = 0.0
            else:
                match_score = 0.4  # Draw slightly below average
            
            score += match_score * weights[i]
        
        return score / total_weight
    
    def get_attack_rating(self, team: str, n: int = 5) -> float:
        """Goals scored per game (normalized)."""
        results = self.recent_results[team][-n:]
        if not results:
            return 0.5
        
        total_goals = sum(r['gf'] for r in results)
        return min(1.0, total_goals / (len(results) * 2.5))
    
    def get_defense_rating(self, team: str, n: int = 5) -> float:
        """Goals conceded per game (inverted, normalized)."""
        results = self.recent_results[team][-n:]
        if not results:
            return 0.5
        
        total_conceded = sum(r['ga'] for r in results)
        # Lower is better, so invert
        return max(0.0, 1.0 - total_conceded / (len(results) * 2.5))


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load data with advanced Elo and form features."""
    print("Loading training data with advanced features...")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT home_team, away_team, home_goals, away_goals, result, match_date,
                   home_odds, draw_odds, away_odds
            FROM matches 
            WHERE result IS NOT NULL
            ORDER BY match_date ASC
        """)
        matches = cursor.fetchall()
    
    if len(matches) < 100:
        print("Not enough data!")
        return None, None, None
    
    print(f"Loaded {len(matches)} matches")
    
    # Initialize tracking systems
    elo = EloSystem()
    form = FormTracker()
    
    X = []
    y = []
    
    # Process each match chronologically
    warmup_matches = 100  # Use first N matches just for building ratings
    
    for i, m in enumerate(matches):
        home = m['home_team']
        away = m['away_team']
        result = m['result']
        home_goals = int(m['home_goals']) if m['home_goals'] else 0
        away_goals = int(m['away_goals']) if m['away_goals'] else 0
        
        # Get odds from The Odds API or generated
        home_odds = float(m['home_odds']) if m['home_odds'] else 2.5
        draw_odds = float(m['draw_odds']) if m['draw_odds'] else 3.3
        away_odds = float(m['away_odds']) if m['away_odds'] else 3.0
        
        # Only add to training after warmup
        if i >= warmup_matches:
            # Get current ratings BEFORE updating
            home_elo = elo.get_rating(home)
            away_elo = elo.get_rating(away)
            home_momentum = elo.get_momentum(home)
            away_momentum = elo.get_momentum(away)
            
            home_form = form.get_form_score(home)
            away_form = form.get_form_score(away)
            home_attack = form.get_attack_rating(home)
            away_attack = form.get_attack_rating(away)
            home_defense = form.get_defense_rating(home)
            away_defense = form.get_defense_rating(away)
            
            # Calculate expected probabilities from Elo
            elo_diff = home_elo - away_elo + 50  # +50 home advantage
            home_expected = elo.expected_score(home_elo + 50, away_elo)
            
            # Calculate implied probabilities from odds (bookmaker wisdom)
            odds_total = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            home_implied = (1/home_odds) / odds_total
            draw_implied = (1/draw_odds) / odds_total
            away_implied = (1/away_odds) / odds_total
            
            # Features (now with odds-based features)
            features = [
                (home_elo - 1500) / 300,  # Normalized Elo
                (away_elo - 1500) / 300,
                elo_diff / 200,           # Elo difference
                home_expected,            # Expected win prob from Elo
                home_momentum,            # Rating trend
                away_momentum,
                home_form,                # Recent results
                away_form,
                home_attack,              # Scoring ability
                away_attack,
                home_defense,             # Defensive strength
                away_defense,
                home_attack - away_defense,  # Attack vs Defense matchup
                away_attack - home_defense,
                home_form - away_form,    # Form difference
                home_implied,             # Bookmaker implied probabilities
                draw_implied,
                away_implied,
                home_implied - away_implied,  # Odds difference
                1.0                       # Baseline/bias
            ]
            
            X.append(features)
            y.append(result)
        
        # Update ratings and form AFTER getting features
        elo.update(home, away, result)
        
        home_result = 'W' if result == 'H' else ('D' if result == 'D' else 'L')
        away_result = 'W' if result == 'A' else ('D' if result == 'D' else 'L')
        form.update(home, home_result, home_goals, away_goals, True)
        form.update(away, away_result, away_goals, home_goals, False)
    
    print(f"Training samples: {len(X)}")
    
    feature_names = [
        'home_elo_norm', 'away_elo_norm', 'elo_diff', 'elo_expected',
        'home_momentum', 'away_momentum', 'home_form', 'away_form',
        'home_attack', 'away_attack', 'home_defense', 'away_defense',
        'home_attack_matchup', 'away_attack_matchup', 'form_diff',
        'home_implied', 'draw_implied', 'away_implied', 'odds_diff', 'bias'
    ]
    
    return np.array(X), np.array(y), feature_names


def train_ensemble() -> float:
    """Train multiple models and combine predictions."""
    print("=" * 60)
    print("ADVANCED ML TRAINING")
    print("=" * 60)
    
    X, y, feature_names = load_training_data()
    if X is None:
        print("No data available!")
        return 0.0
    
    print(f"\nSamples: {len(X)}, Features: {len(feature_names)}")
    for label in ['H', 'D', 'A']:
        count = np.sum(y == label)
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    # Chronological split
    n_train = int(0.8 * len(X))
    X_train, X_test = X_norm[:n_train], X_norm[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\nTrain: {n_train}, Test: {len(y_test)}")
    
    # Train multiple models with different learning rates
    models = []
    weights = []
    
    for lr in [0.05, 0.1, 0.2]:
        for iters in [1000, 1500, 2000]:
            print(f"\nTraining model lr={lr}, iters={iters}...")
            model = SimpleLogisticClassifier(learning_rate=lr, n_iterations=iters)
            model.fit(X_train, y_train)
            
            acc = model.score(X_test, y_test)
            print(f"  Accuracy: {acc*100:.1f}%")
            
            models.append(model)
            weights.append(acc)  # Weight by accuracy
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Ensemble prediction
    print("\nEnsemble averaging...")
    ensemble_preds = np.zeros((len(X_test), 3))
    
    for model, weight in zip(models, weights):
        probs = model.predict_proba(X_test)
        ensemble_preds += probs * weight
    
    # Final prediction
    classes = models[0].classes_
    y_pred_idx = np.argmax(ensemble_preds, axis=1)
    y_pred = classes[y_pred_idx]
    
    ensemble_acc = np.mean(y_pred == y_test)
    
    print(f"\n{'=' * 40}")
    print(f"*** ENSEMBLE ACCURACY: {ensemble_acc*100:.1f}% ***")
    print(f"{'=' * 40}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for label in ['H', 'D', 'A']:
        mask = y_test == label
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == label)
            print(f"  {label}: {acc*100:.1f}% ({mask.sum()} samples)")
    
    # Save best model + ensemble info
    best_idx = np.argmax(weights)
    
    with open(MODELS_DIR / "simple_model.pkl", 'wb') as f:
        pickle.dump({
            'model': models[best_idx],
            'feature_names': feature_names,
            'normalization': {'mean': X_mean, 'std': X_std},
            'accuracy': ensemble_acc,
            'trained_at': datetime.now().isoformat(),
            'type': 'advanced_elo_features'
        }, f)
    
    # Save ensemble
    with open(MODELS_DIR / "ensemble_model.pkl", 'wb') as f:
        pickle.dump({
            'models': models,
            'weights': weights,
            'feature_names': feature_names,
            'normalization': {'mean': X_mean, 'std': X_std},
            'accuracy': ensemble_acc
        }, f)
    
    print(f"\nModels saved to trained_models/")
    
    # Log to database
    with get_db() as conn:
        conn.cursor().execute(
            "INSERT INTO model_logs (model_type, accuracy, training_samples, notes) VALUES (?, ?, ?, ?)",
            ('AdvancedEnsemble', ensemble_acc, len(X), 'Elo + Form + Attack/Defense features')
        )
    
    return ensemble_acc


if __name__ == "__main__":
    train_ensemble()

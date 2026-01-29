"""
ML Model Training Pipeline
Trains XGBoost and LSTM models using historical match data.
"""
import sys
import logging
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path(__file__).parent / "trained_models"
MODELS_DIR.mkdir(exist_ok=True)


def load_training_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load match data from database for training.
    
    Returns:
        X: Feature matrix
        y: Labels (H, D, A)
        feature_names: List of feature names
    """
    logger.info("Loading training data from database...")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                home_team, away_team, 
                home_odds, draw_odds, away_odds,
                home_xg, away_xg,
                result
            FROM matches 
            WHERE result IS NOT NULL 
              AND home_odds IS NOT NULL
              AND draw_odds IS NOT NULL
              AND away_odds IS NOT NULL
            ORDER BY match_date DESC
            LIMIT 1000
        """)
        matches = cursor.fetchall()
    
    if not matches:
        logger.warning("No training data found! Using sample data...")
        return _generate_sample_data()
    
    logger.info(f"Loaded {len(matches)} matches")
    
    X = []
    y = []
    
    for m in matches:
        home_odds = float(m['home_odds']) if m['home_odds'] else 2.0
        draw_odds = float(m['draw_odds']) if m['draw_odds'] else 3.5
        away_odds = float(m['away_odds']) if m['away_odds'] else 3.0
        home_xg = float(m['home_xg']) if m['home_xg'] else 1.5
        away_xg = float(m['away_xg']) if m['away_xg'] else 1.2
        
        # Calculate implied probabilities
        total_prob = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        home_implied = (1/home_odds) / total_prob
        draw_implied = (1/draw_odds) / total_prob
        away_implied = (1/away_odds) / total_prob
        
        # Features
        features = [
            home_implied,              # 0: Home implied probability
            draw_implied,              # 1: Draw implied probability  
            away_implied,              # 2: Away implied probability
            home_xg,                   # 3: Home xG
            away_xg,                   # 4: Away xG
            home_xg - away_xg,         # 5: xG difference
            home_odds,                 # 6: Home odds
            away_odds,                 # 7: Away odds
            1.0,                       # 8: Home advantage (placeholder)
            home_implied - away_implied # 9: Probability difference
        ]
        
        X.append(features)
        y.append(m['result'])
    
    feature_names = [
        'home_implied_prob', 'draw_implied_prob', 'away_implied_prob',
        'home_xg', 'away_xg', 'xg_diff',
        'home_odds', 'away_odds', 'home_advantage', 'prob_diff'
    ]
    
    return np.array(X), np.array(y), feature_names


def _generate_sample_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate sample training data if database is empty."""
    logger.info("Generating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 500
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate random odds (realistic range)
        home_odds = np.random.uniform(1.3, 4.0)
        draw_odds = np.random.uniform(2.8, 4.5)
        away_odds = np.random.uniform(1.5, 6.0)
        
        # Calculate implied probabilities
        total = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        home_implied = (1/home_odds) / total
        draw_implied = (1/draw_odds) / total
        away_implied = (1/away_odds) / total
        
        # Random xG
        home_xg = np.random.uniform(0.8, 2.8)
        away_xg = np.random.uniform(0.6, 2.5)
        
        features = [
            home_implied, draw_implied, away_implied,
            home_xg, away_xg, home_xg - away_xg,
            home_odds, away_odds, 1.0, home_implied - away_implied
        ]
        X.append(features)
        
        # Label based on probabilities (realistic distribution)
        rand = np.random.random()
        if rand < home_implied * 0.9:  # Home wins
            y.append('H')
        elif rand < (home_implied * 0.9 + draw_implied * 1.1):  # Draw
            y.append('D')
        else:  # Away wins
            y.append('A')
    
    feature_names = [
        'home_implied_prob', 'draw_implied_prob', 'away_implied_prob',
        'home_xg', 'away_xg', 'xg_diff',
        'home_odds', 'away_odds', 'home_advantage', 'prob_diff'
    ]
    
    return np.array(X), np.array(y), feature_names


def train_xgboost(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
    """Train XGBoost classifier."""
    logger.info("Training XGBoost model...")
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        
        # Parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': 42
        }
        
        # Train
        num_rounds = 100
        evals = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_rounds, evals=evals, verbose_eval=20)
        
        # Evaluate
        y_pred_proba = model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_path = MODELS_DIR / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'label_encoder': label_encoder,
                'feature_names': feature_names,
                'accuracy': accuracy,
                'trained_at': datetime.now().isoformat()
            }, f)
        
        logger.info(f"XGBoost model saved to {model_path}")
        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        return {
            'model_type': 'XGBoost',
            'accuracy': accuracy,
            'model_path': str(model_path),
            'samples': len(X)
        }
        
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return {'error': 'XGBoost not installed'}


def train_lstm(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train LSTM model for sequence prediction."""
    logger.info("Training LSTM model...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels to one-hot
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=3)
        
        # Reshape X for LSTM: (samples, timesteps, features)
        # We'll treat each sample as a sequence of length 1 with 10 features
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_onehot, test_size=0.2, random_state=42
        )
        
        # Build model
        model = models.Sequential([
            layers.LSTM(64, input_shape=(1, X.shape[1]), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Save model
        model_path = MODELS_DIR / "lstm_model.h5"
        model.save(model_path)
        
        # Save encoder
        encoder_path = MODELS_DIR / "lstm_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump({
                'label_encoder': label_encoder,
                'accuracy': accuracy,
                'trained_at': datetime.now().isoformat()
            }, f)
        
        logger.info(f"LSTM model saved to {model_path}")
        logger.info(f"LSTM Accuracy: {accuracy:.4f}")
        
        return {
            'model_type': 'LSTM',
            'accuracy': accuracy,
            'model_path': str(model_path),
            'samples': len(X)
        }
        
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow")
        return {'error': 'TensorFlow not installed'}


def log_training_results(results: List[Dict]):
    """Log training results to database."""
    with get_db() as conn:
        cursor = conn.cursor()
        for r in results:
            if 'error' not in r:
                cursor.execute("""
                    INSERT INTO model_logs (model_type, accuracy, training_samples, notes)
                    VALUES (?, ?, ?, ?)
                """, (
                    r['model_type'],
                    r['accuracy'],
                    r['samples'],
                    f"Auto-trained via pipeline at {datetime.now().isoformat()}"
                ))
        conn.commit()
    logger.info("Training results logged to database")


def main():
    """Run full training pipeline."""
    print("=" * 60)
    print("[*] Football Analytics ML Training Pipeline")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_training_data()
    print(f"\n[i] Training Data: {len(X)} samples")
    print(f"   Features: {len(feature_names)}")
    print(f"   Labels distribution:")
    for label in ['H', 'D', 'A']:
        count = np.sum(y == label)
        print(f"     - {label}: {count} ({count/len(y)*100:.1f}%)")
    
    results = []
    
    # Train XGBoost
    print("\n" + "=" * 60)
    print("[1] Training XGBoost...")
    print("=" * 60)
    xgb_result = train_xgboost(X, y, feature_names)
    results.append(xgb_result)
    
    # Train LSTM
    print("\n" + "=" * 60)
    print("[2] Training LSTM...")
    print("=" * 60)
    lstm_result = train_lstm(X, y)
    results.append(lstm_result)
    
    # Log results
    log_training_results(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("[OK] Training Complete!")
    print("=" * 60)
    for r in results:
        if 'error' not in r:
            print(f"   {r['model_type']}: {r['accuracy']*100:.1f}% accuracy")
        else:
            print(f"   {r.get('model_type', 'Unknown')}: {r['error']}")
    
    print(f"\n[>] Models saved to: {MODELS_DIR}")
    print("   - xgboost_model.pkl")
    print("   - lstm_model.h5")
    print("   - lstm_encoder.pkl")


if __name__ == "__main__":
    main()

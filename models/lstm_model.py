"""
LSTM model for football match sequence prediction.
Captures temporal patterns in team form.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM model for sequence-based match prediction."""
    
    def __init__(self, model_path: Optional[Path] = None, sequence_length: int = 5):
        """
        Initialize LSTM predictor.
        
        Args:
            model_path: Path to saved model
            sequence_length: Number of past matches to consider
        """
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available. Using demo mode.")
        
        self.sequence_length = sequence_length
        self.feature_dim = 8  # Features per match in sequence
        self.model: Optional[keras.Model] = None
        
        if model_path and model_path.exists() and HAS_TENSORFLOW:
            self.load(model_path)
        elif HAS_TENSORFLOW:
            self._build_model()
    
    def _build_model(self):
        """Build the LSTM architecture."""
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim))
        
        # LSTM layers
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        x = layers.LSTM(32, dropout=0.2)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output (3 classes: H, D, A)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("LSTM model built successfully")
    
    def _prepare_sequence(self, match_data: Dict) -> np.ndarray:
        """
        Prepare sequence features from match data.
        
        Expected match_data keys:
        - home_form_sequence: List of last N match results/stats for home team
        - away_form_sequence: List of last N match results/stats for away team
        
        If not available, uses form string and xG data.
        """
        # If we have proper sequence data
        if 'home_form_sequence' in match_data:
            home_seq = match_data['home_form_sequence']
            away_seq = match_data['away_form_sequence']
            
            # Combine sequences
            combined = []
            for h, a in zip(home_seq[-self.sequence_length:], 
                           away_seq[-self.sequence_length:]):
                features = [
                    h.get('goals_for', 1.5) / 5,
                    h.get('goals_against', 1.0) / 5,
                    h.get('xg', 1.5) / 3,
                    h.get('result_points', 1) / 3,
                    a.get('goals_for', 1.0) / 5,
                    a.get('goals_against', 1.5) / 5,
                    a.get('xg', 1.0) / 3,
                    a.get('result_points', 1) / 3,
                ]
                combined.append(features)
            
            return np.array(combined).reshape(1, self.sequence_length, self.feature_dim)
        
        # Fallback: construct from form string
        def form_to_sequence(form: str, is_home: bool) -> List[List[float]]:
            """Convert form string to sequence."""
            seq = []
            for i, char in enumerate(form.upper()[:self.sequence_length]):
                if char == 'W':
                    points = 1.0
                    gf, ga = (2.0, 0.5) if is_home else (1.5, 0.5)
                elif char == 'D':
                    points = 0.33
                    gf, ga = 1.0, 1.0
                else:  # L
                    points = 0.0
                    gf, ga = (0.5, 2.0) if is_home else (0.5, 1.5)
                
                xg = match_data.get('home_xg' if is_home else 'away_xg', 1.3)
                seq.append([gf/5, ga/5, xg/3, points])
            
            # Pad if needed
            while len(seq) < self.sequence_length:
                seq.insert(0, [0.3, 0.3, 0.4, 0.33])
            
            return seq[-self.sequence_length:]
        
        home_form = match_data.get('home_form', 'DDDDD')
        away_form = match_data.get('away_form', 'DDDDD')
        
        home_seq = form_to_sequence(home_form, is_home=True)
        away_seq = form_to_sequence(away_form, is_home=False)
        
        # Combine into feature matrix
        combined = []
        for h, a in zip(home_seq, away_seq):
            combined.append(h + a)  # 4 + 4 = 8 features
        
        return np.array(combined).reshape(1, self.sequence_length, self.feature_dim)
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """
        Predict match outcome from sequence data.
        
        Args:
            match_data: Dictionary with form sequences
            
        Returns:
            Probability distribution
        """
        if self.model is None or not HAS_TENSORFLOW:
            # Fallback to demo prediction
            return self._demo_predict(match_data)
        
        try:
            sequence = self._prepare_sequence(match_data)
            probabilities = self.model.predict(sequence, verbose=0)[0]
            
            return {
                'H': float(probabilities[0]),
                'D': float(probabilities[1]),
                'A': float(probabilities[2])
            }
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return self._demo_predict(match_data)
    
    def _demo_predict(self, match_data: Dict) -> Dict[str, float]:
        """Demo prediction based on form analysis."""
        home_form = match_data.get('home_form', 'DDD')
        away_form = match_data.get('away_form', 'DDD')
        
        def analyze_form(form: str) -> float:
            """Calculate form strength (0-1)."""
            if not form:
                return 0.5
            score = 0
            weights = [0.35, 0.25, 0.20, 0.12, 0.08]  # Recent > older
            for i, char in enumerate(form.upper()[:5]):
                weight = weights[i] if i < 5 else 0.05
                if char == 'W':
                    score += weight * 1.0
                elif char == 'D':
                    score += weight * 0.4
            return min(1.0, score)
        
        home_strength = analyze_form(home_form)
        away_strength = analyze_form(away_form)
        
        # Add xG influence
        home_xg = match_data.get('home_xg', 1.5)
        away_xg = match_data.get('away_xg', 1.2)
        xg_diff = (home_xg - away_xg) / 3  # Normalize
        
        # Base probabilities
        home_base = 0.45 + (home_strength - away_strength) * 0.2 + xg_diff * 0.1
        away_base = 0.30 - (home_strength - away_strength) * 0.2 - xg_diff * 0.1
        
        # Clamp values
        home_prob = max(0.15, min(0.70, home_base))
        away_prob = max(0.10, min(0.55, away_base))
        draw_prob = 1 - home_prob - away_prob
        draw_prob = max(0.15, min(0.40, draw_prob))
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        
        return {
            'H': round(home_prob / total, 4),
            'D': round(draw_prob / total, 4),
            'A': round(away_prob / total, 4)
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 50, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the LSTM model.
        
        Args:
            X: Sequence data (samples, sequence_length, features)
            y: One-hot encoded labels
            epochs: Training epochs
            validation_split: Validation fraction
            
        Returns:
            Training metrics
        """
        if not HAS_TENSORFLOW or self.model is None:
            logger.warning("Cannot train: TensorFlow not available")
            return {'accuracy': 0, 'loss': 0}
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        metrics = {
            'accuracy': float(history.history['val_accuracy'][-1]),
            'loss': float(history.history['val_loss'][-1])
        }
        
        logger.info(f"LSTM training complete: {metrics}")
        return metrics
    
    def save(self, path: Path):
        """Save model to file."""
        if self.model:
            self.model.save(path)
            logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        if HAS_TENSORFLOW:
            self.model = keras.models.load_model(path)
            logger.info(f"LSTM model loaded from {path}")


# Demo version for testing without TensorFlow
class DemoLSTMPredictor(LSTMPredictor):
    """Demo predictor that doesn't require TensorFlow."""
    
    def __init__(self):
        self.sequence_length = 5
        self.feature_dim = 8
        self.model = None
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Use demo prediction."""
        return self._demo_predict(match_data)

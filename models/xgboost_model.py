"""
XGBoost model for football match outcome prediction.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost classifier for match outcome prediction."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the XGBoost predictor.
        
        Args:
            model_path: Path to saved model file (optional)
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['A', 'D', 'H'])  # Away, Draw, Home
        self.feature_names = [
            'home_form_pts',
            'away_form_pts', 
            'home_xg_avg',
            'away_xg_avg',
            'home_goals_avg',
            'away_goals_avg',
            'home_conceded_avg',
            'away_conceded_avg',
            'home_implied_prob',
            'draw_implied_prob',
            'away_implied_prob',
            'h2h_home_wins',
            'h2h_draws',
            'h2h_away_wins'
        ]
        
        if model_path and model_path.exists():
            self.load(model_path)
        else:
            self._init_default_model()
    
    def _init_default_model(self):
        """Initialize with default hyperparameters."""
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )
        logger.info("Initialized default XGBoost model")
    
    def _prepare_features(self, match_data: Dict) -> np.ndarray:
        """
        Prepare feature vector from match data.
        
        Args:
            match_data: Dictionary with match information
            
        Returns:
            Feature array
        """
        # Convert form string to points (W=3, D=1, L=0)
        def form_to_points(form: str) -> float:
            if not form:
                return 0.5  # Default
            points = 0
            for char in form.upper()[:5]:
                if char == 'W':
                    points += 3
                elif char == 'D':
                    points += 1
            return points / 15  # Normalize to 0-1
        
        # Calculate implied probabilities from odds
        def odds_to_prob(odds: float) -> float:
            return 1 / odds if odds > 0 else 0.33
        
        features = [
            form_to_points(match_data.get('home_form', '')),
            form_to_points(match_data.get('away_form', '')),
            match_data.get('home_xg', 1.5) / 3.0,  # Normalize
            match_data.get('away_xg', 1.0) / 3.0,
            match_data.get('home_goals_avg', 1.5) / 3.0,
            match_data.get('away_goals_avg', 1.0) / 3.0,
            match_data.get('home_conceded_avg', 1.0) / 3.0,
            match_data.get('away_conceded_avg', 1.5) / 3.0,
            odds_to_prob(match_data.get('home_odds', 2.0)),
            odds_to_prob(match_data.get('draw_odds', 3.5)),
            odds_to_prob(match_data.get('away_odds', 4.0)),
            match_data.get('h2h_home_wins', 0.4),
            match_data.get('h2h_draws', 0.3),
            match_data.get('h2h_away_wins', 0.3),
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """
        Predict match outcome probabilities.
        
        Args:
            match_data: Dictionary with match information
            
        Returns:
            Dictionary with probabilities for each outcome
        """
        if self.model is None:
            # Return default probabilities if no model
            return {'H': 0.45, 'D': 0.28, 'A': 0.27}
        
        try:
            features = self._prepare_features(match_data)
            probabilities = self.model.predict_proba(features)[0]
            
            # Map to result labels (sorted: A, D, H)
            return {
                'A': float(probabilities[0]),
                'D': float(probabilities[1]),
                'H': float(probabilities[2])
            }
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return {'H': 0.45, 'D': 0.28, 'A': 0.27}
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target labels (H, D, A)
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, log_loss
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42
        )
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'log_loss': log_loss(y_val, y_proba)
        }
        
        logger.info(f"XGBoost training complete: {metrics}")
        return metrics
    
    def save(self, path: Path):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"XGBoost model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.feature_names = data.get('feature_names', self.feature_names)
        logger.info(f"XGBoost model loaded from {path}")


# Demo predictor with hardcoded logic for testing
class DemoXGBoostPredictor:
    """Demo predictor that uses trained simple model or odds-based heuristics."""
    
    def __init__(self):
        self.feature_names = []
        self.model = None
        self.normalization = None
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Try to load trained simple model."""
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "trained_models" / "simple_model.pkl"
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.normalization = data.get('normalization', {})
                self.feature_names = data.get('feature_names', [])
                logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None

    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Generate probabilities using trained model or fallback to heuristics."""
        
        # If we have a trained model, use it!
        if self.model is not None and self.normalization:
            try:
                return self._predict_with_model(match_data)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}, falling back")
        
        # Fallback to odds-based heuristics
        return self._predict_heuristic(match_data)
    
    def _predict_with_model(self, match_data: Dict) -> Dict[str, float]:
        """Use trained SimpleLogisticClassifier model."""
        import numpy as np
        
        home_odds = match_data.get('home_odds', 2.0)
        draw_odds = match_data.get('draw_odds', 3.5)
        away_odds = match_data.get('away_odds', 3.0)
        
        # Calculate implied probabilities from odds
        odds_total = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        home_implied = (1/home_odds) / odds_total
        draw_implied = (1/draw_odds) / odds_total
        away_implied = (1/away_odds) / odds_total
        
        # Form scores
        home_form = match_data.get('home_form', 'DDD')
        away_form = match_data.get('away_form', 'DDD')
        
        def form_to_score(form):
            score = 0
            for c in str(form).upper()[:5]:
                if c == 'W': score += 0.2
                elif c == 'D': score += 0.08
            return min(1.0, score)
        
        home_form_score = form_to_score(home_form)
        away_form_score = form_to_score(away_form)
        
        # Approximate features from available data
        home_elo_norm = (home_implied - 0.33) * 2
        away_elo_norm = (away_implied - 0.33) * 2
        elo_diff = home_implied - away_implied
        
        home_xg = match_data.get('home_xg', 1.5)
        away_xg = match_data.get('away_xg', 1.2)
        home_attack = min(1.0, home_xg / 3.0)
        away_attack = min(1.0, away_xg / 3.0)
        
        # MATCHING FEATURES WITH train_simple.py (10 Features)
        # 1. home_strength
        # 2. away_strength
        # 3. strength_diff
        # 4. home_adj_strength
        # 5. home_win_rate (approx from odds)
        # 6. away_win_rate (approx from odds)
        # 7. home_attack (from xG)
        # 8. away_attack (from xG)
        # 9. baseline (const 0.45)
        # 10. home_advantage (home_strength + 0.05)
        
        # Calculate features
        f1_home_str = home_form_score * 0.7 + (home_implied * 0.3) # Blend form & odds
        f2_away_str = away_form_score * 0.7 + (away_implied * 0.3)
        f3_diff = f1_home_str - f2_away_str
        f4_home_adj = f1_home_str * 1.1
        f5_home_win = home_implied
        f6_away_win = away_implied
        f7_home_att = home_attack
        f8_away_att = away_attack
        f9_baseline = 0.45
        f10_home_adv = f1_home_str + 0.05
        
        features = np.array([[
            f1_home_str, f2_away_str, f3_diff, f4_home_adj,
            f5_home_win, f6_away_win, f7_home_att, f8_away_att,
            f9_baseline, f10_home_adv
        ]])

        
        # Normalize
        # Ensure we use 10 features defaults if normalization dict missing
        mean = self.normalization.get('mean', np.zeros(10))
        std = self.normalization.get('std', np.ones(10))
        
        # Protect against shape mismatch if old model file loaded
        if mean.shape[0] != features.shape[1]:
            logger.warning(f"Shape mismatch: Model expects {mean.shape[0]}, got {features.shape[1]}. Using zeros.")
            mean = np.zeros(features.shape[1])
            std = np.ones(features.shape[1])

        features_norm = (features - mean) / (std + 1e-10)
        
        # Predict
        probs = self.model.predict_proba(features_norm)[0]
        classes = self.model.classes_
        
        result = {}
        for i, cls in enumerate(classes):
            result[cls] = round(float(probs[i]), 4)
        return result
    
    def _predict_heuristic(self, match_data: Dict) -> Dict[str, float]:
        """Fallback: odds-based heuristics."""
        home_odds = match_data.get('home_odds', 2.0)
        draw_odds = match_data.get('draw_odds', 3.5)
        away_odds = match_data.get('away_odds', 4.0)
        
        total = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        home_prob = (1/home_odds) / total
        draw_prob = (1/draw_odds) / total
        away_prob = (1/away_odds) / total
        
        home_form = match_data.get('home_form', '')
        away_form = match_data.get('away_form', '')
        
        home_adj = sum(0.02 if c=='W' else -0.02 if c=='L' else 0 for c in str(home_form).upper()[:3])
        away_adj = sum(0.02 if c=='W' else -0.02 if c=='L' else 0 for c in str(away_form).upper()[:3])
        
        home_prob = max(0.05, min(0.85, home_prob + home_adj - away_adj * 0.5))
        away_prob = max(0.05, min(0.85, away_prob + away_adj - home_adj * 0.5))
        draw_prob = max(0.10, 1 - home_prob - away_prob)
        
        total = home_prob + draw_prob + away_prob
        return {'H': round(home_prob/total, 4), 'D': round(draw_prob/total, 4), 'A': round(away_prob/total, 4)}

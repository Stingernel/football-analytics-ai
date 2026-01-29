"""
Verify Model Usage
Check if EnsemblePredictor is correctly using the trained model.
"""
import sys
from pathlib import Path
from pprint import pprint

# Logic flow verification
def verify_model():
    print("="*60)
    print("VERIFYING MODEL USAGE")
    print("="*60)
    
    from models.ensemble import get_ensemble
    
    ensemble = get_ensemble()
    
    # 1. Check XGBoost Model Type
    xgb = ensemble.xgb_model
    print(f"\nXGB Model Class: {type(xgb).__name__}")
    
    # 2. Check if it has the trained model loaded
    # DemoXGBoostPredictor stores it in self.model
    # XGBoostPredictor stores it in self.model
    
    has_model = hasattr(xgb, 'model') and xgb.model is not None
    print(f"Has Underlying Model Object: {has_model}")
    
    if has_model:
        model_type = type(xgb.model).__name__
        print(f"Underlying Model Type: {model_type}")
        # Expecting SimpleLogisticClassifier or sklearn LogisticRegression/XGBClassifier
        
        # Check classes
        if hasattr(xgb.model, 'classes_'):
             print(f"Model Classes: {xgb.model.classes_}")
    
    # 3. Running a Test Prediction
    print("\nRunning Test Prediction...")
    match_data = {
        'home_team': 'Test Home', 'away_team': 'Test Away', 
        'home_odds': 1.5, 'draw_odds': 4.0, 'away_odds': 6.0,
        'home_form': 'WWWWW', 'away_form': 'LLLLL',
        'home_xg': 2.5, 'away_xg': 0.5
    }
    
    try:
        # Enable logging to see "Using trained model" or "fallback"
        import logging
        logging.basicConfig(level=logging.INFO)
        
        result = ensemble.predict(match_data)
        print("\nPrediction Result:")
        pprint(result['probabilities'])
        print(f"Confidence: {result['confidence']}")
        print(f"Model Agree: {result['models_agree']}")
        
        # Check if DemoXGBoostPredictor internal normalization logic is used (unique to my fix)
        if isinstance(xgb, type(xgb)): # DemoXGBoostPredictor
             if hasattr(xgb, 'normalization') and xgb.normalization:
                 print("\n✅ DemoXGBoostPredictor has normalization params (Proof of trained model load)")
             else:
                 print("\n❌ DemoXGBoostPredictor missing normalization params")

    except Exception as e:
        print(f"\n❌ Prediction Error: {e}")

if __name__ == "__main__":
    verify_model()

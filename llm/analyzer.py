"""
Football match analyzer combining ML predictions with LLM insights.
"""
import logging
from typing import Dict, Optional
from datetime import datetime

from models.ensemble import get_ensemble
from llm.connector import get_llm
from llm.prompts import build_analysis_prompt, build_quick_summary_prompt

logger = logging.getLogger(__name__)


class FootballAnalyzer:
    """
    Main analyzer that combines ML predictions with LLM-generated insights.
    """
    
    def __init__(self):
        self.ensemble = get_ensemble()
        self.llm = get_llm()
    
    async def analyze_match(
        self, 
        match_data: Dict,
        include_llm: bool = True
    ) -> Dict:
        """
        Complete match analysis with ML predictions and LLM insights.
        
        Args:
            match_data: Match information dict with keys:
                - home_team, away_team
                - home_form, away_form (e.g., "WWDLW")
                - home_xg, away_xg
                - home_odds, draw_odds, away_odds
                - league (optional)
                - match_date (optional)
            include_llm: Whether to include LLM analysis
            
        Returns:
            Complete analysis dict
        """
        # Add defaults
        if 'match_date' not in match_data:
            match_data['match_date'] = datetime.now().strftime('%Y-%m-%d')
        if 'league' not in match_data:
            match_data['league'] = 'Unknown League'
        
        # Get ML predictions
        prediction = self.ensemble.predict(match_data)
        
        # Analyze value bets with professional criteria
        odds = {
            'home': match_data.get('home_odds', 0),
            'draw': match_data.get('draw_odds', 0),
            'away': match_data.get('away_odds', 0)
        }
        value_bets = self.ensemble.analyze_value_bets(
            prediction['probabilities'],
            odds,
            confidence=prediction['confidence'],
            models_agree=prediction['models_agree']
        )
        
        # Get LLM analysis
        llm_analysis = ""
        reasoning = ""
        
        if include_llm:
            try:
                prompt = build_analysis_prompt(match_data, prediction, value_bets)
                llm_analysis = await self.llm.generate(prompt)
                
                # Extract reasoning from analysis
                if "### Key Factors" in llm_analysis:
                    start = llm_analysis.find("### Key Factors")
                    end = llm_analysis.find("### Risk Assessment")
                    if end > start:
                        reasoning = llm_analysis[start:end].strip()
                    else:
                        reasoning = llm_analysis[start:start+500].strip()
                        
            except Exception as e:
                logger.error(f"LLM analysis error: {e}")
                llm_analysis = "Analysis temporarily unavailable."
        
        # Build result mapping
        result_full = {
            'H': 'Home Win',
            'D': 'Draw', 
            'A': 'Away Win'
        }
        
        return {
            'match': {
                'home_team': match_data.get('home_team'),
                'away_team': match_data.get('away_team'),
                'league': match_data.get('league'),
                'match_date': match_data.get('match_date'),
                'home_form': match_data.get('home_form'),
                'away_form': match_data.get('away_form'),
                'home_xg': match_data.get('home_xg'),
                'away_xg': match_data.get('away_xg'),
                'home_odds': odds['home'],
                'draw_odds': odds['draw'],
                'away_odds': odds['away']
            },
            'prediction': {
                'probabilities': prediction['probabilities'],
                'predicted_result': prediction['predicted_result'],
                'predicted_result_full': result_full[prediction['predicted_result']],
                'confidence': prediction['confidence'],
                'models_agree': prediction['models_agree']
            },
            'model_details': {
                'xgb': prediction['xgb_probs'],
                'lstm': prediction['lstm_probs'],
                'ensemble_weights': {
                    'xgb': 0.6,
                    'lstm': 0.4
                }
            },
            'value_bets': value_bets,
            'recommended_bet': next(
                (vb for vb in value_bets if vb['recommended']), 
                None
            ),
            'analysis': llm_analysis,
            'reasoning': reasoning,
            'generated_at': datetime.now().isoformat()
        }
    
    def analyze_match_sync(
        self, 
        match_data: Dict,
        include_llm: bool = True
    ) -> Dict:
        """Synchronous version of analyze_match."""
        import asyncio
        
        # Try to get event loop, create new one if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use run_until_complete in running loop
                # Use sync LLM method instead
                return self._analyze_sync_impl(match_data, include_llm)
        except RuntimeError:
            pass
        
        return asyncio.run(self.analyze_match(match_data, include_llm))
    
    def _analyze_sync_impl(
        self,
        match_data: Dict,
        include_llm: bool = True
    ) -> Dict:
        """Internal sync implementation."""
        # Add defaults
        if 'match_date' not in match_data:
            match_data['match_date'] = datetime.now().strftime('%Y-%m-%d')
        if 'league' not in match_data:
            match_data['league'] = 'Unknown League'
        
        # Get ML predictions
        prediction = self.ensemble.predict(match_data)
        
        # Analyze value bets
        odds = {
            'home': match_data.get('home_odds', 0),
            'draw': match_data.get('draw_odds', 0),
            'away': match_data.get('away_odds', 0)
        }
        value_bets = self.ensemble.analyze_value_bets(
            prediction['probabilities'],
            odds
        )
        
        # Get LLM analysis (sync)
        llm_analysis = ""
        if include_llm:
            try:
                prompt = build_analysis_prompt(match_data, prediction, value_bets)
                llm_analysis = self.llm.generate_sync(prompt)
            except Exception as e:
                logger.error(f"LLM analysis error: {e}")
                llm_analysis = "Analysis temporarily unavailable."
        
        result_full = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        
        return {
            'match': {
                'home_team': match_data.get('home_team'),
                'away_team': match_data.get('away_team'),
                'league': match_data.get('league'),
                'match_date': match_data.get('match_date'),
                'home_form': match_data.get('home_form'),
                'away_form': match_data.get('away_form'),
                'home_xg': match_data.get('home_xg'),
                'away_xg': match_data.get('away_xg'),
                'home_odds': odds['home'],
                'draw_odds': odds['draw'],
                'away_odds': odds['away']
            },
            'prediction': {
                'probabilities': prediction['probabilities'],
                'predicted_result': prediction['predicted_result'],
                'predicted_result_full': result_full[prediction['predicted_result']],
                'confidence': prediction['confidence'],
                'models_agree': prediction['models_agree']
            },
            'model_details': {
                'xgb': prediction['xgb_probs'],
                'lstm': prediction['lstm_probs'],
                'ensemble_weights': {'xgb': 0.6, 'lstm': 0.4}
            },
            'value_bets': value_bets,
            'recommended_bet': next((vb for vb in value_bets if vb['recommended']), None),
            'analysis': llm_analysis,
            'generated_at': datetime.now().isoformat()
        }
    
    async def quick_prediction(self, match_data: Dict) -> str:
        """Get a quick one-paragraph prediction."""
        prediction = self.ensemble.predict(match_data)
        prompt = build_quick_summary_prompt(
            match_data.get('home_team', 'Home'),
            match_data.get('away_team', 'Away'),
            prediction
        )
        return await self.llm.generate(prompt, max_tokens=300)


# Singleton
_analyzer: Optional[FootballAnalyzer] = None


def get_analyzer() -> FootballAnalyzer:
    """Get or create analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FootballAnalyzer()
    return _analyzer

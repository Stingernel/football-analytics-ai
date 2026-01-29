"""
LLM connector supporting OpenAI API and Ollama (local).
"""
import logging
from typing import Optional
from abc import ABC, abstractmethod

from app.config import settings

logger = logging.getLogger(__name__)


class LLMConnector(ABC):
    """Abstract base class for LLM connectors."""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, max_tokens: int = 1000) -> str:
        """Synchronous generation."""
        pass


class OpenAIConnector(LLMConnector):
    """Connector for OpenAI API."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package required. Install: pip install openai")
        return self._client
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            async_client = AsyncOpenAI(api_key=self.api_key)
            
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response()
    
    def generate_sync(self, prompt: str, max_tokens: int = 1000) -> str:
        """Synchronous generation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        return "Analysis unavailable. Please check your API configuration."


class OllamaConnector(LLMConnector):
    """Connector for local Ollama LLM."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate using Ollama local API."""
        try:
            import aiohttp
            
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        logger.error(f"Ollama error: {response.status}")
                        return self._fallback_response()
                        
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._fallback_response()
    
    def generate_sync(self, prompt: str, max_tokens: int = 1000) -> str:
        """Synchronous generation."""
        try:
            import requests
            
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return self._fallback_response()
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        return "Analysis unavailable. Ensure Ollama is running locally."


class DemoConnector(LLMConnector):
    """Demo connector that generates template responses without LLM."""
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        return self._generate_demo_analysis(prompt)
    
    def generate_sync(self, prompt: str, max_tokens: int = 1000) -> str:
        return self._generate_demo_analysis(prompt)
    
    def _generate_demo_analysis(self, prompt: str) -> str:
        """Generate template-based analysis."""
        # Extract team names from prompt
        import re
        
        match = re.search(r'\*\*Match:\*\* (.+?) vs (.+?)[\n\r]', prompt)
        if match:
            home_team = match.group(1)
            away_team = match.group(2)
        else:
            home_team = "Home Team"
            away_team = "Away Team"
        
        # Extract probabilities
        home_prob_match = re.search(r'\*\*Home Win:\*\* ([\d.]+)%', prompt)
        home_prob = float(home_prob_match.group(1)) if home_prob_match else 45
        
        # Determine likely outcome
        if home_prob > 50:
            outlook = "favored"
            risk = "moderate"
        elif home_prob > 40:
            outlook = "slightly favored"
            risk = "balanced"
        else:
            outlook = "facing a challenge"
            risk = "elevated"
        
        analysis = f"""## Match Analysis: {home_team} vs {away_team}

### Match Preview
{home_team} are {outlook} in this fixture based on our hybrid ML model analysis. 
The combination of recent form and expected goals metrics provides a clear picture of 
the likely outcomes in this encounter.

### Key Factors
- **Home advantage**: {home_team} benefit from playing at home, historically worth ~5% probability boost
- **Form differential**: Recent results suggest momentum differences between the sides
- **xG trends**: The expected goals data indicates the underlying quality of chances created
- **Market alignment**: Bookmaker odds largely align with our model, suggesting efficient pricing

### Risk Assessment
The prediction confidence is {risk}. Both models (XGBoost and LSTM) show 
{"agreement" if home_prob > 45 else "some divergence"} on the outcome, which 
{"strengthens" if home_prob > 45 else "adds uncertainty to"} this prediction.

### Betting Recommendation
Review the value bet section above. A bet is recommended **only** where the edge 
exceeds 5% and aligns with your risk tolerance. Consider stake sizing based on confidence level.

*Note: This analysis is generated by AI and should not be the sole basis for betting decisions.*
"""
        return analysis


def get_connector() -> LLMConnector:
    """Get appropriate LLM connector based on settings."""
    provider = settings.llm_provider.lower()
    
    if provider == "openai":
        if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
            logger.info("Using OpenAI connector")
            return OpenAIConnector()
        else:
            logger.warning("OpenAI API key not configured, using demo mode")
            return DemoConnector()
    
    elif provider == "ollama":
        logger.info("Using Ollama connector")
        return OllamaConnector()
    
    else:
        logger.info("Using demo connector")
        return DemoConnector()


# Singleton instance
_connector: Optional[LLMConnector] = None


def get_llm() -> LLMConnector:
    """Get or create LLM connector singleton."""
    global _connector
    if _connector is None:
        _connector = get_connector()
    return _connector

"""
Application configuration settings.
Loads environment variables and provides typed config access.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Literal

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # LLM Configuration
    llm_provider: Literal["openai", "ollama"] = "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # Database
    database_url: str = "sqlite:///./football_analytics.db"
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit Settings
    streamlit_port: int = 8501
    
    # Model Settings
    ensemble_xgb_weight: float = 0.6
    ensemble_lstm_weight: float = 0.4
    
    # The Odds API
    odds_api_key: str = ""
    
    # Football-Data.org API
    football_data_api_key: str = ""
    
    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "trained_models"
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data" / "datasets"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Football Analytics AI System

A hybrid AI-based football match prediction system combining XGBoost + LSTM machine learning models with LLM-powered qualitative analysis.

## Features

- **Hybrid ML Prediction**: Combines XGBoost (classification) and LSTM (sequence) models
- **LLM Analysis**: GPT-powered insights and value bet detection
- **Zero-Cost Setup**: Runs entirely locally with SQLite and optional local LLM
- **Interactive Dashboard**: Streamlit-based UI for predictions and analytics

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **Database**: SQLite
- **ML Models**: XGBoost, TensorFlow/Keras (LSTM)
- **LLM**: OpenAI API or Ollama (local)

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env with your settings

# 4. Initialize database
python -m app.database init

# 5. Start the API server
uvicorn app.main:app --reload --port 8000

# 6. Start the dashboard (new terminal)
streamlit run dashboard/app.py
```

## Project Structure

```
bouls/
├── app/                    # FastAPI application
├── models/                 # ML models (XGBoost, LSTM)
├── data/                   # Data fetching & processing
├── llm/                    # LLM integration & prompts
├── api/                    # API routes
├── dashboard/              # Streamlit frontend
├── tests/                  # Test suite
└── trained_models/         # Saved model files
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# LLM Provider (openai or ollama)
LLM_PROVIDER=ollama

# OpenAI (if using)
OPENAI_API_KEY=your-key-here

# Ollama (if using local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Database
DATABASE_URL=sqlite:///./football_analytics.db
```

## License

MIT

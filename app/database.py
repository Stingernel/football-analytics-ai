"""
SQLite database connection and management.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "football_analytics.db"


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database with schema."""
    schema = """
    -- Teams table
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        short_name TEXT,
        league TEXT,
        country TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Matches table (historical data)
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fixture_id TEXT UNIQUE, -- From API
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        match_date DATE NOT NULL,
        season TEXT,
        league TEXT,
        home_goals INTEGER,
        away_goals INTEGER,
        home_xg REAL,
        away_xg REAL,
        home_shots INTEGER,
        away_shots INTEGER,
        home_shots_on_target INTEGER,
        away_shots_on_target INTEGER,
        home_possession REAL,
        away_possession REAL,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        result TEXT CHECK(result IN ('H', 'D', 'A')),
        -- Advanced analytics columns
        home_opponent_rating REAL,
        away_opponent_rating REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Predictions table
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER REFERENCES matches(id),
        home_team TEXT,
        away_team TEXT,
        league TEXT,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        home_form TEXT,
        away_form TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        home_win_prob REAL NOT NULL,
        draw_prob REAL NOT NULL,
        away_win_prob REAL NOT NULL,
        xgb_home_prob REAL,
        xgb_draw_prob REAL,
        xgb_away_prob REAL,
        lstm_home_prob REAL,
        lstm_draw_prob REAL,
        lstm_away_prob REAL,
        predicted_result TEXT CHECK(predicted_result IN ('H', 'D', 'A')),
        confidence REAL,
        -- Value bet for HOME
        vb_home_implied_prob REAL,
        vb_home_edge REAL,
        vb_home_recommended INTEGER DEFAULT 0,
        -- Value bet for DRAW
        vb_draw_implied_prob REAL,
        vb_draw_edge REAL,
        vb_draw_recommended INTEGER DEFAULT 0,
        -- Value bet for AWAY
        vb_away_implied_prob REAL,
        vb_away_edge REAL,
        vb_away_recommended INTEGER DEFAULT 0,
        -- Legacy fields for backward compatibility
        value_bet TEXT,
        value_bet_odds REAL,
        value_bet_edge REAL,
        llm_analysis TEXT,
        actual_result TEXT CHECK(actual_result IN ('H', 'D', 'A', NULL))
    );

    -- Model performance logs
    CREATE TABLE IF NOT EXISTS model_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_type TEXT NOT NULL,
        model_version TEXT,
        accuracy REAL,
        log_loss REAL,
        precision_home REAL,
        precision_draw REAL,
        precision_away REAL,
        recall_home REAL,
        recall_draw REAL,
        recall_away REAL,
        training_samples INTEGER,
        notes TEXT
    );

    -- Cached fixtures from The Odds API
    CREATE TABLE IF NOT EXISTS fixtures (
        id TEXT PRIMARY KEY,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        league TEXT NOT NULL,
        match_date DATETIME NOT NULL,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        source TEXT DEFAULT 'the-odds-api',
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Market Movement Tracking
    CREATE TABLE IF NOT EXISTS odds_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fixture_id TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        source TEXT,
        FOREIGN KEY(fixture_id) REFERENCES fixtures(id)
    );

    -- API usage tracking
    CREATE TABLE IF NOT EXISTS api_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_name TEXT NOT NULL,
        endpoint TEXT,
        calls_count INTEGER DEFAULT 1,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Cached predictions (auto-generated when fixtures fetched)
    CREATE TABLE IF NOT EXISTS cached_predictions (
        fixture_id TEXT PRIMARY KEY,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        league TEXT NOT NULL,
        match_date DATETIME,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        -- Form
        home_form TEXT,
        away_form TEXT,
        -- Probabilities
        home_prob REAL,
        draw_prob REAL,
        away_prob REAL,
        -- Prediction
        predicted_result TEXT,
        confidence REAL,
        models_agree INTEGER DEFAULT 0,
        -- Value bet
        best_bet TEXT,
        best_bet_edge REAL,
        best_bet_odds REAL,
        recommended INTEGER DEFAULT 0,
        -- Rating
        rating_stars INTEGER DEFAULT 1,
        rating_label TEXT,
        stake_amount REAL,
        expected_value REAL,
        -- Timestamps
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Team form cache (from Football-Data.org)
    CREATE TABLE IF NOT EXISTS team_form (
        team_name TEXT PRIMARY KEY,
        short_name TEXT,
        league TEXT,
        position INTEGER,
        form TEXT,  -- "WDLWW" format
        points INTEGER,
        played INTEGER,
        won INTEGER,
        draw INTEGER,
        lost INTEGER,
        goals_for INTEGER,
        goals_against INTEGER,
        goal_diff INTEGER,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
    CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
    CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at);
    CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(match_date);
    CREATE INDEX IF NOT EXISTS idx_fixtures_league ON fixtures(league);
    CREATE INDEX IF NOT EXISTS idx_odds_history_fixture ON odds_history(fixture_id);
    CREATE INDEX IF NOT EXISTS idx_team_form_league ON team_form(league);
    """
    
    with get_db() as conn:
        conn.executescript(schema)
        logger.info("Database initialized successfully")


def drop_all_tables():
    """Drop all tables (use with caution!)."""
    with get_db() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS predictions;
            DROP TABLE IF EXISTS team_form;
            DROP TABLE IF EXISTS matches;
            DROP TABLE IF EXISTS teams;
            DROP TABLE IF EXISTS model_logs;
        """)
        logger.info("All tables dropped")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "init":
            init_db()
            print("Database initialized!")
        elif command == "drop":
            confirm = input("Are you sure? This will delete all data! (yes/no): ")
            if confirm.lower() == "yes":
                drop_all_tables()
                print("All tables dropped!")
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m app.database [init|drop]")
    else:
        print("Usage: python -m app.database [init|drop]")

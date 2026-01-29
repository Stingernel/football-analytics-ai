"""
Live Validation System - Track Predictions vs Actual Results
Simple version for testing prediction accuracy.
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from app.database import get_db


def setup_validation():
    """Ensure validation columns exist."""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE cached_predictions ADD COLUMN validated INTEGER DEFAULT 0")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE cached_predictions ADD COLUMN actual_result TEXT")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE cached_predictions ADD COLUMN was_correct INTEGER")
        except:
            pass
        conn.commit()


def get_accuracy_stats():
    """Get current accuracy statistics."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM cached_predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Validated predictions
        cursor.execute("SELECT COUNT(*) FROM cached_predictions WHERE validated = 1")
        validated = cursor.fetchone()[0]
        
        # Correct predictions
        cursor.execute("SELECT COUNT(*) FROM cached_predictions WHERE was_correct = 1")
        correct = cursor.fetchone()[0]
        
        return {
            'total_predictions': total_predictions,
            'validated': validated,
            'correct': correct,
            'accuracy': (correct / validated * 100) if validated > 0 else 0
        }


def manual_validate(fixture_id: str, actual_result: str):
    """Manually validate a prediction with actual result."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get prediction
        cursor.execute("SELECT predicted_result FROM cached_predictions WHERE fixture_id = ?", (fixture_id,))
        row = cursor.fetchone()
        
        if not row:
            print(f"Fixture {fixture_id} not found")
            return
        
        predicted = row[0]
        is_correct = predicted == actual_result
        
        # Update
        cursor.execute("""
            UPDATE cached_predictions 
            SET validated = 1, actual_result = ?, was_correct = ?
            WHERE fixture_id = ?
        """, (actual_result, 1 if is_correct else 0, fixture_id))
        conn.commit()
        
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"{status} - Predicted: {predicted}, Actual: {actual_result}")


def show_pending():
    """Show predictions waiting for validation."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fixture_id, home_team, away_team, predicted_result, match_date
            FROM cached_predictions 
            WHERE validated = 0 OR validated IS NULL
            ORDER BY match_date DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        
        print("\n📋 Pending Validation:")
        print("-" * 60)
        for r in rows:
            print(f"  {r[0][:20]}... | {r[1]} vs {r[2]} | Pred: {r[3]} | {r[4]}")
        print(f"\nTotal pending: {len(rows)}")


def show_report():
    """Show accuracy report."""
    stats = get_accuracy_stats()
    
    print("\n" + "=" * 60)
    print("📊 ACCURACY REPORT")
    print("=" * 60)
    print(f"  Total Predictions: {stats['total_predictions']}")
    print(f"  Validated: {stats['validated']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Accuracy: {stats['accuracy']:.1f}%")
    print("=" * 60)


def main():
    print("=" * 60)
    print("LIVE VALIDATION SYSTEM")
    print("=" * 60)
    
    # Setup
    setup_validation()
    
    # Show pending
    show_pending()
    
    # Show report
    show_report()
    
    print("\n💡 To validate a prediction manually:")
    print("   python -c \"from validate_predictions import manual_validate; manual_validate('FIXTURE_ID', 'H')\"")
    print("   Where result is H (Home), D (Draw), or A (Away)")


if __name__ == "__main__":
    main()

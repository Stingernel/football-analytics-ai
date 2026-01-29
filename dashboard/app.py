"""
Football Analytics Dashboard - Streamlit Application
"""
import streamlit as st
import requests
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Football Analytics AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(16, 22, 54) 0%, rgb(10, 10, 20) 90%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(16, 22, 54, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Premium Glassmorphism Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #ffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Custom Badges */
    .value-bet {
        background: rgba(0, 184, 148, 0.2);
        color: #00b894;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(0, 184, 148, 0.3);
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.1);
    }
    
    .confidence-high {
        background: rgba(253, 203, 110, 0.2);
        color: #f1c40f;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(253, 203, 110, 0.3);
    }

    /* Probability Bars */
    .prob-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        height: 8px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(92.83deg, #ff7426 0, #f93a13 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(249, 58, 19, 0.4);
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: white;
        border: 1px solid transparent;
        transition: all 0.2s;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #fff !important;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# API base URL
API_URL = "http://localhost:8000/api"


def main():
    """Main dashboard application."""
    
    # Sidebar navigation
    st.sidebar.image("https://img.icons8.com/color/96/football2--v1.png", width=80)
    st.sidebar.title("⚽ Football Analytics AI")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "⚽ Predict Match", "📊 History", "📈 Analytics 🆕"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    Hybrid AI system combining:
    - **XGBoost** (60%)
    - **LSTM** (40%)
    - **LLM Analysis**
    """)
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "⚽ Predict Match":
        show_predict()
    elif page == "📊 History":
        show_history()
    elif page == "📈 Analytics 🆕":
        show_analytics()


def show_home():
    """Home page with overview."""
    st.title("🏆 Football Analytics AI System")
    st.markdown("### Hybrid Machine Learning Match Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 ML Models</h3>
            <p>XGBoost + LSTM ensemble for accurate probability estimation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 LLM Analysis</h3>
            <p>AI-powered insights and reasoning for every prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Value Bets</h3>
            <p>Automatic detection of betting value based on odds vs probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick prediction form
    st.subheader("⚡ Quick Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.text_input("Home Team", "Manchester City")
        home_odds = st.number_input("Home Odds", min_value=1.01, value=1.50, step=0.05)
        home_form = st.text_input("Home Form (last 5)", "WWWDW")
    
    with col2:
        away_team = st.text_input("Away Team", "Arsenal")
        away_odds = st.number_input("Away Odds", min_value=1.01, value=4.50, step=0.05)
        away_form = st.text_input("Away Form (last 5)", "WDWWL")
    
    draw_odds = st.number_input("Draw Odds", min_value=1.01, value=4.00, step=0.05)
    
    if st.button("🔮 Generate Prediction", type="primary"):
        with st.spinner("Analyzing match..."):
            try:
                response = requests.post(
                    f"{API_URL}/predictions/predict",
                    json={
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_form": home_form,
                        "away_form": away_form,
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "home_xg": 1.8,
                        "away_xg": 1.3,
                        "include_llm_analysis": True
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    display_prediction(result)
                else:
                    st.error(f"API Error: {response.status_code}")
                    # Use demo prediction
                    display_demo_prediction(home_team, away_team, home_odds, draw_odds, away_odds)
                    
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ API not available. Showing demo prediction...")
                display_demo_prediction(home_team, away_team, home_odds, draw_odds, away_odds)


def display_demo_prediction(home_team, away_team, home_odds, draw_odds, away_odds):
    """Display demo prediction when API is unavailable."""
    # Calculate from odds
    total = (1/home_odds) + (1/draw_odds) + (1/away_odds)
    home_prob = (1/home_odds) / total
    draw_prob = (1/draw_odds) / total
    away_prob = (1/away_odds) / total
    
    result = {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': home_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_prob,
        'predicted_result': 'H' if home_prob > draw_prob and home_prob > away_prob else ('A' if away_prob > draw_prob else 'D'),
        'predicted_result_full': 'Home Win' if home_prob > draw_prob and home_prob > away_prob else ('Away Win' if away_prob > draw_prob else 'Draw'),
        'confidence': max(home_prob, draw_prob, away_prob),
        'models_agree': True,
        'value_bets': [],
        'analysis': f"Based on the provided odds, {home_team} appears to be the favorite. The model suggests backing the predicted outcome with appropriate stake sizing."
    }
    
    display_prediction(result)


def display_prediction(result):
    """Display prediction result with visualizations."""
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    # Team matchup header
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"<div class='team-name' style='text-align: right;'>{result['home_team']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='vs-text' style='text-align: center; font-size: 1.5rem;'>VS</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='team-name'>{result['away_team']}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability bars
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_pct = result['home_win_prob'] * 100
        st.metric("Home Win", f"{home_pct:.1f}%")
        st.progress(result['home_win_prob'])
    
    with col2:
        draw_pct = result['draw_prob'] * 100
        st.metric("Draw", f"{draw_pct:.1f}%")
        st.progress(result['draw_prob'])
    
    with col3:
        away_pct = result['away_win_prob'] * 100
        st.metric("Away Win", f"{away_pct:.1f}%")
        st.progress(result['away_win_prob'])
    
    # Predicted result
    st.markdown("---")
    
    result_emoji = {"Home Win": "🏠", "Draw": "🤝", "Away Win": "✈️"}.get(
        result.get('predicted_result_full', 'Home Win'), "⚽"
    )
    
    st.success(f"""
    **{result_emoji} Predicted Result: {result.get('predicted_result_full', 'N/A')}**
    
    Confidence: **{result['confidence']*100:.1f}%** | Models Agree: **{'✅ Yes' if result.get('models_agree', False) else '❌ No'}**
    """)
    
    # Professional Value Bet Analysis
    if result.get('value_bets'):
        st.subheader("💰 Value Bet Analysis (Professional)")
        
        for vb in result['value_bets']:
            # Get rating info
            rating = vb.get('rating', {})
            stars = rating.get('stars_display', '⭐')
            label = rating.get('label', 'N/A')
            color = rating.get('color', '#888')
            
            edge_pct = vb.get('edge_pct', vb['edge'] * 100)
            stake = vb.get('stake', 0)
            stake_pct = vb.get('stake_pct', 0)
            ev = vb.get('expected_value', 0)
            
            # Determine border color based on edge
            if vb['recommended']:
                border = "#00b894"
                bg = "rgba(0, 184, 148, 0.1)"
            elif vb['edge'] >= 0:
                border = "#fdcb6e"
                bg = "rgba(253, 203, 110, 0.05)"
            else:
                border = "#d63031"
                bg = "rgba(214, 48, 49, 0.05)"
            
            rec_badge = "✅ RECOMMENDED" if vb['recommended'] else ""
            stake_html = f'<span style="float: right; color: #00b894;"><b>💰 Stake: ${stake:.0f} ({stake_pct:.1f}%)</b></span>' if vb['recommended'] else ''
            
            st.markdown(f"""
            <div style="background: {bg}; border-left: 4px solid {border}; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: {color};">{vb['bet_type']} @ {vb['odds']:.2f} {stars}</h4>
                        <p style="margin: 5px 0; font-size: 0.9rem;">
                            <b>Rating:</b> {label} | 
                            <b>Edge:</b> <span style="color: {border};">{edge_pct:+.1f}%</span> | 
                            <b>EV:</b> {ev:+.1f}%
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.2rem; color: {color};">{rec_badge}</span>
                    </div>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <span style="font-size: 0.85rem;">
                        Model: <b>{vb['model_prob']*100:.1f}%</b> vs 
                        Market: <b>{vb['implied_prob']*100:.1f}%</b>
                    </span>
                    {stake_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Legend
        st.markdown("""
        <p style="font-size: 0.8rem; color: #888; margin-top: 15px;">
        📊 <b>Rating:</b> ⭐⭐⭐⭐⭐ STRONG → ⭐ SKIP | 
        <b>Min Edge:</b> 7% | <b>Min Confidence:</b> 55% | 
        <b>Stake:</b> Kelly Criterion (¼ Kelly)
        </p>
        """, unsafe_allow_html=True)
    
    # Analysis
    if result.get('analysis'):
        st.subheader("🧠 AI Analysis")
        st.markdown(f"""
        <div class="analysis-box">
        {result['analysis']}
        </div>
        """, unsafe_allow_html=True)


def show_predict():
    """Full prediction page with Manual and Auto-Predict tabs."""
    st.title("⚽ Match Prediction")
    
    # Create tabs for Manual and Auto-Predict
    tab1, tab2 = st.tabs(["📝 Manual Predict", "🔄 Auto-Predict"])
    
    # ==================== MANUAL PREDICT TAB ====================
    with tab1:
        st.markdown("Enter match details for a comprehensive prediction analysis.")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Home Team")
                home_team = st.text_input("Team Name", "Liverpool", key="home")
                home_form = st.text_input("Recent Form (newest → oldest)", "WWDWW", key="hf")
                home_xg = st.number_input("Avg xG (last 5)", min_value=0.0, value=2.1, key="hxg")
                home_odds = st.number_input("Win Odds", min_value=1.01, value=1.65, key="ho")
            
            with col2:
                st.subheader("Away Team")
                away_team = st.text_input("Team Name", "Chelsea", key="away")
                away_form = st.text_input("Recent Form (newest → oldest)", "WLDWW", key="af")
                away_xg = st.number_input("Avg xG (last 5)", min_value=0.0, value=1.5, key="axg")
                away_odds = st.number_input("Win Odds", min_value=1.01, value=4.20, key="ao")
            
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.80)
            league = st.selectbox("League", ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
            
            include_llm = st.checkbox("Include AI Analysis", value=True)
            
            submitted = st.form_submit_button("🔮 Analyze Match", type="primary")
            
            if submitted:
                with st.spinner("Running hybrid ML analysis..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predictions/predict",
                            json={
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_form": home_form,
                                "away_form": away_form,
                                "home_odds": home_odds,
                                "draw_odds": draw_odds,
                                "away_odds": away_odds,
                                "home_xg": home_xg,
                                "away_xg": away_xg,
                                "league": league,
                                "include_llm_analysis": include_llm
                            },
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            display_prediction(result)
                            
                            # Show model breakdown
                            st.markdown("---")
                            st.subheader("📈 Model Breakdown")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**XGBoost (60% weight)**")
                                xgb = result.get('xgb_probs', {})
                                st.write(f"Home: {xgb.get('home_win', 0)*100:.1f}%")
                                st.write(f"Draw: {xgb.get('draw', 0)*100:.1f}%")
                                st.write(f"Away: {xgb.get('away_win', 0)*100:.1f}%")
                            
                            with col2:
                                st.markdown("**LSTM (40% weight)**")
                                lstm = result.get('lstm_probs', {})
                                st.write(f"Home: {lstm.get('home_win', 0)*100:.1f}%")
                                st.write(f"Draw: {lstm.get('draw', 0)*100:.1f}%")
                                st.write(f"Away: {lstm.get('away_win', 0)*100:.1f}%")
                        else:
                            st.error("API Error")
                            display_demo_prediction(home_team, away_team, home_odds, draw_odds, away_odds)
                            
                    except requests.exceptions.ConnectionError:
                        st.warning("API unavailable. Demo mode active.")
                        display_demo_prediction(home_team, away_team, home_odds, draw_odds, away_odds)
    
    # ==================== AUTO-PREDICT TAB (Cached Fixtures) ====================
    with tab2:
        st.markdown("### ⚽ Upcoming Matches")
        st.markdown("Matches updated automatically. Select and predict!")
        
        # Date tabs
        date_tab1, date_tab2, date_tab3 = st.tabs(["📅 Today", "📆 Tomorrow", "📊 This Week"])
        
        # Helper function to display matches
        def display_cached_matches(matches, tab_key):
            if not matches:
                st.info("No matches scheduled for this period.")
                return
            
            # Show last update time
            try:
                response = requests.get(f"{API_URL}/fixtures/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    last_update = status.get('last_update', 'Unknown')
                    st.caption(f"🔄 Last updated: {last_update} | {len(matches)} matches")
            except:
                st.caption(f"📊 {len(matches)} matches")
            
            st.markdown("---")
            
            # Initialize selection state
            if f'selected_{tab_key}' not in st.session_state:
                st.session_state[f'selected_{tab_key}'] = []
            
            # Group by league
            leagues = {}
            for m in matches:
                league = m.get('league', 'Other')
                if league not in leagues:
                    leagues[league] = []
                leagues[league].append(m)
            
            # Display by league
            for league, league_matches in leagues.items():
                with st.expander(f"🏆 {league} ({len(league_matches)} matches)", expanded=True):
                    for match in league_matches:
                        match_id = match['id']
                        home = match['home_team']
                        away = match['away_team']
                        
                        col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1.5, 1.5, 1.5])
                        
                        with col1:
                            is_selected = match_id in st.session_state[f'selected_{tab_key}']
                            if st.checkbox("", value=is_selected, key=f"sel_{tab_key}_{match_id}"):
                                if match_id not in st.session_state[f'selected_{tab_key}']:
                                    st.session_state[f'selected_{tab_key}'].append(match_id)
                            else:
                                if match_id in st.session_state[f'selected_{tab_key}']:
                                    st.session_state[f'selected_{tab_key}'].remove(match_id)
                        
                        with col2:
                            # Check for European competition badge
                            euro_badge = ""
                            try:
                                from models.congestion import get_congestion_badge
                                home_badge = get_congestion_badge(home, league)
                                away_badge = get_congestion_badge(away, league)
                                if home_badge:
                                    euro_badge += f" {home_badge['emoji']}"
                                if away_badge:
                                    euro_badge += f" {away_badge['emoji']}"
                            except:
                                pass
                            
                            st.markdown(f"**{home}** vs **{away}**{euro_badge}")
                        
                        with col3:
                            st.markdown(f"<span style='color:#00b894'>H: {match['home_odds']:.2f}</span>", unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"<span style='color:#fdcb6e'>D: {match['draw_odds']:.2f}</span>", unsafe_allow_html=True)
                        
                        with col5:
                            st.markdown(f"<span style='color:#e94560'>A: {match['away_odds']:.2f}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Predict button
            selected_count = len(st.session_state.get(f'selected_{tab_key}', []))
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.markdown(f"**{selected_count} selected**")
            with col2:
                include_llm = st.checkbox("AI Analysis", value=False, key=f"llm_{tab_key}")
            with col3:
                if st.button(f"🔮 Predict Selected", type="primary", key=f"predict_{tab_key}", disabled=selected_count == 0):
                    with st.spinner(f"Predicting {selected_count} matches..."):
                        try:
                            # Get full match data for selected IDs
                            selected_ids = st.session_state[f'selected_{tab_key}']
                            selected_matches = [m for m in matches if m['id'] in selected_ids]
                            
                            response = requests.post(
                                f"{API_URL}/predictions/auto",
                                json={
                                    "match_ids": selected_ids,
                                    "include_llm": include_llm
                                },
                                timeout=120
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ {result['success']} predictions completed!")
                                
                                # Display results
                                for pred in result.get('predictions', []):
                                    result_map = {'H': '🏠 Home', 'D': '🤝 Draw', 'A': '✈️ Away'}
                                    result_text = result_map.get(pred['predicted_result'], pred['predicted_result'])
                                    
                                    rating = pred.get('rating', {})
                                    stars = rating.get('stars_display', '⭐')
                                    
                                    st.markdown(f"""
                                    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0;">
                                        <b>{pred['home_team']} vs {pred['away_team']}</b><br>
                                        {result_text} ({pred['confidence']*100:.0f}%) {stars}
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Clear selection
                                st.session_state[f'selected_{tab_key}'] = []
                        except Exception as e:
                            st.error(f"Error: {e}")
        
        # Fetch fixtures with predictions for each tab
        with date_tab1:  # Today with Predictions
            try:
                response = requests.get(f"{API_URL}/predictions/fixtures/predictions/today", timeout=15)
                if response.status_code == 200:
                    matches = response.json()
                    if not matches:
                        st.info("📅 No matches scheduled for today or predictions not yet generated.")
                        # Offer manual refresh
                        if st.button("🔄 Refresh Predictions", key="refresh_today"):
                            with st.spinner("Refreshing..."):
                                requests.post(f"{API_URL}/predictions/fixtures/refresh", timeout=60)
                                st.rerun()
                    else:
                        st.caption(f"📊 {len(matches)} matches with predictions")
                        st.markdown("---")
                        
                        # Fetch Ratings (for display)
                        try:
                            ratings_response = requests.get(f"{API_URL}/analytics/ratings", timeout=5)
                            team_ratings = {r['team']: r['rating'] for r in ratings_response.json()} if ratings_response.status_code == 200 else {}
                        except:
                            team_ratings = {}
                        
                        # Group by league
                        leagues = {}
                        for m in matches:
                            league = m.get('league', 'Other')
                            if league not in leagues:
                                leagues[league] = []
                            leagues[league].append(m)
                            
                        # Fetch Market Movments (Smart Money)
                        try:
                            mm_response = requests.get(f"{API_URL}/analytics/market-movement?threshold=0.05", timeout=5)
                            movements = mm_response.json() if mm_response.status_code == 200 else []
                            # Map alerts by fixture_id
                            alerts = {m['fixture_id']: m for m in movements}
                        except:
                            alerts = {}
                        
                        # Display by league with full predictions
                        for league, league_matches in leagues.items():
                            with st.expander(f"🏆 {league} ({len(league_matches)} matches)", expanded=True):
                                for match in league_matches:
                                    # Result emoji
                    
                                    # Get Ratings
                                    home_rating = team_ratings.get(match['home_team'], 1000)
                                    away_rating = team_ratings.get(match['away_team'], 1000)
                                    
                                    # Check for Smart Money Alert
                                    alert_badge = ""
                                    market_status = "<span style='color: #636e72; font-size: 0.8em;'>📉 Market: Stable</span>"
                                    
                                    fix_id = match.get('id')
                                    if fix_id in alerts:
                                        alert = alerts[fix_id]
                                        drop = alert['drop_percent']
                                        side = "HOME" if alert['alert_type'] == 'HOME_DROP' else "AWAY"
                                        alert_badge = f"""<span style='background-color: #d63031; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 8px;'>📉 SHARP MONEY: {side} -{drop}%</span>"""
                                        market_status = "" # Hide stable status if alert is present

                                    result_map = {'H': '🏠', 'D': '🤝', 'A': '✈️'}
                                    result_emoji = result_map.get(match.get('predicted_result', ''), '⚽')
                                    
                                    # Rating stars
                                    stars = "⭐" * match.get('rating_stars', 1)
                                    rating_label = match.get('rating_label', 'N/A')
                                    
                                    # Value bet info
                                    is_value = match.get('recommended', False)
                                    best_bet = match.get('best_bet', '')
                                    edge = match.get('best_bet_edge', 0) * 100 if match.get('best_bet_edge') else 0
                                    stake = match.get('stake', 0)
                                    ev = match.get('expected_value', 0)
                                    
                                    # Colors
                                    if is_value:
                                        border_color = "#00b894"
                                        bg = "rgba(0, 184, 148, 0.1)"
                                    elif match.get('confidence', 0) > 0.5:
                                        border_color = "#fdcb6e"
                                        bg = "rgba(253, 203, 110, 0.05)"
                                    else:
                                        border_color = "#636e72"
                                        bg = "rgba(99, 110, 114, 0.05)"
                                    
                                    # Confidence
                                    conf = match.get('confidence', 0) * 100
                                    
                                    # Value bet HTML (separate to avoid nested f-string issues)
                                    value_bet_html = ""
                                    if is_value and best_bet:
                                        value_bet_html = f' | <span style="color: #00b894;">✅ <b>VALUE BET</b>: {best_bet} @ {round(match.get("best_bet_odds", 0), 2)} (Edge: +{edge:.1f}%, Stake: ${stake:.0f})</span>'
                                    
                                    # Match card (single-line for Railway compatibility)
                                    card_html = f'<div class="metric-card" style="border-left: 4px solid {border_color}; margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; align-items: flex-start;"><div style="flex: 2;"><div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;"><h3 style="margin: 0; font-size: 1.1rem; color: white;">{match["home_team"]} <span style="color:#aaa;font-weight:normal;font-size:0.8em">({home_rating})</span></h3><span style="color: #888; font-size: 0.9rem;">vs</span><h3 style="margin: 0; font-size: 1.1rem; color: white;">{match["away_team"]} <span style="color:#aaa;font-weight:normal;font-size:0.8em">({away_rating})</span></h3>{alert_badge}</div><div style="margin-top: 8px; display:flex; gap:15px; align-items:center; flex-wrap: wrap;"><span style="background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:6px; font-size: 0.8rem; color:#ddd;">🏠 {match["home_odds"]:.2f}</span><span style="background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:6px; font-size: 0.8rem; color:#ddd;">🤝 {match["draw_odds"]:.2f}</span><span style="background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:6px; font-size: 0.8rem; color:#ddd;">✈️ {match["away_odds"]:.2f}</span>{market_status}</div></div><div style="text-align: right; flex: 1;"><div style="font-size: 1.5rem; margin-bottom: 4px;">{result_emoji}</div><div style="font-weight: bold; font-size: 1.1rem; color: #00C9FF;">{conf:.0f}%</div><div style="color: #f1c40f; font-size: 0.8rem; margin-top: 2px;">{stars}</div></div></div><div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;"><span style="font-size: 0.9rem; color: #ccc;"><b>Prediction:</b> {result_emoji} {match.get("predicted_result", "N/A")} | <b>Rating:</b> {rating_label}</span>{value_bet_html}</div></div>'
                                    st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Summary stats
                        value_bets = [m for m in matches if m.get('recommended')]
                        st.markdown("---")
                        st.markdown(f"""
                        **📊 Summary**: {len(matches)} predictions | {len(value_bets)} value bets found
                        """)
                else:
                    st.warning("Could not load predictions. Try refreshing.")
                    if st.button("🔄 Refresh", key="refresh_err"):
                        requests.post(f"{API_URL}/predictions/fixtures/refresh", timeout=60)
                        st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("API unavailable. Please make sure the backend server is running.")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Tomorrow Tab
        with date_tab2:
            try:
                response = requests.get(f"{API_URL}/predictions/fixtures/predictions/tomorrow", timeout=15)
                if response.status_code == 200:
                    matches = response.json()
                    
                    if not matches:
                        st.info("📆 No matches scheduled for tomorrow.")
                    else:
                        st.caption(f"📊 {len(matches)} matches found")
                        st.markdown("---")
                        
                        # Use same rendering logic as Today tab (simplified repeat)
                        # Fetch Ratings & Market Movements if needed (omitted for speed in this tab or reuse)
                        try:
                            ratings_response = requests.get(f"{API_URL}/analytics/ratings", timeout=5)
                            team_ratings = {r['team']: r['rating'] for r in ratings_response.json()} if ratings_response.status_code == 200 else {}
                        except:
                            team_ratings = {}
                            
                        # Group by league
                        leagues = {}
                        for m in matches:
                            league = m.get('league', 'Other')
                            if league not in leagues: leagues[league] = []
                            leagues[league].append(m)
                            
                        for league, league_matches in leagues.items():
                            with st.expander(f"🏆 {league} ({len(league_matches)} matches)", expanded=True):
                                for match in league_matches:
                                    home_rating = team_ratings.get(match['home_team'], 1000)
                                    away_rating = team_ratings.get(match['away_team'], 1000)
                                    
                                    # Initialize alert variables (not used in Tomorrow tab)
                                    alert_badge = ""
                                    market_status = ""
                                    
                                    result_map = {'H': '🏠', 'D': '🤝', 'A': '✈️'}
                                    result_emoji = result_map.get(match.get('predicted_result', ''), '⚽')
                                    stars = "⭐" * match.get('rating_stars', 1)
                                    rating_label = match.get('rating_label', 'N/A')
                                    
                                    # Value bet
                                    is_value = match.get('recommended', False)
                                    best_bet = match.get('best_bet', '')
                                    edge = match.get('best_bet_edge', 0) * 100 if match.get('best_bet_edge') else 0
                                    
                                    if is_value:
                                        border_color = "#00b894"
                                        bg = "rgba(0, 184, 148, 0.1)"
                                    elif match.get('confidence', 0) > 0.5:
                                        border_color = "#fdcb6e"
                                        bg = "rgba(253, 203, 110, 0.05)"
                                    else:
                                        border_color = "#636e72"
                                        bg = "rgba(99, 110, 114, 0.05)"
                                        
                                    conf = match.get('confidence', 0) * 100
                                    value_bet_html = f' | <span style="color: #00b894;">✅ <b>VALUE</b>: {best_bet} (+{edge:.1f}%)</span>' if is_value else ""
                                    
                                    # Match card (single-line for Railway compatibility)
                                    card_html = f'<div class="metric-card" style="border-left: 4px solid {border_color}; margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; align-items: flex-start;"><div style="flex: 2;"><div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;"><h3 style="margin: 0; font-size: 1.1rem; color: white;">{match["home_team"]} <span style="color:#aaa;font-weight:normal;font-size:0.8em">({home_rating})</span></h3><span style="color: #888; font-size: 0.9rem;">vs</span><h3 style="margin: 0; font-size: 1.1rem; color: white;">{match["away_team"]} <span style="color:#aaa;font-weight:normal;font-size:0.8em">({away_rating})</span></h3>{alert_badge}</div><div style="margin-top: 8px; display:flex; gap:15px; align-items:center; flex-wrap: wrap;"><span style="background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:6px; font-size: 0.8rem; color:#ddd;">🏠 {match["home_odds"]:.2f}</span><span style="background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:6px; font-size: 0.8rem; color:#ddd;">🤝 {match["draw_odds"]:.2f}</span><span style="background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:6px; font-size: 0.8rem; color:#ddd;">✈️ {match["away_odds"]:.2f}</span>{market_status}</div></div><div style="text-align: right; flex: 1;"><div style="font-size: 1.5rem; margin-bottom: 4px;">{result_emoji}</div><div style="font-weight: bold; font-size: 1.1rem; color: #00C9FF;">{conf:.0f}%</div><div style="color: #f1c40f; font-size: 0.8rem; margin-top: 2px;">{stars}</div></div></div><div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;"><span style="font-size: 0.9rem; color: #ccc;"><b>Prediction:</b> {result_emoji} {match.get("predicted_result", "N/A")} | <b>Rating:</b> {rating_label}</span>{value_bet_html}</div></div>'
                                    st.markdown(card_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading tomorrow's fixtures: {e}")

        # This Week Tab
        with date_tab3:
            try:
                response = requests.get(f"{API_URL}/predictions/fixtures/predictions/week", timeout=15)
                if response.status_code == 200:
                    matches = response.json()
                    
                    if not matches:
                        st.info("📊 No matches scheduled for the next 7 days.")
                    else:
                        st.caption(f"📊 {len(matches)} matches upcoming this week")
                        st.markdown("---")
                        
                        # Group by Date then League
                        # Sort by commence_time
                        matches.sort(key=lambda x: x.get('commence_time', ''))
                        
                        current_date = ""
                        for match in matches:
                            m_date = match.get('commence_time', '').split('T')[0]
                            if m_date != current_date:
                                st.markdown(f"#### 📅 {m_date}")
                                current_date = m_date
                            
                            # Simple card for week view
                            result_map = {'H': '🏠', 'D': '🤝', 'A': '✈️'}
                            emoji = result_map.get(match.get('predicted_result', ''), '⚽')
                            rec = "💰 VALUE" if match.get('recommended') else ""
                            
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.markdown(f"**{match['home_team']}** vs **{match['away_team']}** ({match['league']})")
                                with col2:
                                    st.markdown(f"{emoji} {match.get('predicted_result', '')} ({match.get('confidence',0)*100:.0f}%)")
                                with col3:
                                    if rec:
                                        st.markdown(f"<span style='color:#00b894; font-weight:bold;'>{rec}</span>", unsafe_allow_html=True)
                                st.divider()
                                
            except Exception as e:
                st.error(f"Error loading week fixtures: {e}")
        
        # Display upcoming matches if available
        if 'upcoming_matches' in st.session_state and st.session_state['upcoming_matches']:
            matches = st.session_state['upcoming_matches']
            
            st.markdown("---")
            st.markdown("### 📋 Upcoming Matches")
            
            # Demo mode indicator
            if matches and matches[0].get('source') == 'demo':
                st.info("⚠️ **Demo Mode**: Showing simulated fixtures. Add ODDS_API_KEY to .env for real data.")
            
            # Select all / Deselect all
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("✅ Select All"):
                    st.session_state['selected_matches'] = [m['id'] for m in matches]
                    st.rerun()
            with col2:
                if st.button("❌ Deselect All"):
                    st.session_state['selected_matches'] = []
                    st.rerun()
            
            # Initialize selected_matches if not exists
            if 'selected_matches' not in st.session_state:
                st.session_state['selected_matches'] = []
            
            # Display matches as selectable cards
            for i, match in enumerate(matches):
                col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 2, 1.5, 1.5, 1.5])
                
                with col1:
                    is_selected = match['id'] in st.session_state['selected_matches']
                    if st.checkbox("", value=is_selected, key=f"sel_{match['id']}"):
                        if match['id'] not in st.session_state['selected_matches']:
                            st.session_state['selected_matches'].append(match['id'])
                    else:
                        if match['id'] in st.session_state['selected_matches']:
                            st.session_state['selected_matches'].remove(match['id'])
                
                with col2:
                    st.markdown(f"**🏠 {match['home_team']}**")
                
                with col3:
                    st.markdown(f"**✈️ {match['away_team']}**")
                
                with col4:
                    st.markdown(f"<span style='color: #00b894;'>{match['home_odds']:.2f}</span>", unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"<span style='color: #fdcb6e;'>{match['draw_odds']:.2f}</span>", unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"<span style='color: #e94560;'>{match['away_odds']:.2f}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Predict selected matches
            selected_count = len(st.session_state.get('selected_matches', []))
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{selected_count} matches selected**")
            
            with col2:
                predict_btn = st.button(
                    f"🔮 Predict Selected ({selected_count})",
                    type="primary",
                    disabled=selected_count == 0
                )
            
            if predict_btn and selected_count > 0:
                with st.spinner(f"Generating predictions for {selected_count} matches..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predictions/auto",
                            json={
                                "match_ids": st.session_state['selected_matches'],
                                "include_llm": include_llm_auto
                            },
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success(f"✅ {result['success']} predictions completed!")
                            
                            if result['failed'] > 0:
                                st.warning(f"⚠️ {result['failed']} failed")
                            
                            # Show results
                            if result['predictions']:
                                st.markdown("### 📊 Prediction Results")
                                
                                for pred in result['predictions']:
                                    result_map = {'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'}
                                    result_emoji = result_map.get(pred['predicted_result'], pred['predicted_result'])
                                    
                                    rec = pred.get('recommended_bet')
                                    rec_badge = f"💰 {rec}" if rec else ""
                                    
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <strong>{pred['home_team']} vs {pred['away_team']}</strong><br>
                                        Prediction: <b>{result_emoji}</b> ({pred['confidence']*100:.0f}% confidence) {rec_badge}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Clear selections after successful prediction
                            st.session_state['selected_matches'] = []
                            st.session_state['upcoming_matches'] = []
                            
                        else:
                            st.error(f"Prediction failed: {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("API unavailable. Please start the backend server.")


def show_history():
    """Enhanced prediction history page with full transparency."""
    st.title("📊 Prediction History")
    st.markdown("View all predictions with complete transparency and model breakdown.")
    
    # Fetch performance stats for summary cards
    stats = None
    try:
        stats_response = requests.get(f"{API_URL}/history/stats/performance", timeout=10)
        if stats_response.status_code == 200:
            stats = stats_response.json()
    except:
        pass
    
    # Summary Cards
    if stats:
        st.markdown("### 📈 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #00cec9;">📋 {stats['total_predictions']}</h3>
                <p style="margin:0; color: #888;">Total Predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            acc_pct = stats['accuracy'] * 100
            acc_color = "#00b894" if acc_pct >= 50 else "#e17055"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: {acc_color};">🎯 {acc_pct:.1f}%</h3>
                <p style="margin:0; color: #888;">Accuracy ({stats['correct_predictions']}/{stats['predictions_with_result']})</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            roi_pct = stats['roi'] * 100
            roi_color = "#00b894" if roi_pct >= 0 else "#d63031"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: {roi_color};">💰 {roi_pct:+.1f}%</h3>
                <p style="margin:0; color: #888;">ROI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_conf = stats.get('avg_confidence', 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0; color: #fdcb6e;">⚡ {avg_conf:.1f}%</h3>
                <p style="margin:0; color: #888;">Avg Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Filters Section
    st.markdown("### 🔍 Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        team_search = st.text_input("🔎 Search Team", "", placeholder="e.g. Manchester")
    
    with filter_col2:
        result_filter = st.selectbox("📊 Predicted Result", ["All", "Home (H)", "Draw (D)", "Away (A)"])
    
    with filter_col3:
        verified_only = st.checkbox("✅ Verified Only", value=False)
    
    with filter_col4:
        limit = st.selectbox("📄 Show", [20, 50, 100], index=0)
    
    # Build query params
    params = {"limit": limit}
    if team_search:
        params["team_name"] = team_search
    if result_filter != "All":
        params["predicted_result"] = result_filter.split("(")[1].replace(")", "")
    if verified_only:
        params["only_verified"] = "true"
    
    st.markdown("---")
    
    # Fetch history data
    try:
        response = requests.get(f"{API_URL}/history/", params=params, timeout=10)
        
        if response.status_code == 200:
            history = response.json()
            
            if history:
                st.markdown(f"### 📋 Predictions ({len(history)} results)")
                
                for item in history:
                    # Build match title
                    home = item.get('home_team') or 'Team A'
                    away = item.get('away_team') or 'Team B'
                    date_str = item['created_at'][:10] if item['created_at'] else 'N/A'
                    league = item.get('league') or ''
                    
                    # Status indicator
                    if item.get('actual_result'):
                        if item.get('correct'):
                            status_icon = "✅"
                            status_text = "Correct"
                        else:
                            status_icon = "❌"
                            status_text = "Wrong"
                    else:
                        status_icon = "⏳"
                        status_text = "Pending"
                    
                    # Result mapping
                    result_map = {'H': 'Home', 'D': 'Draw', 'A': 'Away'}
                    pred_full = result_map.get(item['predicted_result'], item['predicted_result'])
                    
                    with st.expander(f"{status_icon} {home} vs {away} — Pred: **{pred_full}** ({item['confidence']*100:.0f}%) | {date_str}"):
                        # Match Info Row
                        info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
                        
                        with info_col1:
                            st.markdown(f"**🏟️ Match:** {home} vs {away}")
                            if league:
                                st.markdown(f"**🏆 League:** {league}")
                            st.markdown(f"**📅 Date:** {date_str}")
                        
                        with info_col2:
                            if item.get('home_odds'):
                                st.markdown("**📊 Odds:**")
                                st.markdown(f"H: {item['home_odds']:.2f} | D: {item['draw_odds']:.2f} | A: {item['away_odds']:.2f}")
                        
                        with info_col3:
                            st.markdown(f"**🎯 Status:** {status_icon} {status_text}")
                            if item.get('actual_result'):
                                actual_full = result_map.get(item['actual_result'], item['actual_result'])
                                st.markdown(f"**Actual:** {actual_full}")
                        
                        st.markdown("---")
                        
                        # Probabilities Section
                        st.markdown("**📈 Model Probabilities**")
                        prob_col1, prob_col2, prob_col3 = st.columns(3)
                        
                        with prob_col1:
                            home_pct = item['home_win_prob'] * 100
                            is_pred = item['predicted_result'] == 'H'
                            st.metric("🏠 Home Win", f"{home_pct:.1f}%", delta="PREDICTED" if is_pred else None)
                            st.progress(item['home_win_prob'])
                        
                        with prob_col2:
                            draw_pct = item['draw_prob'] * 100
                            is_pred = item['predicted_result'] == 'D'
                            st.metric("🤝 Draw", f"{draw_pct:.1f}%", delta="PREDICTED" if is_pred else None)
                            st.progress(item['draw_prob'])
                        
                        with prob_col3:
                            away_pct = item['away_win_prob'] * 100
                            is_pred = item['predicted_result'] == 'A'
                            st.metric("✈️ Away Win", f"{away_pct:.1f}%", delta="PREDICTED" if is_pred else None)
                            st.progress(item['away_win_prob'])
                        
                        # Value Bet Info
                        if item.get('value_bet'):
                            edge_pct = (item.get('value_bet_edge') or 0) * 100
                            st.success(f"💰 **Value Bet Detected:** {item['value_bet']} @ {item.get('value_bet_odds', 0):.2f} (Edge: {edge_pct:+.1f}%)")
                        
                        st.markdown("---")
                        
                        # Fetch full detail for model breakdown
                        try:
                            detail_resp = requests.get(f"{API_URL}/history/{item['id']}", timeout=5)
                            if detail_resp.status_code == 200:
                                detail = detail_resp.json()
                                
                                # Model Breakdown
                                st.markdown("**🤖 Model Breakdown (Transparency)**")
                                model_col1, model_col2 = st.columns(2)
                                
                                with model_col1:
                                    st.markdown("""
                                    <div class="metric-card">
                                        <h4 style="color: #00cec9;">XGBoost (60% weight)</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    xgb_h = (detail.get('xgb_home_prob') or 0) * 100
                                    xgb_d = (detail.get('xgb_draw_prob') or 0) * 100
                                    xgb_a = (detail.get('xgb_away_prob') or 0) * 100
                                    st.markdown(f"- Home: **{xgb_h:.1f}%**")
                                    st.markdown(f"- Draw: **{xgb_d:.1f}%**")
                                    st.markdown(f"- Away: **{xgb_a:.1f}%**")
                                
                                with model_col2:
                                    st.markdown("""
                                    <div class="metric-card">
                                        <h4 style="color: #e94560;">LSTM (40% weight)</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    lstm_h = (detail.get('lstm_home_prob') or 0) * 100
                                    lstm_d = (detail.get('lstm_draw_prob') or 0) * 100
                                    lstm_a = (detail.get('lstm_away_prob') or 0) * 100
                                    st.markdown(f"- Home: **{lstm_h:.1f}%**")
                                    st.markdown(f"- Draw: **{lstm_d:.1f}%**")
                                    st.markdown(f"- Away: **{lstm_a:.1f}%**")
                                
                                # Complete Value Bet Analysis Section
                                st.markdown("---")
                                st.markdown("**💰 Value Bet Analysis (Full Transparency)**")
                                
                                # Create value bet table
                                vb_col1, vb_col2, vb_col3 = st.columns(3)
                                
                                with vb_col1:
                                    h_odds = item.get('home_odds') or 0
                                    h_implied = (detail.get('vb_home_implied_prob') or 0) * 100
                                    h_model = item['home_win_prob'] * 100
                                    h_edge = (detail.get('vb_home_edge') or 0) * 100
                                    h_rec = detail.get('vb_home_recommended')
                                    
                                    rec_badge = "✅ RECOMMENDED" if h_rec else ""
                                    edge_color = "#00b894" if h_edge >= 5 else ("#fdcb6e" if h_edge >= 0 else "#d63031")
                                    
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left: 4px solid {edge_color};">
                                        <h4 style="margin:0;">🏠 HOME</h4>
                                        <p style="margin:5px 0;">Odds: <b>{h_odds:.2f}</b></p>
                                        <p style="margin:5px 0;">Implied: <b>{h_implied:.1f}%</b></p>
                                        <p style="margin:5px 0;">Model: <b>{h_model:.1f}%</b></p>
                                        <p style="margin:5px 0; color: {edge_color};">Edge: <b>{h_edge:+.1f}%</b></p>
                                        <p style="margin:5px 0; color: #00b894;"><b>{rec_badge}</b></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with vb_col2:
                                    d_odds = item.get('draw_odds') or 0
                                    d_implied = (detail.get('vb_draw_implied_prob') or 0) * 100
                                    d_model = item['draw_prob'] * 100
                                    d_edge = (detail.get('vb_draw_edge') or 0) * 100
                                    d_rec = detail.get('vb_draw_recommended')
                                    
                                    rec_badge = "✅ RECOMMENDED" if d_rec else ""
                                    edge_color = "#00b894" if d_edge >= 5 else ("#fdcb6e" if d_edge >= 0 else "#d63031")
                                    
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left: 4px solid {edge_color};">
                                        <h4 style="margin:0;">🤝 DRAW</h4>
                                        <p style="margin:5px 0;">Odds: <b>{d_odds:.2f}</b></p>
                                        <p style="margin:5px 0;">Implied: <b>{d_implied:.1f}%</b></p>
                                        <p style="margin:5px 0;">Model: <b>{d_model:.1f}%</b></p>
                                        <p style="margin:5px 0; color: {edge_color};">Edge: <b>{d_edge:+.1f}%</b></p>
                                        <p style="margin:5px 0; color: #00b894;"><b>{rec_badge}</b></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with vb_col3:
                                    a_odds = item.get('away_odds') or 0
                                    a_implied = (detail.get('vb_away_implied_prob') or 0) * 100
                                    a_model = item['away_win_prob'] * 100
                                    a_edge = (detail.get('vb_away_edge') or 0) * 100
                                    a_rec = detail.get('vb_away_recommended')
                                    
                                    rec_badge = "✅ RECOMMENDED" if a_rec else ""
                                    edge_color = "#00b894" if a_edge >= 5 else ("#fdcb6e" if a_edge >= 0 else "#d63031")
                                    
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left: 4px solid {edge_color};">
                                        <h4 style="margin:0;">✈️ AWAY</h4>
                                        <p style="margin:5px 0;">Odds: <b>{a_odds:.2f}</b></p>
                                        <p style="margin:5px 0;">Implied: <b>{a_implied:.1f}%</b></p>
                                        <p style="margin:5px 0;">Model: <b>{a_model:.1f}%</b></p>
                                        <p style="margin:5px 0; color: {edge_color};">Edge: <b>{a_edge:+.1f}%</b></p>
                                        <p style="margin:5px 0; color: #00b894;"><b>{rec_badge}</b></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Value Bet Legend
                                st.markdown("""
                                <p style="font-size: 0.8rem; color: #888; margin-top: 10px;">
                                📊 <b>Edge</b> = Model Prob - Implied Prob | 
                                🟢 Edge ≥5% = Value | 
                                🟡 Edge 0-5% = Neutral | 
                                🔴 Edge <0% = No Value
                                </p>
                                """, unsafe_allow_html=True)
                                
                                # Form Data
                                if detail.get('home_form') or detail.get('away_form'):
                                    st.markdown("---")
                                    st.markdown("**📊 Form Data Used**")
                                    form_col1, form_col2 = st.columns(2)
                                    with form_col1:
                                        st.markdown(f"**{home}:** `{detail.get('home_form', 'N/A')}`")
                                    with form_col2:
                                        st.markdown(f"**{away}:** `{detail.get('away_form', 'N/A')}`")
                                
                                # LLM Analysis
                                if detail.get('llm_analysis'):
                                    st.markdown("---")
                                    st.markdown("**🧠 AI Analysis**")
                                    st.markdown(f"""
                                    <div class="analysis-box">
                                    {detail['llm_analysis'][:1000]}{'...' if len(detail.get('llm_analysis', '')) > 1000 else ''}
                                    </div>
                                    """, unsafe_allow_html=True)
                        except:
                            pass
                        
                        # Update Result Section (if not verified)
                        if not item.get('actual_result'):
                            st.markdown("---")
                            st.markdown("**📝 Update Actual Result**")
                            update_col1, update_col2, update_col3, update_col4 = st.columns(4)
                            
                            with update_col1:
                                if st.button("🏠 Home Win", key=f"h_{item['id']}"):
                                    try:
                                        requests.put(f"{API_URL}/history/{item['id']}/result?actual_result=H", timeout=5)
                                        st.success("Updated to Home Win!")
                                        st.rerun()
                                    except:
                                        st.error("Failed to update")
                            
                            with update_col2:
                                if st.button("🤝 Draw", key=f"d_{item['id']}"):
                                    try:
                                        requests.put(f"{API_URL}/history/{item['id']}/result?actual_result=D", timeout=5)
                                        st.success("Updated to Draw!")
                                        st.rerun()
                                    except:
                                        st.error("Failed to update")
                            
                            with update_col3:
                                if st.button("✈️ Away Win", key=f"a_{item['id']}"):
                                    try:
                                        requests.put(f"{API_URL}/history/{item['id']}/result?actual_result=A", timeout=5)
                                        st.success("Updated to Away Win!")
                                        st.rerun()
                                    except:
                                        st.error("Failed to update")
                            
                            with update_col4:
                                if st.button("🗑️ Delete", key=f"del_{item['id']}"):
                                    try:
                                        requests.delete(f"{API_URL}/history/{item['id']}", timeout=5)
                                        st.success("Deleted!")
                                        st.rerun()
                                    except:
                                        st.error("Failed to delete")
            else:
                st.info("📭 No predictions found. Make some predictions first!")
                st.markdown("""
                **How to get started:**
                1. Go to **⚽ Predict Match** page
                2. Enter match details
                3. Generate a prediction
                4. Come back here to track your prediction history
                """)
        else:
            st.error(f"❌ Could not fetch history (Error {response.status_code})")
            
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API unavailable. Start the backend server first.")
        st.code("uvicorn app.main:app --reload --port 8000")
        
        st.markdown("""
        ### 🚀 Quick Start
        1. Open a terminal in the project folder
        2. Activate the virtual environment: `venv\\Scripts\\activate`
        3. Start the API: `uvicorn app.main:app --reload --port 8000`
        4. Refresh this page
        """)


def show_analytics():
    """Analytics and model performance page."""
    st.title("📈 Advanced Analytics Center")
    st.markdown("Access professional-grade tools for strategy validation and market analysis.")
    
    # Tabs for different analytics modules
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🧪 Backtesting Engine", "📉 Market Monitor", "🏆 Team Rankings"])
    
    # ==================== PERFORMANCE TAB ====================
    with tab1:
        st.subheader("System Performance")
        try:
            response = requests.get(f"{API_URL}/history/stats/performance", timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", stats['total_predictions'])
                with col2:
                    st.metric("Verified", stats['predictions_with_result'])
                with col3:
                    st.metric("Accuracy", f"{stats['accuracy']*100:.1f}%")
                with col4:
                    st.metric("ROI", f"{stats['roi']*100:.1f}%")
                
                st.markdown("---")
                
                # Model info
                st.subheader("🤖 Model Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **XGBoost Model (60% weight)**
                    - Algorithm: Gradient Boosted Trees
                    - Features: Form, xG, odds, H2H
                    - Objective: Multi-class classification
                    """)
                
                with col2:
                    st.markdown("""
                    **LSTM Model (40% weight)**
                    - Architecture: 2-layer LSTM (64, 32 units)
                    - Sequence: Last 5 matches
                    - Captures: Temporal patterns
                    """)
            else:
                st.error("Could not fetch analytics")
                
        except requests.exceptions.ConnectionError:
            st.warning("API unavailable")
    
    # ==================== BACKTESTING TAB ====================
    with tab2:
        st.subheader("🧪 Backtesting Engine")
        st.markdown("""
        **Validate strategies before betting.**
        This tool replays historical match data, effectively "traveling back in time" to test how the model WOULD have predicted matches in the past.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            test_limit = st.slider("Number of past matches to test", 10, 200, 50)
        with col2:
            st.info("Simulation assumes $100 stake per bet on high-confidence (>60%) predictions.")
            
        if st.button("▶️ Run Simulation", type="primary"):
            with st.spinner(f"Replaying history for last {test_limit} matches..."):
                try:
                    resp = requests.post(f"{API_URL}/analytics/backtest", json={"limit": test_limit}, timeout=120)
                    if resp.status_code == 200:
                        res = resp.json()
                        
                        # Metrics Row
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Matches Analyzed", res['total_matches_analyzed'])
                        m2.metric("Bets Placed", res['total_bets_placed'])
                        m3.metric("Win Rate", f"{res['win_rate']}%")
                        m4.metric("ROI", f"{res['roi_percent']}%", delta=f"{res['final_bankroll'] - res['initial_bankroll']:.2f}")
                        
                        st.markdown("### 📜 Transaction Log")
                        if res['history']:
                            for item in res['history']:
                                color = "green" if item['result'] == 'WIN' else "red"
                                icon = "✅" if item['result'] == 'WIN' else "❌"
                                st.markdown(f"""
                                <div style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                    <b>{item['match']}</b><br>
                                    Pred: {item['prediction']} @ {item['odds']} | Result: {icon} <b style="color:{color}">{item['result']}</b> (${item['pnl']})
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No bets placed. Model was too cautious (confidence < 60%).")
                    else:
                        st.error("Simulation failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ==================== MARKET MONITOR TAB ====================
    with tab3:
        st.subheader("📉 Market Monitor (Smart Money)")
        st.markdown("""
        **Detecting Sharp Money & Insider Info.**
        Real-time tracking of significant odds movements (>5% drop). Drops often indicate where the "smart money" is going.
        """)
        
        try:
            mm_resp = requests.get(f"{API_URL}/analytics/market-movement?threshold=0.05", timeout=10)
            if mm_resp.status_code == 200:
                alerts = mm_resp.json()
                
                if alerts:
                    st.markdown(f"**Found {len(alerts)} significant movements**")
                    for alert in alerts:
                        side = "HOME" if alert['alert_type'] == 'HOME_DROP' else "AWAY"
                        team = alert['home_team'] if side == "HOME" else alert['away_team']
                        
                        st.markdown(f"""
                        <div style="background: rgba(214, 48, 49, 0.1); border-left: 4px solid #d63031; padding: 15px; border-radius: 5px; margin: 10px 0;">
                            <h4 style="margin:0; color: #ff7675;">📉 SHARP DROP: {side} ({team})</h4>
                            <p style="margin:5px 0 0 0;">
                                Drop: <b>{alert['drop_percent']}%</b> | 
                                Open: {alert['opening_odds']:.2f} ➔ Current: {alert['current_odds']:.2f}
                            </p>
                            <small>{alert['home_team']} vs {alert['away_team']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No abnormal market movements detected currently. Market is stable.")
            else:
                st.warning("Could not fetch market data.")
        except:
            st.error("Market Monitor unavailable.")

    # ==================== RATINGS TAB ====================
    with tab4:
        st.subheader("🏆 Global Team Rankings")
        st.markdown("Power rankings calculated using **Opponent-Adjusted Elo Rating** system.")
        
        try:
            rat_resp = requests.get(f"{API_URL}/analytics/ratings", timeout=10)
            if rat_resp.status_code == 200:
                ratings = rat_resp.json()
                
                if ratings:
                    # Search
                    search = st.text_input("Find team", "")
                    
                    filtered = [r for r in ratings if search.lower() in r['team'].lower()]
                    
                    # Display as table with custom formatting
                    st.markdown("""
                    | Rank | Team | Elo Rating |
                    |------|------|------------|
                    """)
                    
                    for i, r in enumerate(filtered[:50]): # Show top 50
                        rank = i + 1
                        icon = "🔥" if r['rating'] > 1200 else ("❄️" if r['rating'] < 900 else "➖")
                        st.markdown(f"| #{rank} | **{r['team']}** | {r['rating']} {icon} |")
                else:
                    st.info("No ratings calculated yet. Run historical fetcher/backtest first.")
            else:
                st.warning("Ratings endpoint error.")
        except Exception as e:
            st.error(f"Error fetching ratings: {e}")



if __name__ == "__main__":
    main()

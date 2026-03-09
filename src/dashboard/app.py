import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.inference.predictor import YasenAlphaPredictor
from src.database import (
    authenticate_user, create_user, get_user_trades, 
    save_trade, update_user_balance, get_position, 
    save_position, delete_position
)

# Page config
st.set_page_config(
    page_title="YASEN-ALPHA Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #00ff88;
    }
    .signal-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #262730 0%, #1e1e1e 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 1px solid #3d3d3d;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        border-color: #00ff88;
    }
    .profit { color: #00ff88; font-weight: bold; }
    .loss { color: #ff4b4b; font-weight: bold; }
    .open { color: #ffa500; font-weight: bold; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .buy-button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: black;
    }
    .sell-button {
        background: linear-gradient(135deg, #ff4b4b 0%, #cc0000 100%);
        color: white;
    }
    .trade-input {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3d3d3d;
        margin-bottom: 20px;
    }
    .login-box {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-radius: 15px;
        border-left: 5px solid #00ff88;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Auto-refresh
st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

# ============================================
# AUTHENTICATION SECTION
# ============================================

def login_ui():
    """Display login/register interface."""
    st.markdown("""
        <div class="login-box">
            <h1 style="text-align:center; color:white;">🔐 YASEN-ALPHA</h1>
            <p style="text-align:center; color:#888;">Please login to continue</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state['user'] = user
                        st.session_state['authenticated'] = True
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username", help="Choose a unique username")
                new_email = st.text_input("Email", help="Valid email address")
                new_password = st.text_input("Password", type="password", help="Min 8 characters")
                confirm_password = st.text_input("Confirm Password", type="password")
                initial_balance = st.number_input("Starting Balance ($)", min_value=10, max_value=1000000, value=100, step=10)
                
                submitted = st.form_submit_button("Register", use_container_width=True)
                
                if submitted:
                    errors = []
                    if len(new_username) < 3:
                        errors.append("Username must be at least 3 characters")
                    if '@' not in new_email or '.' not in new_email:
                        errors.append("Invalid email format")
                    if len(new_password) < 8:
                        errors.append("Password must be at least 8 characters")
                    if new_password != confirm_password:
                        errors.append("Passwords don't match")
                    
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        success = create_user(new_username, new_email, new_password, initial_balance)
                        if success:
                            st.success("Account created! Please login.")
                        else:
                            st.error("Username or email already exists")

# Check authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_ui()
    st.stop()

# Get current user
user = st.session_state['user']
user_id = user['id']

# Load user's data from database
trades_list = get_user_trades(user_id)
open_position_dict = get_position(user_id)

# Initialize session state with database data
if trades_list:
    st.session_state.trades = []
    for trade in trades_list:
        st.session_state.trades.append({
            'entry_date': trade['entry_date'],
            'exit_date': trade['exit_date'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'pnl_pct': trade['pnl_pct'],
            'pnl_usd': trade['pnl_usd'],
            'confidence': trade['confidence'],
            'signal_type': trade['signal_type'],
            'trade_type': trade.get('trade_type', 'auto')
        })
else:
    st.session_state.trades = []

if open_position_dict:
    st.session_state.open_position = {
        'entry_date': open_position_dict['entry_date'],
        'entry_price': open_position_dict['entry_price'],
        'size': open_position_dict['size'],
        'confidence': open_position_dict['confidence'],
        'signal_type': open_position_dict['signal_type']
    }
else:
    st.session_state.open_position = None

st.session_state.account_balance = user['current_balance']
st.session_state.initial_balance = user['initial_balance']

# Header with user info
st.markdown(f"""
    <div class="main-header">
        <h1 style="margin:0; color:white;">📊 YASEN-ALPHA Trading System</h1>
        <p style="margin:0; color:#888;">Welcome back, {user['username']}! | Balance: ${user['current_balance']:,.2f}</p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# REST OF DASHBOARD CODE
# ============================================

# Load predictor
@st.cache_resource(ttl=300)
def load_predictor():
    return YasenAlphaPredictor()

predictor = load_predictor()

# Get current signal
signal = predictor.get_current_signal()

# Signal color
signal_color = {
    'BUY': '#00ff88',
    'SELL': '#ff4b4b',
    'HOLD': '#ffa500'
}.get(signal['signal'], 'gray')

# Main signal display
st.markdown(f"""
    <div class="signal-box" style="border-left-color: {signal_color};">
        <h2 style="color: {signal_color}; margin:0;">{signal['signal']} SIGNAL</h2>
        <p style="font-size: 36px; margin:10px 0 0 0; color:white;">Confidence: {signal['confidence']:.1%}</p>
        <p style="color:#888; margin:5px 0 0 0;">Threshold: {signal['threshold_used']:.2f} | Volatility: {signal['volatility']:.4f}</p>
    </div>
""", unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Current Signal", signal['signal'])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Confidence", f"{signal['confidence']:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Volatility", f"{signal['volatility']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Account Balance", f"${st.session_state.account_balance:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    total_pnl = st.session_state.account_balance - st.session_state.initial_balance
    pnl_pct = (total_pnl / st.session_state.initial_balance)
    delta_color = "normal" if total_pnl >= 0 else "inverse"
    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:.2%}", delta_color=delta_color)
    st.markdown('</div>', unsafe_allow_html=True)

# Load chart data
@st.cache_data(ttl=300)
def load_chart_data():
    try:
        df = pd.read_parquet('data/processed/features_latest.parquet')
        
        # Get historical confidence scores
        model = load_predictor()
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].tail(336)
        
        # Calculate confidence scores
        confidence_scores = []
        for i in range(len(X)):
            try:
                pred = model.model['models'][0].predict_proba(X.iloc[i:i+1])[0][1]
                confidence_scores.append(pred)
            except:
                confidence_scores.append(0.5)
        
        df_with_confidence = df.tail(336).copy()
        df_with_confidence['confidence'] = confidence_scores
        return df_with_confidence
    except Exception as e:
        st.error(f"Error loading chart data: {e}")
        return pd.DataFrame()

df = load_chart_data()

if not df.empty:
    current_price = df['close'].iloc[-1]
    
    # Manual Trade Entry
    with st.expander("📝 Manual Trade Entry (Track Trades You Made Elsewhere)", expanded=False):
        st.markdown('<div class="trade-input">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            manual_entry = st.number_input("Entry Price ($)", min_value=0.01, value=float(current_price), step=100.0)
        with col2:
            manual_exit = st.number_input("Exit Price ($)", min_value=0.01, value=float(current_price), step=100.0)
        with col3:
            manual_confidence = st.slider("Confidence (%)", min_value=0, max_value=100, value=50) / 100
        with col4:
            manual_signal = st.selectbox("Signal Type", ["BUY", "SELL", "HOLD"])
        
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("➕ Add Manual Trade", use_container_width=True):
                pnl_pct = (manual_exit - manual_entry) / manual_entry
                pnl_usd = pnl_pct * st.session_state.account_balance * 0.02 * 50
                
                trade_data = {
                    'entry_date': datetime.now().isoformat(),
                    'exit_date': datetime.now().isoformat(),
                    'entry_price': manual_entry,
                    'exit_price': manual_exit,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'confidence': manual_confidence,
                    'signal_type': manual_signal,
                    'trade_type': 'manual'
                }
                
                save_trade(user_id, trade_data)
                st.session_state.trades.append(trade_data)
                st.session_state.account_balance += pnl_usd
                update_user_balance(user_id, st.session_state.account_balance)
                
                st.success(f"✅ Manual trade added! P&L: ${pnl_usd:,.2f} ({pnl_pct:.2%})")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto Trading Section
    st.markdown("### 🤖 Auto Trading")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if signal['signal'] == 'BUY' and signal['confidence'] > 0.45 and not st.session_state.open_position:
            if st.button("💰 EXECUTE BUY", type="primary", use_container_width=True):
                position_size = (st.session_state.account_balance * 0.02) / current_price
                position_data = {
                    'entry_date': datetime.now().isoformat(),
                    'entry_price': current_price,
                    'size': position_size,
                    'confidence': signal['confidence'],
                    'signal_type': 'BUY'
                }
                st.session_state.open_position = position_data
                save_position(user_id, position_data)
                st.success(f"✅ Bought {position_size:.4f} BTC at ${current_price:,.2f}")
                st.rerun()
    
    with col2:
        if st.session_state.open_position and st.button("📉 EXIT POSITION", type="secondary", use_container_width=True):
            entry = st.session_state.open_position['entry_price']
            pnl_pct = (current_price - entry) / entry
            pnl_usd = pnl_pct * st.session_state.open_position['size'] * entry
            
            st.session_state.account_balance += pnl_usd
            update_user_balance(user_id, st.session_state.account_balance)
            
            trade_data = {
                'entry_date': st.session_state.open_position['entry_date'],
                'exit_date': datetime.now().isoformat(),
                'entry_price': entry,
                'exit_price': current_price,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'confidence': st.session_state.open_position['confidence'],
                'signal_type': st.session_state.open_position['signal_type'],
                'trade_type': 'auto'
            }
            
            save_trade(user_id, trade_data)
            delete_position(user_id)
            st.session_state.open_position = None
            st.session_state.trades.append(trade_data)
            
            if pnl_usd > 0:
                st.success(f"✅ Sold at ${current_price:,.2f} | Profit: ${pnl_usd:,.2f} ({pnl_pct:.2%})")
            else:
                st.error(f"❌ Sold at ${current_price:,.2f} | Loss: ${pnl_usd:,.2f} ({pnl_pct:.2%})")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Account", use_container_width=True):
            st.session_state.trades = []
            st.session_state.open_position = None
            st.session_state.account_balance = st.session_state.initial_balance
            update_user_balance(user_id, st.session_state.initial_balance)
            st.success("✅ Account reset to initial balance")
            st.rerun()

    # Professional tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Confidence", "📊 Market Analysis", "📋 Trade History", "📈 P&L Performance"])
    
    with tab1:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=('BTC Price with Entry Points', 'Trading Volume', 'Model Confidence', 'Volatility')
        )

        # Price line
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'],
                      mode='lines', name='BTC Price',
                      line=dict(color='gold', width=2)),
            row=1, col=1
        )

        # Add entry points
        for trade in st.session_state.trades:
            trade_date = pd.to_datetime(trade['entry_date'])
            if trade_date in df.index:
                fig.add_trace(
                    go.Scatter(x=[trade_date], y=[trade['entry_price']],
                              mode='markers', name='Entry',
                              marker=dict(color='lime' if trade['pnl_usd'] > 0 else 'red', 
                                        size=12, symbol='triangle-up',
                                        line=dict(color='white', width=1)),
                              showlegend=False),
                    row=1, col=1
                )

        # Add open position
        if st.session_state.open_position:
            entry_date = pd.to_datetime(st.session_state.open_position['entry_date'])
            if entry_date in df.index:
                fig.add_trace(
                    go.Scatter(x=[entry_date], y=[st.session_state.open_position['entry_price']],
                              mode='markers', name='Open Position',
                              marker=dict(color='orange', size=15, symbol='star',
                                        line=dict(color='white', width=1)),
                              showlegend=False),
                    row=1, col=1
                )

        # Volume bars
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'],
                   name='Volume', marker_color='cyan',
                   marker_line_color='white', marker_line_width=1),
            row=2, col=1
        )

        # Actual Confidence scores
        fig.add_trace(
            go.Scatter(x=df.index, y=df['confidence'],
                      mode='lines', name='Confidence',
                      line=dict(color='lime', width=2),
                      fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'),
            row=3, col=1
        )
        
        # Add threshold line
        fig.add_hline(y=signal['threshold_used'], line_dash="dash", 
                     line_color="red", row=3, col=1,
                     annotation_text=f"Threshold: {signal['threshold_used']:.2f}")

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(24).std()
        fig.add_trace(
            go.Scatter(x=df.index, y=df['volatility'],
                      mode='lines', name='Volatility',
                      line=dict(color='orange', width=2),
                      fill='tozeroy', fillcolor='rgba(255,165,0,0.1)'),
            row=4, col=1
        )

        fig.update_layout(
            template='plotly_dark',
            height=900,
            showlegend=True,
            hovermode='x unified',
            title_text="YASEN-ALPHA Market Analysis Dashboard"
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=3, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Volatility", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Market Statistics")
            
            returns = df['close'].pct_change().dropna()
            
            stats_data = {
                'Metric': [
                    'Current Price',
                    '24h Change',
                    '7d Change',
                    '30d Change',
                    '24h High',
                    '24h Low',
                    'ATR (14)',
                    'RSI (14)',
                    'Volume 24h',
                    'Volatility (24h)'
                ],
                'Value': [
                    f"${current_price:,.2f}",
                    f"{((df['close'].iloc[-1]/df['close'].iloc[-2])-1)*100:.2f}%",
                    f"{((df['close'].iloc[-1]/df['close'].iloc[-7])-1)*100:.2f}%" if len(df) > 7 else "N/A",
                    f"{((df['close'].iloc[-1]/df['close'].iloc[-30])-1)*100:.2f}%" if len(df) > 30 else "N/A",
                    f"${df['high'].iloc[-24:].max():,.2f}",
                    f"${df['low'].iloc[-24:].min():,.2f}",
                    f"${df['close'].diff().abs().rolling(14).mean().iloc[-1]:.2f}",
                    f"{50 + (returns.iloc[-14:].mean() / returns.iloc[-14:].std() * 10):.1f}" if len(returns) > 14 else "N/A",
                    f"${df['volume'].iloc[-24:].mean():,.0f}",
                    f"{df['volatility'].iloc[-1]:.4f}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📈 Model Performance")
            
            if st.session_state.trades:
                trades_df = pd.DataFrame(st.session_state.trades)
                wins = len(trades_df[trades_df['pnl_usd'] > 0])
                losses = len(trades_df[trades_df['pnl_usd'] < 0])
                win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
                
                perf_data = {
                    'Metric': [
                        'Win Rate', 'Total Trades', 'Winning Trades', 'Losing Trades',
                        'Avg Win', 'Avg Loss', 'Largest Win', 'Largest Loss',
                        'Profit Factor', 'Sharpe Ratio (est)'
                    ],
                    'Value': [
                        f"{win_rate:.2%}", str(len(trades_df)), str(wins), str(losses),
                        f"${trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean():.2f}" if wins > 0 else "$0",
                        f"${trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].mean():.2f}" if losses > 0 else "$0",
                        f"${trades_df['pnl_usd'].max():.2f}" if len(trades_df) > 0 else "$0",
                        f"${trades_df['pnl_usd'].min():.2f}" if len(trades_df) > 0 else "$0",
                        f"{abs(trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum() / trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum()):.2f}" if losses > 0 else "N/A",
                        f"{((trades_df['pnl_usd'].mean() / trades_df['pnl_usd'].std()) * np.sqrt(252)):.2f}" if len(trades_df) > 1 else "N/A"
                    ]
                }
            else:
                perf_data = {
                    'Metric': ['Win Rate', 'Total Trades', 'Winning Trades', 'Losing Trades',
                              'Avg Win', 'Avg Loss', 'Largest Win', 'Largest Loss',
                              'Profit Factor', 'Sharpe Ratio (est)'],
                    'Value': ['0%', '0', '0', '0', '$0', '$0', '$0', '$0', 'N/A', 'N/A']
                }
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("📋 Complete Trade History")
        
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
            trades_df['pnl_usd'] = trades_df['pnl_usd'].apply(lambda x: f"${x:,.2f}")
            trades_df['confidence'] = trades_df['confidence'].apply(lambda x: f"{x:.1%}")
            trades_df['type'] = trades_df['trade_type'].apply(lambda x: '📝 Manual' if x == 'manual' else '🤖 Auto')
            
            def color_pnl(val):
                if isinstance(val, str):
                    if '-' in val or '−' in val:
                        return 'color: #ff4b4b'
                return 'color: #00ff88'
            
            styled_df = trades_df.style.applymap(color_pnl, subset=['pnl_pct', 'pnl_usd'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "entry_date": "Entry Date",
                    "exit_date": "Exit Date",
                    "entry_price": "Entry Price",
                    "exit_price": "Exit Price",
                    "pnl_pct": "P&L %",
                    "pnl_usd": "P&L $",
                    "confidence": "Confidence",
                    "signal_type": "Signal",
                    "type": "Type"
                }
            )
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Trades", len(trades_df))
            with col2:
                wins = len(trades_df[trades_df['pnl_usd'].str.replace('$', '').str.replace(',', '').astype(float) > 0])
                st.metric("Winning Trades", wins)
            with col3:
                losses = len(trades_df[trades_df['pnl_usd'].str.replace('$', '').str.replace(',', '').astype(float) < 0])
                st.metric("Losing Trades", losses)
            with col4:
                win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.2%}")
            with col5:
                auto_trades = len(trades_df[trades_df['trade_type'] == 'auto'])
                st.metric("Auto Trades", auto_trades)
        else:
            st.info("No trades yet. Use Auto Trading or Manual Entry to start!")

    with tab4:
        st.subheader("📈 P&L Performance Analysis")
        
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df = trades_df.sort_values('exit_date')
            
            trades_df['pnl_float'] = trades_df['pnl_usd'].str.replace('$', '').str.replace(',', '').astype(float)
            trades_df['cumulative_pnl'] = trades_df['pnl_float'].cumsum()
            trades_df['equity_curve'] = st.session_state.initial_balance + trades_df['cumulative_pnl']
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.4, 0.3, 0.3],
                subplot_titles=('Equity Curve', 'Individual Trade P&L', 'Win/Loss Distribution')
            )
            
            # Equity curve
            fig.add_trace(
                go.Scatter(x=trades_df['exit_date'], y=trades_df['equity_curve'],
                          mode='lines+markers', name='Account Balance',
                          line=dict(color='gold', width=3),
                          marker=dict(size=6, color='gold', symbol='circle'),
                          fill='tozeroy', fillcolor='rgba(255,215,0,0.1)'),
                row=1, col=1
            )
            
            fig.add_hline(y=st.session_state.initial_balance, line_dash="dash", 
                         line_color="gray", row=1, col=1,
                         annotation_text="Initial Balance")
            
            # Individual trade bars
            colors = ['lime' if x > 0 else 'red' for x in trades_df['pnl_float']]
            fig.add_trace(
                go.Bar(x=trades_df['exit_date'], 
                      y=trades_df['pnl_float'],
                      name='Trade P&L',
                      marker_color=colors,
                      marker_line_color='white', marker_line_width=1,
                      text=trades_df['pnl_float'].apply(lambda x: f'${x:,.2f}'),
                      textposition='outside'),
                row=2, col=1
            )
            
            # Win/Loss distribution
            wins = len(trades_df[trades_df['pnl_float'] > 0])
            losses = len(trades_df[trades_df['pnl_float'] < 0])
            
            fig.add_trace(
                go.Pie(labels=['Wins', 'Losses'], 
                       values=[wins, losses],
                       marker_colors=['lime', 'red'],
                       textinfo='label+percent',
                       hole=0.4,
                       showlegend=False),
                row=3, col=1
            )
            
            fig.update_layout(template='plotly_dark', height=900, showlegend=True, hovermode='x unified')
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Account Balance ($)", row=1, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_return = (st.session_state.account_balance - st.session_state.initial_balance) / st.session_state.initial_balance
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Final Balance", f"${st.session_state.account_balance:,.2f}")
            with col3:
                avg_trade = trades_df['pnl_float'].mean()
                st.metric("Avg Trade", f"${avg_trade:,.2f}", delta=f"{avg_trade/st.session_state.initial_balance:.2%}")
            with col4:
                max_drawdown = (trades_df['equity_curve'].min() - st.session_state.initial_balance) / st.session_state.initial_balance
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        else:
            st.info("No trades yet. Execute trades to see P&L performance!")

else:
    st.error("⚠️ No data available. Please run the data pipeline first: `python scripts/run_pipeline.py --stage all`")

# Sidebar
with st.sidebar:
    st.header(f"👤 User: {user['username']}")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state.pop('user', None)
        st.rerun()
    
    st.markdown("---")
    st.header("⚙️ Account Settings")
    
    new_balance = st.number_input(
        "Account Balance ($)",
        min_value=10,
        max_value=1000000,
        value=int(st.session_state.account_balance),
        step=10
    )
    if new_balance != st.session_state.account_balance:
        st.session_state.account_balance = float(new_balance)
        update_user_balance(user_id, new_balance)
        st.rerun()
    
    st.info("💰 **Fixed Risk: 2% per trade**")
    risk_per_trade = 0.02
    
    if signal['signal'] == 'BUY' and signal['confidence'] > 0.45 and not df.empty:
        position_value = st.session_state.account_balance * risk_per_trade * 50
        position_btc = position_value / current_price
        st.success(f"📈 **Recommended Position:** ${position_value:,.2f} ({position_btc:.4f} BTC)")
        st.caption(f"Risk Amount: ${st.session_state.account_balance * risk_per_trade:,.2f}")
    else:
        st.info("⏸️ No trade recommended at this time")
    
    st.markdown("---")
    
    if st.session_state.open_position:
        entry = st.session_state.open_position['entry_price']
        current_pnl = (current_price - entry) / entry if not df.empty else 0
        st.warning(f"📊 **Open Position**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Entry:** ${entry:,.2f}")
            st.write(f"**Current:** ${current_price:,.2f}")
        with col2:
            pnl_color = "profit" if current_pnl > 0 else "loss"
            st.markdown(f"**P&L:** <span class='{pnl_color}'>{current_pnl:.2%}</span>", unsafe_allow_html=True)
            st.write(f"**Size:** {st.session_state.open_position['size']:.4f} BTC")
    
    st.markdown("---")
    st.header("📋 Trading Rules")
    rules = [
        "✅ **Only trade BUY signals**",
        "✅ **Confidence > 45% required**",
        "✅ **Fixed 2% risk per trade**",
        "✅ **Always use stop losses**",
        "✅ **No emotional overrides**",
        "✅ **Trust the system**"
    ]
    for rule in rules:
        st.markdown(rule)
    
    st.markdown("---")
    st.caption(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"**Model Win Rate:** 59.19%")
    st.caption(f"**Total Backtested Trades:** 8,274")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("⚠️ **Risk Warning:** Trading involves substantial risk.")
with col2:
    st.caption(f"📊 **Stats:** {len(st.session_state.trades)} trades | ${st.session_state.account_balance:,.2f} balance")
with col3:
    st.caption("🤖 **Powered by:** YASEN-ALPHA ML Pipeline v2.0")

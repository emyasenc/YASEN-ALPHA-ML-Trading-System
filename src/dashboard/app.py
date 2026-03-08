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

# Page config
st.set_page_config(
    page_title="YASEN-ALPHA Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .signal-box { padding: 20px; border-radius: 10px; background-color: #1e1e1e; border-left: 5px solid #ff4b4b; margin-bottom: 20px; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; text-align: center; }
    .profit { color: #00ff00; font-weight: bold; }
    .loss { color: #ff4b4b; font-weight: bold; }
    .open { color: #ffa500; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Auto-refresh
st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

# Title
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 📊")
with col2:
    st.title("YASEN-ALPHA Bitcoin Trading System")
    st.markdown("### Production-Grade ML Pipeline • 59.19% Win Rate Champion")

# Initialize session state for trade tracking
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'open_position' not in st.session_state:
    st.session_state.open_position = None
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 1000.0
if 'initial_balance' not in st.session_state:
    st.session_state.initial_balance = 1000.0

# File to persist trades
TRADES_FILE = 'data/trades.json'

# Load trades from file if exists
def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f:
            data = json.load(f)
            st.session_state.trades = data.get('trades', [])
            st.session_state.open_position = data.get('open_position')
            st.session_state.account_balance = data.get('account_balance', 1000.0)
            st.session_state.initial_balance = data.get('initial_balance', 1000.0)

# Save trades to file
def save_trades():
    data = {
        'trades': st.session_state.trades,
        'open_position': st.session_state.open_position,
        'account_balance': st.session_state.account_balance,
        'initial_balance': st.session_state.initial_balance
    }
    os.makedirs('data', exist_ok=True)
    with open(TRADES_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# Load trades on startup
load_trades()

# Auto-refresh logic
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 300:
    st.session_state.last_refresh = time.time()
    st.rerun()

# Load predictor
@st.cache_resource(ttl=300)
def load_predictor():
    return YasenAlphaPredictor()

predictor = load_predictor()

# Get current signal
signal = predictor.get_current_signal()

# Signal color
signal_color = {
    'BUY': 'lime',
    'SELL': 'red',
    'HOLD': 'orange'
}.get(signal['signal'], 'gray')

# Main signal display
st.markdown(f"""
    <div class="signal-box" style="border-left-color: {signal_color};">
        <h2 style="color: {signal_color};">{signal['signal']} SIGNAL</h2>
        <p class="big-font">Confidence: {signal['confidence']:.1%}</p>
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
    st.metric("Volatility", f"{signal['volatility']:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Account Balance", f"${st.session_state.account_balance:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    total_pnl = st.session_state.account_balance - st.session_state.initial_balance
    pnl_pct = (total_pnl / st.session_state.initial_balance) * 100
    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Load chart data
@st.cache_data(ttl=300)
def load_chart_data():
    try:
        df = pd.read_parquet('data/processed/features_latest.parquet')
        return df.tail(336)
    except:
        return pd.DataFrame()

df = load_chart_data()

if not df.empty:
    current_price = df['close'].iloc[-1]
    
    # Trade execution buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if signal['signal'] == 'BUY' and signal['confidence'] > 0.6 and not st.session_state.open_position:
            if st.button("💰 EXECUTE BUY", type="primary", use_container_width=True):
                st.session_state.open_position = {
                    'entry_date': datetime.now().isoformat(),
                    'entry_price': current_price,
                    'size': st.session_state.account_balance * 0.02 / (current_price * 0.02),  # Position sizing
                    'confidence': signal['confidence'],
                    'signal_type': 'BUY'
                }
                save_trades()
                st.success(f"✅ Bought at ${current_price:,.2f}")
                st.rerun()
    
    with col2:
        if st.session_state.open_position and st.button("📉 EXIT POSITION", type="secondary", use_container_width=True):
            # Calculate P&L
            entry = st.session_state.open_position['entry_price']
            pnl_pct = (current_price - entry) / entry
            pnl_usd = pnl_pct * st.session_state.open_position['size'] * entry
            
            # Update account balance
            st.session_state.account_balance += pnl_usd
            
            # Record trade
            trade = {
                'entry_date': st.session_state.open_position['entry_date'],
                'exit_date': datetime.now().isoformat(),
                'entry_price': entry,
                'exit_price': current_price,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'confidence': st.session_state.open_position['confidence'],
                'signal_type': st.session_state.open_position['signal_type']
            }
            st.session_state.trades.append(trade)
            st.session_state.open_position = None
            save_trades()
            
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
            save_trades()
            st.success("✅ Account reset")
            st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Analysis", "📋 Trade History", "📈 P&L Performance"])
    
    with tab1:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('BTC Price', 'Trading Volume', 'Model Confidence')
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
                                        size=10, symbol='triangle-up'),
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
                              marker=dict(color='orange', size=12, symbol='star'),
                              showlegend=False),
                    row=1, col=1
                )

        # Volume bars
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'],
                   name='Volume', marker_color='cyan'),
            row=2, col=1
        )

        # Confidence placeholder
        fig.add_trace(
            go.Scatter(x=df.index, y=[0.5] * len(df),
                      mode='lines', name='Confidence',
                      line=dict(color='lime', width=2)),
            row=3, col=1
        )

        fig.update_layout(template='plotly_dark', height=700, showlegend=True)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=3, col=1, range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Market Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Current Price', '24h Change', '7d Change', '30d Change', 'ATR'],
                'Value': [
                    f"${current_price:,.0f}",
                    f"{(df['close'].iloc[-1]/df['close'].iloc[-2]-1)*100:.2f}%",
                    f"{(df['close'].iloc[-1]/df['close'].iloc[-7]-1)*100:.2f}%" if len(df) > 7 else "N/A",
                    f"{(df['close'].iloc[-1]/df['close'].iloc[-30]-1)*100:.2f}%" if len(df) > 30 else "N/A",
                    f"${df['close'].diff().abs().rolling(14).mean().iloc[-1]:.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📈 Model Performance")
            if st.session_state.trades:
                trades_df = pd.DataFrame(st.session_state.trades)
                wins = len(trades_df[trades_df['pnl_usd'] > 0])
                losses = len(trades_df[trades_df['pnl_usd'] < 0])
                win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
                
                perf_df = pd.DataFrame({
                    'Metric': ['Win Rate', 'Total Trades', 'Wins', 'Losses', 'Avg Win', 'Avg Loss'],
                    'Value': [
                        f"{win_rate:.2%}",
                        str(len(trades_df)),
                        str(wins),
                        str(losses),
                        f"${trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].mean():.2f}" if wins > 0 else "$0",
                        f"${trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].mean():.2f}" if losses > 0 else "$0"
                    ]
                })
            else:
                perf_df = pd.DataFrame({
                    'Metric': ['Win Rate', 'Total Trades', 'Wins', 'Losses', 'Avg Win', 'Avg Loss'],
                    'Value': ['0%', '0', '0', '0', '$0', '$0']
                })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("📋 Complete Trade History")
        
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
            trades_df['pnl_usd'] = trades_df['pnl_usd'].apply(lambda x: f"${x:.2f}")
            trades_df['confidence'] = trades_df['confidence'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(
                trades_df,
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
                    "signal_type": "Signal"
                }
            )
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", len(trades_df))
            with col2:
                wins = len(trades_df[trades_df['pnl_usd'].str.replace('$', '').astype(float) > 0])
                st.metric("Winning Trades", wins)
            with col3:
                losses = len(trades_df[trades_df['pnl_usd'].str.replace('$', '').astype(float) < 0])
                st.metric("Losing Trades", losses)
            with col4:
                win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.2%}")
        else:
            st.info("No trades yet. Execute a BUY signal to start trading!")

    with tab4:
        st.subheader("📈 P&L Performance")
        
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df = trades_df.sort_values('exit_date')
            
            # Calculate cumulative P&L
            trades_df['cumulative_pnl'] = trades_df['pnl_usd'].str.replace('$', '').astype(float).cumsum()
            
            # Create P&L chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4],
                subplot_titles=('Cumulative P&L', 'Individual Trade P&L')
            )
            
            # Cumulative P&L line
            fig.add_trace(
                go.Scatter(x=trades_df['exit_date'], y=trades_df['cumulative_pnl'],
                          mode='lines+markers', name='Cumulative P&L',
                          line=dict(color='gold', width=3),
                          marker=dict(size=8)),
                row=1, col=1
            )
            
            # Individual trade bars
            colors = ['lime' if x > 0 else 'red' for x in trades_df['pnl_usd'].str.replace('$', '').astype(float)]
            fig.add_trace(
                go.Bar(x=trades_df['exit_date'], 
                      y=trades_df['pnl_usd'].str.replace('$', '').astype(float),
                      name='Trade P&L',
                      marker_color=colors),
                row=2, col=1
            )
            
            fig.update_layout(template='plotly_dark', height=600, showlegend=False)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
            fig.update_yaxes(title_text="Trade P&L ($)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final stats
            total_pnl = trades_df['pnl_usd'].str.replace('$', '').astype(float).sum()
            st.metric("Total P&L", f"${total_pnl:,.2f}", 
                     delta=f"{(total_pnl/st.session_state.initial_balance):.2%}")
        else:
            st.info("No trades yet. Execute trades to see P&L performance!")

else:
    st.warning("⚠️ No data available. Please run the data pipeline first.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Trading Dashboard")
    
    account_balance = st.number_input(
        "Account Balance ($)",
        min_value=100,
        max_value=1000000,
        value=int(st.session_state.account_balance),
        step=100,
        key="account_input"
    )
    st.session_state.account_balance = float(account_balance)
    
    risk_per_trade = st.slider(
        "Risk Per Trade (%)",
        min_value=0.5,
        max_value=3.0,
        value=2.0,
        step=0.5
    ) / 100
    
    st.markdown("---")
    
    # Position info
    if st.session_state.open_position:
        entry = st.session_state.open_position['entry_price']
        current_pnl = (current_price - entry) / entry if not df.empty else 0
        st.warning(f"📊 **Open Position**")
        st.write(f"Entry: ${entry:,.2f}")
        st.write(f"Current: ${current_price:,.2f}")
        st.write(f"P&L: {current_pnl:.2%}")
    
    st.markdown("---")
    st.header("📋 Trading Rules")
    for rule in ["✅ Only trade BUY signals", "✅ Confidence > 60%", "✅ Risk max 2%", "✅ Always use stops", "✅ No emotions"]:
        st.markdown(rule)
    
    st.markdown("---")
    st.caption(f"Trades Saved: {TRADES_FILE}")
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("⚠️ **Risk Warning:** Trading involves risk. Never risk more than you can afford to lose.")
with col2:
    st.caption(f"📊 **Trades:** {len(st.session_state.trades)} | **Balance:** ${st.session_state.account_balance:,.2f}")
with col3:
    st.caption("🤖 **Powered by:** YASEN-ALPHA ML Pipeline")

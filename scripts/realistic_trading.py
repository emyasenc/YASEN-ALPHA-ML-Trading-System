import pandas as pd
import numpy as np
import joblib

print("="*60)
print("YASEN-ALPHA: REALISTIC TRADING SIMULATION")
print("="*60)

# Load data and model
df = pd.read_parquet('data/processed/btc_with_features.parquet')
model = joblib.load('data/models/yasen_alpha_v1.joblib')

# Prepare features
exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].dropna()

# Get predictions
probabilities = model.predict_proba(X)[:, 1]

# Trading parameters
initial_capital = 1000  # Start with $1000
risk_per_trade = 0.02   # Risk 2% per trade
threshold = 0.34        # Use threshold just below 0.35 for more trades

# Create trading simulation
df_trade = df.loc[X.index].copy()
df_trade['prediction'] = probabilities
df_trade['signal'] = (probabilities > threshold).astype(int)

# Calculate actual returns
df_trade['next_return'] = df_trade['close'].pct_change().shift(-1)

# Trading simulation with risk management
capital = initial_capital
trades = []
equity_curve = []

for i in range(len(df_trade) - 1):  # -1 because we need next return
    if df_trade['signal'].iloc[i] == 1:
        # Enter trade
        entry_price = df_trade['close'].iloc[i]
        exit_price = df_trade['close'].iloc[i + 1]
        
        # Calculate return (simplified - no leverage)
        trade_return = (exit_price - entry_price) / entry_price
        
        # Risk management: only risk 2% of capital
        position_size = (capital * risk_per_trade) / abs(trade_return) if trade_return != 0 else 0
        position_size = min(position_size, capital)  # Can't risk more than capital
        
        trade_pnl = position_size * trade_return
        capital += trade_pnl
        
        trades.append({
            'date': df_trade.index[i],
            'entry': entry_price,
            'exit': exit_price,
            'return': trade_return,
            'pnl': trade_pnl,
            'capital': capital
        })
    
    equity_curve.append({'date': df_trade.index[i], 'capital': capital})

# Convert to DataFrame
trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

# Calculate metrics
if len(trades_df) > 0:
    total_trades = len(trades_df)
    winning_trades = (trades_df['return'] > 0).sum()
    win_rate = winning_trades / total_trades
    total_return = (capital - initial_capital) / initial_capital * 100
    avg_win = trades_df[trades_df['return'] > 0]['return'].mean() * 100 if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() * 100 if (total_trades - winning_trades) > 0 else 0
    
    print("\n" + "="*60)
    print("📊 TRADING SIMULATION RESULTS")
    print("="*60)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades) + 0.001)):.2f}")
    
    # Monthly returns
    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly = trades_df.groupby('month').agg({
        'pnl': 'sum',
        'return': 'count'
    }).rename(columns={'return': 'trades'})
    monthly['monthly_return'] = monthly['pnl'] / initial_capital * 100
    
    print("\n📆 Best 3 Months:")
    print(monthly.nlargest(3, 'monthly_return')[['monthly_return', 'trades']].to_string())
    
    print("\n📆 Worst 3 Months:")
    print(monthly.nsmallest(3, 'monthly_return')[['monthly_return', 'trades']].to_string())
    
    # Save results
    trades_df.to_csv('data/simulation_trades.csv', index=False)
    equity_df.to_csv('data/simulation_equity.csv', index=False)
    print("\n💾 Simulation results saved to data/simulation_trades.csv")
else:
    print("No trades generated with this threshold")

# Try different thresholds
print("\n" + "="*60)
print("📊 THRESHOLD OPTIMIZATION")
print("="*60)

thresholds = [0.32, 0.33, 0.34, 0.35, 0.36]
for thresh in thresholds:
    signals = (probabilities > thresh).astype(int)
    trades = signals.sum()
    if trades > 0:
        win_rate = ((signals == 1) & (df_trade['next_return'] > 0)).sum() / trades
        print(f"Threshold {thresh:.2f}: {trades} trades, {win_rate:.2%} win rate")
    else:
        print(f"Threshold {thresh:.2f}: 0 trades")

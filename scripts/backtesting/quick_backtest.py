import pandas as pd
import numpy as np
import joblib

print("="*60)
print("YASEN-ALPHA: QUICK BACKTEST")
print("="*60)

# Load data and model
print("\n📂 Loading data and model...")
df = pd.read_parquet('data/processed/btc_with_features.parquet')
model = joblib.load('data/models/yasen_alpha_v1.joblib')

# Prepare features
exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].dropna()

print(f"✅ Loaded {len(X):,} test samples")

# Make predictions
print("\n🤖 Generating predictions...")
probabilities = model.predict_proba(X)[:, 1]

# Create backtest dataframe
df_backtest = df.loc[X.index].copy()
df_backtest['prediction'] = probabilities
df_backtest['signal'] = (probabilities > 0.6).astype(int)  # Only trade when >60% confident
df_backtest['actual_return'] = df_backtest['close'].pct_change().shift(-1)  # Next hour's return

# Calculate strategy returns
df_backtest['strategy_return'] = df_backtest['signal'] * df_backtest['actual_return']
df_backtest['strategy_return'] = df_backtest['strategy_return'].fillna(0)

# Metrics
total_trades = df_backtest['signal'].sum()
winning_trades = ((df_backtest['signal'] == 1) & (df_backtest['actual_return'] > 0)).sum()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

# Calculate cumulative returns
strategy_return = (1 + df_backtest['strategy_return']).prod() - 1
bh_return = (1 + df_backtest['actual_return'].fillna(0)).prod() - 1

print("\n" + "="*60)
print("📊 BACKTEST RESULTS")
print("="*60)
print(f"Total trades: {total_trades}")
print(f"Win rate: {win_rate:.2%}")
print(f"Strategy return: {strategy_return:.2%}")
print(f"Buy & hold return: {bh_return:.2%}")
print(f"Excess return: {strategy_return - bh_return:.2%}")
print("="*60)

# Monthly breakdown
df_backtest['year_month'] = df_backtest.index.to_period('M')
monthly = df_backtest.groupby('year_month').agg({
    'strategy_return': 'sum',
    'signal': 'sum'
}).rename(columns={'strategy_return': 'monthly_return', 'signal': 'trades'})

print("\n📆 Best 5 Months:")
print(monthly.nlargest(5, 'monthly_return').to_string())

print("\n📆 Worst 5 Months:")
print(monthly.nsmallest(5, 'monthly_return').to_string())

# Save results
results = pd.DataFrame({
    'metric': ['total_trades', 'win_rate', 'strategy_return', 'bh_return', 'excess_return'],
    'value': [total_trades, win_rate, strategy_return, bh_return, strategy_return - bh_return]
})
results.to_csv('data/backtest_results.csv', index=False)
print("\n💾 Results saved to data/backtest_results.csv")

# Quick equity curve plot (if matplotlib is installed)
try:
    import matplotlib.pyplot as plt
    df_backtest['strategy_equity'] = (1 + df_backtest['strategy_return']).cumprod()
    df_backtest['bh_equity'] = (1 + df_backtest['actual_return'].fillna(0)).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_backtest.index, df_backtest['strategy_equity'], label='Strategy', linewidth=2)
    plt.plot(df_backtest.index, df_backtest['bh_equity'], label='Buy & Hold', linewidth=2, alpha=0.7)
    plt.title('YASEN-ALPHA Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Equity (1.0 = start)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data/backtest_equity.png')
    print("📈 Equity curve saved to data/backtest_equity.png")
except:
    print("📈 Install matplotlib for equity curve visualization")

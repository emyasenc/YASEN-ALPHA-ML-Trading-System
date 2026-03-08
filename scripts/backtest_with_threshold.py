import pandas as pd
import numpy as np
import joblib

print("="*60)
print("YASEN-ALPHA: OPTIMIZED BACKTEST")
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

# Try different thresholds
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
results = []

print("\n📊 Testing different thresholds...")
print("-" * 80)
print(f"{'Threshold':>10} {'Trades':>10} {'Win Rate':>10} {'Strategy':>12} {'Buy&Hold':>12} {'Excess':>12}")
print("-" * 80)

for thresh in thresholds:
    # Create signals
    signals = (probabilities > thresh).astype(int)
    
    # Calculate returns
    df_backtest = df.loc[X.index].copy()
    df_backtest['signal'] = signals
    df_backtest['actual_return'] = df_backtest['close'].pct_change().shift(-1)
    df_backtest['strategy_return'] = df_backtest['signal'] * df_backtest['actual_return']
    df_backtest['strategy_return'] = df_backtest['strategy_return'].fillna(0)
    
    # Metrics
    total_trades = signals.sum()
    if total_trades > 0:
        winning_trades = ((signals == 1) & (df_backtest['actual_return'] > 0)).sum()
        win_rate = winning_trades / total_trades
    else:
        win_rate = 0
    
    strategy_return = (1 + df_backtest['strategy_return']).prod() - 1
    bh_return = (1 + df_backtest['actual_return'].fillna(0)).prod() - 1
    
    results.append({
        'threshold': thresh,
        'trades': total_trades,
        'win_rate': win_rate,
        'strategy_return': strategy_return,
        'bh_return': bh_return,
        'excess_return': strategy_return - bh_return
    })
    
    print(f"{thresh:10.2f} {total_trades:10d} {win_rate:10.2%} {strategy_return:12.2%} {bh_return:12.2%} {strategy_return - bh_return:12.2%}")

# Find best threshold
best = max(results, key=lambda x: x['excess_return'])
print("\n" + "="*60)
print(f"✅ OPTIMAL THRESHOLD: {best['threshold']:.2f}")
print(f"   Trades: {best['trades']}")
print(f"   Win Rate: {best['win_rate']:.2%}")
print(f"   Strategy Return: {best['strategy_return']:.2%}")
print(f"   Buy & Hold Return: {best['bh_return']:.2%}")
print(f"   Excess Return: {best['excess_return']:.2%}")
print("="*60)

# Save results
pd.DataFrame(results).to_csv('data/threshold_analysis.csv', index=False)
print("\n💾 Results saved to data/threshold_analysis.csv")

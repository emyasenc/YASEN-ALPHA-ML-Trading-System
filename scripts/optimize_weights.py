import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("="*60)
print("YASEN-ALPHA: ENSEMBLE WEIGHT OPTIMIZATION")
print("="*60)

# Load data
df = pd.read_parquet('data/processed/btc_with_features.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

X = df_clean[feature_cols]
y = df_clean['target']

# Load the 5 models from ensemble
ensemble_data = joblib.load('data/models/yasen_alpha_ensemble_v2.joblib')
models = ensemble_data['models']

# Get individual predictions
individual_preds = []
for i, model in enumerate(models):
    pred = model.predict_proba(X)[:, 1]
    individual_preds.append(pred)
    print(f"Model {i+1} predictions loaded")

# Try different weight combinations
best_win_rate = 0
best_weights = None
best_threshold = 0

print("\n🔍 Searching for optimal weights...")

# Grid search over weights (simplified - sum to 1)
for w1 in np.arange(0.1, 0.5, 0.05):
    for w2 in np.arange(0.1, 0.5, 0.05):
        for w3 in np.arange(0.1, 0.5, 0.05):
            for w4 in np.arange(0.1, 0.5, 0.05):
                w5 = 1 - (w1 + w2 + w3 + w4)
                if w5 < 0.1 or w5 > 0.5:
                    continue
                
                weights = [w1, w2, w3, w4, w5]
                
                # Weighted ensemble
                ensemble_pred = np.zeros(len(X))
                for i, pred in enumerate(individual_preds):
                    ensemble_pred += weights[i] * pred
                
                # Find best threshold for these weights
                for thresh in np.arange(0.35, 0.45, 0.01):
                    signals = (ensemble_pred > thresh).astype(int)
                    trades = signals.sum()
                    
                    if trades > 0:
                        actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
                        wins = ((signals == 1) & (actual_returns > 0)).sum()
                        win_rate = wins / trades
                        
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            best_weights = weights
                            best_threshold = thresh
                            print(f"\n🎯 New best: {win_rate:.4f}")
                            print(f"   Weights: {[round(w,2) for w in weights]}")
                            print(f"   Threshold: {thresh}")

print("\n" + "="*60)
print(f"🏆 OPTIMAL CONFIGURATION")
print(f"Win Rate: {best_win_rate:.4f}")
print(f"Weights: {[round(w,2) for w in best_weights]}")
print(f"Threshold: {best_threshold}")
print("="*60)

# Calculate final predictions with optimal weights
final_pred = np.zeros(len(X))
for i, pred in enumerate(individual_preds):
    final_pred += best_weights[i] * pred

signals = (final_pred > best_threshold).astype(int)
trades = signals.sum()
actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
wins = ((signals == 1) & (actual_returns > 0)).sum()
win_rate = wins / trades

# Calculate realistic returns
initial_capital = 1000
capital = initial_capital
risk_per_trade = 0.02

trade_history = []
for i in range(len(X) - 1):
    if signals[i] == 1:
        entry = df_clean['close'].iloc[i]
        exit_price = df_clean['close'].iloc[i + 1]
        trade_return = (exit_price - entry) / entry
        
        position = capital * risk_per_trade / abs(trade_return) if trade_return != 0 else 0
        position = min(position, capital)
        
        pnl = position * trade_return
        capital += pnl
        
        trade_history.append({
            'date': df_clean.index[i],
            'return': trade_return,
            'pnl': pnl,
            'capital': capital
        })

print(f"\n📊 Realistic Simulation:")
print(f"Starting: ${initial_capital:,.2f}")
print(f"Ending: ${capital:,.2f}")
print(f"Return: {(capital/initial_capital - 1)*100:.2f}%")
print(f"Trades: {len(trade_history)}")

# Save optimized model
joblib.dump({
    'models': models,
    'weights': best_weights,
    'threshold': best_threshold,
    'win_rate': best_win_rate,
    'features': feature_cols
}, 'data/models/yasen_alpha_optimized_weights.joblib')
print("\n💾 Optimized ensemble saved!")

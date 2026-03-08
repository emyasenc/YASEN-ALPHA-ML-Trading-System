import pandas as pd
import numpy as np
import joblib

print("="*60)
print("YASEN-ALPHA: DYNAMIC THRESHOLD OPTIMIZATION")
print("="*60)

# Load your best model
model_data = joblib.load('data/models/yasen_alpha_optimized_weights.joblib')
models = model_data['models']
weights = model_data['weights']
base_threshold = model_data['threshold']

# Load data
df = pd.read_parquet('data/processed/btc_with_features.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

feature_cols = [col for col in df_clean.columns if col not in 
                ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']]
X = df_clean[feature_cols]

# Get base predictions
predictions = np.zeros(len(X))
for i, model in enumerate(models):
    predictions += weights[i] * model.predict_proba(X)[:, 1]

# Calculate volatility for dynamic threshold
df_clean['volatility'] = df_clean['close'].pct_change().rolling(24).std()
vol_percentiles = {
    'low': df_clean['volatility'].quantile(0.33),
    'medium': df_clean['volatility'].quantile(0.66),
    'high': df_clean['volatility'].max()
}

print(f"\n📊 Volatility Regimes:")
print(f"Low: < {vol_percentiles['low']:.4f}")
print(f"Medium: {vol_percentiles['low']:.4f} - {vol_percentiles['medium']:.4f}")
print(f"High: > {vol_percentiles['medium']:.4f}")

# Try different thresholds per regime
best_config = None
best_win_rate = 0

for low_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
    for med_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for high_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
            
            # Dynamic thresholds
            low_thresh = base_threshold * low_mult
            med_thresh = base_threshold * med_mult
            high_thresh = base_threshold * high_mult
            
            signals = np.zeros(len(X))
            for i in range(len(X)):
                vol = df_clean['volatility'].iloc[i]
                if pd.isna(vol):
                    continue
                if vol <= vol_percentiles['low']:
                    signals[i] = predictions[i] > low_thresh
                elif vol <= vol_percentiles['medium']:
                    signals[i] = predictions[i] > med_thresh
                else:
                    signals[i] = predictions[i] > high_thresh
            
            trades = signals.sum()
            if trades > 3000:
                actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
                wins = ((signals == 1) & (actual_returns > 0)).sum()
                win_rate = wins / trades
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_config = (low_mult, med_mult, high_mult)
                    print(f"\n🎯 New best: {win_rate:.4f}")
                    print(f"   Multipliers: Low={low_mult}, Med={med_mult}, High={high_mult}")

print("\n" + "="*60)
print(f"🏆 DYNAMIC THRESHOLD RESULT")
print(f"Win Rate: {best_win_rate:.4f}")
print(f"Multipliers: {best_config}")
print("="*60)

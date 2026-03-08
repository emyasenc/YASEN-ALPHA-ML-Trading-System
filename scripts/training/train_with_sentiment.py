import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

print("="*60)
print("YASEN-ALPHA: TRAINING WITH SENTIMENT")
print("="*60)

# Load enhanced data
df = pd.read_parquet('data/processed/btc_with_sentiment.parquet')
print(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")

# Create target
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

# Feature columns (including new sentiment features)
exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

X = df_clean[feature_cols]
y = df_clean['target']

print(f"\n📊 Training with {len(feature_cols)} features")
print(f"📈 Class balance: {y.mean():.2%}")

# Train ensemble with 5 seeds
models = []
for seed in [42, 123, 456, 789, 999]:
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X, y)
    models.append(model)
    print(f"✅ Model {seed} trained")

# Ensemble predictions
predictions = np.zeros(len(X))
for model in models:
    predictions += model.predict_proba(X)[:, 1]
predictions /= len(models)

# Find optimal threshold
best_win_rate = 0
best_thresh = 0

for thresh in np.arange(0.35, 0.45, 0.01):
    signals = (predictions > thresh).astype(int)
    trades = signals.sum()
    
    if trades > 0:
        actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
        wins = ((signals == 1) & (actual_returns > 0)).sum()
        win_rate = wins / trades
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_thresh = thresh

print("\n" + "="*60)
print(f"🏆 RESULTS WITH SENTIMENT")
print(f"Win Rate: {best_win_rate:.4f}")
print(f"Threshold: {best_thresh}")
print(f"Trades: {(predictions > best_thresh).sum()}")
print("="*60)

# Save model
joblib.dump({
    'models': models,
    'threshold': best_thresh,
    'features': feature_cols,
    'win_rate': best_win_rate
}, 'data/models/yasen_alpha_sentiment.joblib')
print("\n💾 Model with sentiment saved!")

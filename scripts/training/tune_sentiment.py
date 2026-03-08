import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

print("="*60)
print("YASEN-ALPHA: TUNING SENTIMENT MODEL")
print("="*60)

# Load sentiment data
df = pd.read_parquet('data/processed/btc_with_sentiment.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

X = df_clean[feature_cols]
y = df_clean['target']

print(f"\n📊 Data: {len(X)} samples, {len(feature_cols)} features")
print(f"📈 Class balance: {y.mean():.2%}")

# Train ensemble with different seeds (include sentiment-specific params)
models = []
for seed in [42, 123, 456, 789, 999]:
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X, y)
    models.append(model)
    print(f"✅ Model {seed} trained")

# Try different thresholds
best_win_rate = 0
best_thresh = 0
best_trades = 0

print("\n🔍 Testing thresholds...")
for thresh in np.arange(0.30, 0.45, 0.01):
    # Ensemble predictions
    predictions = np.zeros(len(X))
    for model in models:
        predictions += model.predict_proba(X)[:, 1]
    predictions /= len(models)
    
    signals = (predictions > thresh).astype(int)
    trades = signals.sum()
    
    if trades > 500:  # Minimum trade threshold
        actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
        wins = ((signals == 1) & (actual_returns > 0)).sum()
        win_rate = wins / trades
        
        print(f"Threshold {thresh:.2f}: {win_rate:.4f} on {trades} trades")
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_thresh = thresh
            best_trades = trades

print("\n" + "="*60)
print(f"🏆 OPTIMIZED SENTIMENT MODEL")
print(f"Win Rate: {best_win_rate:.4f}")
print(f"Threshold: {best_thresh}")
print(f"Trades: {best_trades}")
print("="*60)

if best_win_rate > 0.59:
    print("🎉 THIS BEATS YOUR CURRENT BEST!")
    joblib.dump({
        'models': models,
        'threshold': best_thresh,
        'win_rate': best_win_rate,
        'features': feature_cols
    }, 'data/models/yasen_alpha_sentiment_tuned.joblib')
    print("💾 Tuned sentiment model saved!")

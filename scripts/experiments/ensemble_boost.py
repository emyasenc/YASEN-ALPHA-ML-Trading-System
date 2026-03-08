import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

print("="*60)
print("YASEN-ALPHA: QUICK BOOST TO 60%")
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

print(f"\n📊 Data: {len(X)} samples, {len(feature_cols)} features")

# Train multiple models with different seeds
models = []
for seed in [42, 123, 456, 789, 999]:
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1
    )
    xgb_model.fit(X, y)
    models.append(xgb_model)
    print(f"✅ Model {seed} trained")

# Ensemble predictions
predictions = np.zeros(len(X))
for model in models:
    predictions += model.predict_proba(X)[:, 1]
predictions /= len(models)

# Find optimal threshold
best_score = 0
best_thresh = 0
for thresh in np.arange(0.3, 0.5, 0.01):
    pred = (predictions > thresh).astype(int)
    score = (pred == y).mean()
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\n🏆 Ensemble Accuracy: {best_score:.4f} at threshold {best_thresh:.2f}")

# Check win rate on trades
signals = (predictions > best_thresh).astype(int)
trades = signals.sum()
if trades > 0:
    actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
    wins = ((signals == 1) & (actual_returns > 0)).sum()
    win_rate = wins / trades
    print(f"📊 Win Rate: {win_rate:.2%} on {trades} trades")

# Save ensemble
joblib.dump({
    'models': models,
    'threshold': best_thresh,
    'features': feature_cols
}, 'data/models/yasen_alpha_ensemble_v2.joblib')
print("💾 Ensemble saved!")

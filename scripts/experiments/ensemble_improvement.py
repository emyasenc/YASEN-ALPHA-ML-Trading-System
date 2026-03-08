import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

print("="*60)
print("YASEN-ALPHA: ENSEMBLE IMPROVEMENT")
print("="*60)

# Load data
df = pd.read_parquet('data/processed/btc_with_features.parquet')
exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].dropna()

# Create target (same as before)
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df = df.dropna()
y = df.loc[X.index, 'target']

print(f"\n📊 Training with {len(X)} samples, {len(feature_cols)} features")

# Train multiple models
print("\n🤖 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X, y)
xgb_pred = xgb_model.predict_proba(X)[:, 1]

print("🤖 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X, y)
lgb_pred = lgb_model.predict_proba(X)[:, 1]

print("🤖 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X, y)
rf_pred = rf_model.predict_proba(X)[:, 1]

# Ensemble predictions
ensemble_pred = (xgb_pred + lgb_pred + rf_pred) / 3

# Evaluate each at threshold 0.34
threshold = 0.34
models = {
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'Random Forest': rf_pred,
    'Ensemble': ensemble_pred
}

print("\n" + "="*60)
print("📊 MODEL COMPARISON (Threshold 0.34)")
print("="*60)

for name, pred in models.items():
    signals = (pred > threshold).astype(int)
    trades = signals.sum()
    if trades > 0:
        actual_returns = df.loc[X.index, 'close'].pct_change().shift(-1).fillna(0)
        wins = ((signals == 1) & (actual_returns > 0)).sum()
        win_rate = wins / trades if trades > 0 else 0
        
        # Simple return calculation
        strategy_returns = signals * actual_returns
        total_return = (1 + strategy_returns).prod() - 1
        
        print(f"\n{name}:")
        print(f"  Trades: {trades}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Return: {total_return:.2%}")
        
        # Save if better than current
        if win_rate > 0.5572:  # Current best
            print(f"  🎯 BEATS CURRENT MODEL!")
            
            if name == 'Ensemble':
                joblib.dump({
                    'xgb': xgb_model,
                    'lgb': lgb_model,
                    'rf': rf_model
                }, 'data/models/yasen_alpha_ensemble.joblib')
                print("  💾 Ensemble saved!")

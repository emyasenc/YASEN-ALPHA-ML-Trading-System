import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

print("="*60)
print("YASEN-ALPHA: ENSEMBLE IMPROVEMENT (FIXED)")
print("="*60)

# Load data
df = pd.read_parquet('data/processed/btc_with_features.parquet')
print(f"\n📊 Original data shape: {df.shape}")

# Create target first
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)

# Drop NaN values
df_clean = df.dropna()
print(f"📊 Clean data shape: {df_clean.shape}")

# Prepare features
exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

X = df_clean[feature_cols]
y = df_clean['target']

print(f"\n📊 Training with {len(X)} samples, {len(feature_cols)} features")
print(f"📈 Class balance: {y.mean():.2%} positive")

# Train multiple models
print("\n🤖 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X, y)
xgb_pred = xgb_model.predict_proba(X)[:, 1]

print("🤖 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1
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

# Calculate actual returns for evaluation
actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)

# Evaluate each at different thresholds
thresholds = [0.32, 0.33, 0.34, 0.35]

print("\n" + "="*60)
print("📊 MODEL COMPARISON")
print("="*60)

models = {
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'Random Forest': rf_pred,
    'Ensemble': ensemble_pred
}

best_win_rate = 0
best_model_name = ""
best_threshold = 0

for threshold in thresholds:
    print(f"\n{'─'*60}")
    print(f"📈 THRESHOLD: {threshold}")
    print(f"{'─'*60}")
    
    for name, pred in models.items():
        signals = (pred > threshold).astype(int)
        trades = signals.sum()
        
        if trades > 0:
            wins = ((signals == 1) & (actual_returns > 0)).sum()
            win_rate = wins / trades
            
            # Simple return calculation
            strategy_returns = signals * actual_returns
            total_return = (1 + strategy_returns).prod() - 1
            
            print(f"\n{name}:")
            print(f"  Trades: {trades}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Return: {total_return:.2%}")
            
            # Track best
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_model_name = name
                best_threshold = threshold
        else:
            print(f"\n{name}: No trades")

# Save the best model
print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name} at threshold {best_threshold} with {best_win_rate:.2%} win rate")
print("="*60)

if best_model_name == 'Ensemble':
    joblib.dump({
        'xgb': xgb_model,
        'lgb': lgb_model,
        'rf': rf_model,
        'threshold': best_threshold
    }, 'data/models/yasen_alpha_ensemble.joblib')
    print("💾 Ensemble model saved to data/models/yasen_alpha_ensemble.joblib")
elif best_model_name == 'XGBoost':
    joblib.dump(xgb_model, 'data/models/yasen_alpha_xgb_optimized.joblib')
    print("💾 XGBoost model saved to data/models/yasen_alpha_xgb_optimized.joblib")
elif best_model_name == 'LightGBM':
    joblib.dump(lgb_model, 'data/models/yasen_alpha_lgb_optimized.joblib')
    print("💾 LightGBM model saved to data/models/yasen_alpha_lgb_optimized.joblib")
elif best_model_name == 'Random Forest':
    joblib.dump(rf_model, 'data/models/yasen_alpha_rf_optimized.joblib')
    print("💾 Random Forest model saved to data/models/yasen_alpha_rf_optimized.joblib")

# Feature importance comparison
print("\n" + "="*60)
print("📊 TOP FEATURES BY MODEL")
print("="*60)

# XGBoost importance
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print("\n🔝 XGBoost Top Features:")
print(xgb_importance.to_string(index=False))

# Save comparison
comparison = []
for threshold in thresholds:
    for name, pred in models.items():
        signals = (pred > threshold).astype(int)
        trades = signals.sum()
        if trades > 0:
            wins = ((signals == 1) & (actual_returns > 0)).sum()
            win_rate = wins / trades
            strategy_returns = signals * actual_returns
            total_return = (1 + strategy_returns).prod() - 1
            
            comparison.append({
                'model': name,
                'threshold': threshold,
                'trades': trades,
                'win_rate': win_rate,
                'return': total_return
            })

pd.DataFrame(comparison).to_csv('data/ensemble_comparison.csv', index=False)
print("\n💾 Comparison saved to data/ensemble_comparison.csv")

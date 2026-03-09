#!/usr/bin/env python3
"""
Walk-forward validation to test model stability
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from src.backtesting.backtest import Backtester

print("="*60)
print("WALK-FORWARD VALIDATION")
print("="*60)

# Load data
df = pd.read_parquet('data/processed/features_latest.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df = df.dropna()

exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['target']

# Load new model params
new_model_data = joblib.load('data/models/yasen_alpha_optimized_fast.joblib')
params = new_model_data['params']

# Walk-forward test
tscv = TimeSeriesSplit(n_splits=5)
walk_forward_scores = []

print("\n🔄 Running walk-forward validation...")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Test accuracy
    pred = model.predict(X_test)
    acc = (pred == y_test).mean()
    walk_forward_scores.append(acc)
    print(f"Fold {fold}: {acc:.4f}")

print(f"\n📊 Walk-forward average: {np.mean(walk_forward_scores):.4f}")
print(f"📊 Walk-forward std: {np.std(walk_forward_scores):.4f}")
print(f"📊 Min: {min(walk_forward_scores):.4f}")
print(f"📊 Max: {max(walk_forward_scores):.4f}")

# Compare to original model parameters
print("\n" + "="*60)
print("✅ RECOMMENDATION:")
if np.mean(walk_forward_scores) > 0.60:
    print("New model is STABLE (>60% average) - USE IT!")
elif np.mean(walk_forward_scores) > 0.59:
    print("New model is SIMILAR to old - Your choice")
else:
    print("New model may be OVERFIT - Keep old model")
print("="*60)

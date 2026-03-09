#!/usr/bin/env python3
"""
Walk-forward validation to compare models realistically
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("WALK-FORWARD VALIDATION")
print("="*60)

# Load models
old_model_data = joblib.load('data/models/yasen_alpha_champion.joblib')
new_model_data = joblib.load('data/models/yasen_alpha_optimized_fast.joblib')

# Extract models
def extract_model(model_data):
    if isinstance(model_data, dict):
        if 'model' in model_data:
            return model_data['model']
        elif 'models' in model_data:
            return model_data['models'][0]  # Use first model
    return model_data

old_model = extract_model(old_model_data)
new_model = extract_model(new_model_data)

# Load and prepare data
df = pd.read_parquet('data/processed/features_latest.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df = df.dropna()

exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['target']

# Walk-forward parameters
window_size = 5000  # Train on 5000 samples
step_size = 1000    # Test 1000 samples at a time

print(f"\n📊 Total data: {len(X)} samples")
print(f"Training window: {window_size} samples")
print(f"Test step: {step_size} samples")

old_scores = []
new_scores = []
positions = []

for start in range(0, len(X) - window_size - step_size, step_size):
    train_end = start + window_size
    test_end = train_end + step_size
    
    X_train = X.iloc[start:train_end]
    y_train = y.iloc[start:train_end]
    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]
    
    # Test old model
    old_pred = old_model.predict(X_test)
    old_acc = accuracy_score(y_test, old_pred)
    old_scores.append(old_acc)
    
    # Test new model
    new_pred = new_model.predict(X_test)
    new_acc = accuracy_score(y_test, new_pred)
    new_scores.append(new_acc)
    
    positions.append(f"{train_end}-{test_end}")
    
    print(f"\n📈 Fold {len(old_scores)}: {train_end}-{test_end}")
    print(f"   Old model: {old_acc:.4f}")
    print(f"   New model: {new_acc:.4f}")
    print(f"   Difference: {new_acc-old_acc:+.4f}")

print("\n" + "="*60)
print("📊 WALK-FORWARD RESULTS")
print("="*60)
print(f"{'Fold':<10} {'Old':<10} {'New':<10} {'Diff':<10}")
print("-"*40)

for i, (old, new) in enumerate(zip(old_scores, new_scores)):
    print(f"{i+1:<10} {old:.4f}    {new:.4f}    {new-old:+.4f}")

print("-"*40)
print(f"{'AVG':<10} {np.mean(old_scores):.4f}    {np.mean(new_scores):.4f}    {np.mean(new_scores)-np.mean(old_scores):+.4f}")
print(f"{'STD':<10} {np.std(old_scores):.4f}    {np.std(new_scores):.4f}")
print("="*60)

if np.mean(new_scores) > np.mean(old_scores):
    print("\n✅ NEW MODEL WINS in walk-forward!")
else:
    print("\n✅ OLD MODEL WINS in walk-forward!")
print("="*60)

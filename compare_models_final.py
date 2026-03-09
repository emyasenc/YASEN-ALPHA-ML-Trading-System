import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Load models
old_model_data = joblib.load('data/models/yasen_alpha_champion.joblib')
new_model_data = joblib.load('data/models/yasen_alpha_optimized_fast.joblib')

print(f"✅ Old model loaded")
print(f"✅ New model loaded with accuracy: {new_model_data['accuracy']:.2%}")

# Load test data
df = pd.read_parquet('data/processed/features_latest.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df = df.dropna()

exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['target']

# Split data (use last 30% for testing)
split = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\n📊 Test set size: {len(X_test)} samples")

# Function to extract model from different formats
def extract_model(model_data):
    if isinstance(model_data, dict):
        if 'model' in model_data:
            return model_data['model']
        elif 'models' in model_data:
            # Use the first model for simplicity
            print(f"   (Using first of {len(model_data['models'])} ensemble models)")
            return model_data['models'][0]
        else:
            return model_data
    else:
        return model_data

# Get models
old_model = extract_model(old_model_data)
new_model = extract_model(new_model_data)

# Test old model
old_pred = old_model.predict(X_test)
old_acc = accuracy_score(y_test, old_pred)

# Calculate additional metrics for old model
old_proba = old_model.predict_proba(X_test)[:, 1]
old_signals = (old_proba > 0.47).astype(int)
old_trades = old_signals.sum()
old_wins = ((old_signals == 1) & (y_test == 1)).sum()
old_win_rate = old_wins / old_trades if old_trades > 0 else 0

# Test new model
new_pred = new_model.predict(X_test)
new_acc = accuracy_score(y_test, new_pred)

# Calculate additional metrics for new model
new_proba = new_model.predict_proba(X_test)[:, 1]
new_signals = (new_proba > 0.47).astype(int)
new_trades = new_signals.sum()
new_wins = ((new_signals == 1) & (y_test == 1)).sum()
new_win_rate = new_wins / new_trades if new_trades > 0 else 0

print("\n" + "="*60)
print("📈 COMPARISON RESULTS")
print("="*60)
print(f"{'Metric':<20} {'OLD MODEL':<15} {'NEW MODEL':<15} {'IMPROVEMENT':<15}")
print("-"*60)
print(f"{'Accuracy':<20} {old_acc:.2%}        {new_acc:.2%}        {new_acc-old_acc:+.2%}")
print(f"{'Win Rate':<20} {old_win_rate:.2%}        {new_win_rate:.2%}        {new_win_rate-old_win_rate:+.2%}")
print(f"{'Total Trades':<20} {old_trades:<14} {new_trades:<14} {new_trades-old_trades:<+14}")
print("="*60)

# Recommendation
print("\n✅ RECOMMENDATION:")
if new_win_rate > old_win_rate:
    print(f"   NEW MODEL WINS! ({new_win_rate:.2%} vs {old_win_rate:.2%})")
    print("   Run this command to make it champion:")
    print("   cp data/models/yasen_alpha_optimized_fast.joblib data/models/yasen_alpha_champion.joblib")
else:
    print(f"   OLD MODEL WINS! ({old_win_rate:.2%} vs {new_win_rate:.2%})")
    print("   Keep current champion")
print("="*60)

# Show best parameters from new model
print("\n📊 Best parameters from new model:")
for key, value in new_model_data['params'].items():
    print(f"  {key}: {value}")
print("="*60)

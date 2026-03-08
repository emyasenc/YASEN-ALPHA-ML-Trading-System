import pandas as pd
import numpy as np
from src.models.train import BitcoinPredictor

print("="*60)
print("YASEN-ALPHA: TRAINING FIRST MODEL")
print("="*60)

# Load features
print("\n📂 Loading feature dataset...")
df = pd.read_parquet('data/processed/btc_with_features.parquet')
print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

# Initialize predictor
predictor = BitcoinPredictor()

# Create target (predict 24h ahead, 2% threshold)
print("\n🎯 Creating target variable...")
df = predictor.create_target(df, horizon=24, threshold_pct=0.02)

# Get feature columns (exclude target and price columns)
exclude_cols = ['target', 'open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\n📊 Using {len(feature_cols)} features")

# Prepare data
X, y = predictor.prepare_features_target(df, feature_cols)

print(f"\n🔢 Training data shape: {X.shape}")
print(f"📊 Class balance: {y.mean():.2%} positive")

# Train model
print("\n🤖 Training XGBoost model (this will take 5-10 minutes)...")
results = predictor.train_xgboost(X, y)

# Save model
model_path = predictor.save_model('yasen_alpha_v1')
print(f"\n💾 Model saved to {model_path}")

# Print final results
print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print(f"🎯 Cross-validation accuracy: {results['mean_accuracy']:.2%} (+/- {results['std_accuracy']:.2%})")
print(f"📈 Best fold: {max(results['cv_scores']):.2%}")
print("="*60)

# Quick feature importance
import matplotlib.pyplot as plt
import xgboost as xgb

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': results['model'].feature_importances_
}).sort_values('importance', ascending=False).head(20)

print("\n🔝 Top 10 Most Important Features:")
for i, row in importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save importance to file
importance.to_csv('data/feature_importance.csv', index=False)
print("\n💾 Feature importance saved to data/feature_importance.csv")

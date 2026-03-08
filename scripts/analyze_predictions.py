import pandas as pd
import numpy as np
import joblib

print("="*60)
print("YASEN-ALPHA: PREDICTION ANALYSIS")
print("="*60)

# Load data and model
df = pd.read_parquet('data/processed/btc_with_features.parquet')
model = joblib.load('data/models/yasen_alpha_v1.joblib')

# Prepare features
exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].dropna()

# Get predictions
probabilities = model.predict_proba(X)[:, 1]

print(f"\n📊 Prediction Statistics:")
print(f"Min probability: {probabilities.min():.4f}")
print(f"Max probability: {probabilities.max():.4f}")
print(f"Mean probability: {probabilities.mean():.4f}")
print(f"Median probability: {np.median(probabilities):.4f}")

# Distribution by percentile
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
print(f"\n📈 Percentile Distribution:")
for p in percentiles:
    val = np.percentile(probabilities, p)
    print(f"  {p}th percentile: {val:.4f}")

# Find optimal threshold
thresholds = np.arange(0.3, 0.7, 0.02)
print(f"\n🎯 Trade Count by Threshold:")
for thresh in thresholds:
    trades = (probabilities > thresh).sum()
    pct = trades / len(probabilities) * 100
    print(f"  Threshold {thresh:.2f}: {trades} trades ({pct:.1f}% of periods)")

# Save analysis
results = pd.DataFrame({
    'probability': probabilities,
    'date': X.index
})
results.to_csv('data/prediction_analysis.csv', index=False)
print("\n💾 Detailed predictions saved to data/prediction_analysis.csv")

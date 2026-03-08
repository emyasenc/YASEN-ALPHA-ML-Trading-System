import pandas as pd
import numpy as np
import joblib

print("="*60)
print("YASEN-ALPHA: ULTIMATE ENSEMBLE")
print("="*60)

# Load both best models
weights_model = joblib.load('data/models/yasen_alpha_optimized_weights.joblib')
sentiment_model = joblib.load('data/models/yasen_alpha_sentiment.joblib')

# Load data
df = pd.read_parquet('data/processed/btc_with_sentiment.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

# Get features for both models
feature_cols = [col for col in df_clean.columns if col not in 
                ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']]
X = df_clean[feature_cols]

# Get predictions from weights model (5 models)
weights_pred = np.zeros(len(X))
for i, model in enumerate(weights_model['models']):
    weights_pred += weights_model['weights'][i] * model.predict_proba(X)[:, 1]

# Get predictions from sentiment model (5 models)
sentiment_pred = np.zeros(len(X))
for model in sentiment_model['models']:
    sentiment_pred += model.predict_proba(X)[:, 1]
sentiment_pred /= len(sentiment_model['models'])

# Try different combinations
print("\n🔍 Testing model combinations...")
print("-" * 60)

best_win_rate = 0
best_combo = None
best_thresh = 0

for w1 in np.arange(0.3, 0.8, 0.1):
    w2 = 1 - w1
    
    combined_pred = w1 * weights_pred + w2 * sentiment_pred
    
    for thresh in np.arange(0.35, 0.48, 0.01):
        signals = (combined_pred > thresh).astype(int)
        trades = signals.sum()
        
        if trades > 3000:
            actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
            wins = ((signals == 1) & (actual_returns > 0)).sum()
            win_rate = wins / trades
            
            print(f"Weight {w1:.1f}/{w2:.1f} @ {thresh:.2f}: {win_rate:.4f} on {trades} trades")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_combo = (w1, w2)
                best_thresh = thresh

print("\n" + "="*60)
print(f"🏆 ULTIMATE ENSEMBLE RESULT")
print(f"Win Rate: {best_win_rate:.4f}")
print(f"Combination: {best_combo[0]:.1f} weights + {best_combo[1]:.1f} sentiment")
print(f"Threshold: {best_thresh}")
print("="*60)

if best_win_rate > 0.5903:
    print("🎉 NEW RECORD! This beats your previous best!")
    
    # Calculate final predictions
    final_pred = best_combo[0] * weights_pred + best_combo[1] * sentiment_pred
    signals = (final_pred > best_thresh).astype(int)
    
    # Save
    joblib.dump({
        'weights_model': weights_model,
        'sentiment_model': sentiment_model,
        'weights': best_combo,
        'threshold': best_thresh,
        'win_rate': best_win_rate
    }, 'data/models/yasen_alpha_ultimate.joblib')
    print("💾 Ultimate ensemble saved!")

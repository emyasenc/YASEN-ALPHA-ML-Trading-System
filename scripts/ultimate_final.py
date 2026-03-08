import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

print("="*60)
print("YASEN-ALPHA: ULTIMATE FINAL ENSEMBLE")
print("="*60)

# Load ALL your best models
print("\n📂 Loading models...")
weights_model = joblib.load('data/models/yasen_alpha_optimized_weights.joblib')
sentiment_model = joblib.load('data/models/yasen_alpha_sentiment_tuned.joblib')
dynamic_config = {'low': 1.0, 'med': 1.2, 'high': 1.2}  # From dynamic threshold

# Load data with sentiment
df = pd.read_parquet('data/processed/btc_with_sentiment.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

# Calculate volatility
df_clean['volatility'] = df_clean['close'].pct_change().rolling(24).std()
vol_low = df_clean['volatility'].quantile(0.33)
vol_med = df_clean['volatility'].quantile(0.66)

print(f"\n📊 Volatility thresholds:")
print(f"Low: < {vol_low:.4f}")
print(f"Medium: {vol_low:.4f} - {vol_med:.4f}")
print(f"High: > {vol_med:.4f}")

# Prepare features - use sentiment features (they include everything)
feature_cols = [col for col in df_clean.columns if col not in 
                ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']]
X = df_clean[feature_cols]

print(f"\n📊 Using {len(feature_cols)} features")

# Get predictions from weights model (5 models)
print("\n🤖 Getting predictions from weights model...")
weights_pred = np.zeros(len(X))
for i, model in enumerate(weights_model['models']):
    # Ensure model uses correct features
    weights_pred += weights_model['weights'][i] * model.predict_proba(X[weights_model['features']])[:, 1]

# Get predictions from sentiment model (5 models)
print("🤖 Getting predictions from sentiment model...")
sentiment_pred = np.zeros(len(X))
for model in sentiment_model['models']:
    sentiment_pred += model.predict_proba(X[sentiment_model['features']])[:, 1]
sentiment_pred /= len(sentiment_model['models'])

# Try different combinations with dynamic threshold
print("\n🔍 Testing ultimate combinations...")
print("-" * 70)
print(f"{'W1':>5} {'W2':>5} {'Thresh':>7} {'Trades':>8} {'Win Rate':>10} {'Return':>12}")
print("-" * 70)

best_win_rate = 0
best_config = None

for w1 in np.arange(0.3, 0.8, 0.1):
    w2 = 1 - w1
    
    # Combined prediction
    combined_pred = w1 * weights_pred + w2 * sentiment_pred
    
    # Dynamic threshold
    base_threshold = 0.41  # From your best model
    
    for thresh_mult in np.arange(0.8, 1.3, 0.1):
        signals = np.zeros(len(X))
        
        for i in range(len(X)):
            vol = df_clean['volatility'].iloc[i]
            if pd.isna(vol):
                continue
                
            if vol <= vol_low:
                threshold = base_threshold * dynamic_config['low'] * thresh_mult
            elif vol <= vol_med:
                threshold = base_threshold * dynamic_config['med'] * thresh_mult
            else:
                threshold = base_threshold * dynamic_config['high'] * thresh_mult
                
            signals[i] = combined_pred[i] > threshold
        
        trades = signals.sum()
        if trades > 3000:
            actual_returns = df_clean['close'].pct_change().shift(-1).fillna(0)
            wins = ((signals == 1) & (actual_returns > 0)).sum()
            win_rate = wins / trades
            
            # Simple return calculation
            strategy_returns = signals * actual_returns
            total_return = (1 + strategy_returns).prod() - 1
            
            print(f"{w1:5.1f} {w2:5.1f} {base_threshold*thresh_mult:7.3f} {trades:8.0f} {win_rate:10.4f} {total_return:12.2f}")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_config = {
                    'w1': w1,
                    'w2': w2,
                    'thresh_mult': thresh_mult,
                    'trades': trades,
                    'return': total_return
                }

print("\n" + "="*70)
print("🏆 ULTIMATE FINAL RESULT")
print("="*70)
print(f"Win Rate: {best_win_rate:.4f}")
print(f"Configuration:")
print(f"  Weight1 (weights model): {best_config['w1']:.1f}")
print(f"  Weight2 (sentiment model): {best_config['w2']:.1f}")
print(f"  Threshold multiplier: {best_config['thresh_mult']:.1f}")
print(f"  Trades: {best_config['trades']}")
print(f"  Return: {best_config['return']:.2f}%")
print("="*70)

if best_win_rate > 0.5919:
    print("\n🎉 NEW RECORD! This beats your dynamic threshold model!")
    
    # Save the ultimate model
    ultimate_model = {
        'weights_model': weights_model,
        'sentiment_model': sentiment_model,
        'dynamic_config': dynamic_config,
        'best_config': best_config,
        'win_rate': best_win_rate,
        'volatility_thresholds': {'low': vol_low, 'med': vol_med}
    }
    joblib.dump(ultimate_model, 'data/models/yasen_alpha_ultimate_final.joblib')
    print("💾 Ultimate model saved to data/models/yasen_alpha_ultimate_final.joblib")
else:
    print("\n📊 Dynamic threshold model remains champion at 59.19%")

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime

print("="*60)
print("YASEN-ALPHA: SAVING CHAMPION MODEL")
print("="*60)

# Load your best weights model
weights_model = joblib.load('data/models/yasen_alpha_optimized_weights.joblib')

# Load data for reference
df = pd.read_parquet('data/processed/btc_with_features.parquet')
df['future_close'] = df['close'].shift(-24)
df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
df_clean = df.dropna()

# Calculate volatility thresholds
df_clean['volatility'] = df_clean['close'].pct_change().rolling(24).std()
vol_low = df_clean['volatility'].quantile(0.33)
vol_med = df_clean['volatility'].quantile(0.66)

# Champion configuration
champion = {
    'name': 'YASEN-ALPHA Dynamic Threshold v1.0',
    'win_rate': 0.5919,
    'models': weights_model['models'],
    'weights': weights_model['weights'],
    'base_threshold': 0.41,  # From your optimal configuration
    'dynamic_multipliers': {
        'low': 1.0,
        'medium': 1.2,
        'high': 1.2
    },
    'volatility_thresholds': {
        'low': vol_low,
        'medium': vol_med
    },
    'features': weights_model['features'],
    'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'version': '1.0'
}

# Save champion
joblib.dump(champion, 'data/models/yasen_alpha_champion.joblib')
print(f"\n🏆 Champion model saved with {champion['win_rate']:.2%} win rate!")
print(f"📊 Configuration:")
print(f"  - Base threshold: {champion['base_threshold']}")
print(f"  - Low volatility: ×{champion['dynamic_multipliers']['low']}")
print(f"  - Medium volatility: ×{champion['dynamic_multipliers']['medium']}")
print(f"  - High volatility: ×{champion['dynamic_multipliers']['high']}")
print(f"  - Features: {len(champion['features'])}")
print(f"\n💾 Saved to: data/models/yasen_alpha_champion.joblib")

# Create a simple prediction function for later use
print("\n📝 Creating prediction function...")

prediction_code = """
import numpy as np
import pandas as pd
import joblib

class YasenAlphaPredictor:
    def __init__(self, model_path='data/models/yasen_alpha_champion.joblib'):
        self.champion = joblib.load(model_path)
        self.models = self.champion['models']
        self.weights = self.champion['weights']
        self.base_threshold = self.champion['base_threshold']
        self.multipliers = self.champion['dynamic_multipliers']
        self.vol_thresholds = self.champion['volatility_thresholds']
        self.features = self.champion['features']
        
    def predict(self, df):
        # Calculate volatility
        df = df.copy()
        df['volatility'] = df['close'].pct_change().rolling(24).std()
        
        # Get model predictions
        ensemble_pred = np.zeros(len(df))
        for i, model in enumerate(self.models):
            ensemble_pred += self.weights[i] * model.predict_proba(df[self.features])[:, 1]
        
        # Apply dynamic threshold
        signals = np.zeros(len(df))
        for i in range(len(df)):
            vol = df['volatility'].iloc[i]
            if pd.isna(vol):
                continue
                
            if vol <= self.vol_thresholds['low']:
                threshold = self.base_threshold * self.multipliers['low']
            elif vol <= self.vol_thresholds['medium']:
                threshold = self.base_threshold * self.multipliers['medium']
            else:
                threshold = self.base_threshold * self.multipliers['high']
            
            signals[i] = ensemble_pred[i] > threshold
        
        return {
            'probabilities': ensemble_pred,
            'signals': signals,
            'threshold_used': threshold
        }
"""

with open('src/predictor.py', 'w') as f:
    f.write(prediction_code)

print("✅ Prediction class created at src/predictor.py")

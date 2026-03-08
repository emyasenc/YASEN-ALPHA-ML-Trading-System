
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

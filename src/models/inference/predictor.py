"""
Real-time Prediction Module
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YasenAlphaPredictor:
    def __init__(self, model_path='data/models/yasen_alpha_champion.joblib'):
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model with {self.model.get('win_rate', 0):.2%} win rate")
        
    def get_current_signal(self):
        """Get current trading signal"""
        try:
            # Load latest data
            df = pd.read_parquet('data/processed/features_latest.parquet')
            latest = df.iloc[-1:].copy()
            
            # Get predictions
            X = latest[self.model['features']]
            
            predictions = np.zeros(len(X))
            for model in self.model['models']:
                predictions += model.predict_proba(X)[:, 1]
            predictions /= len(self.model['models'])
            
            # Calculate volatility
            volatility = df['close'].pct_change().rolling(24).std().iloc[-1]
            
            # Apply dynamic threshold
            threshold = self.model['threshold']
            if volatility < 0.0046:
                threshold *= 1.0
            elif volatility < 0.0081:
                threshold *= 1.2
            else:
                threshold *= 1.2
            
            signal = "BUY" if predictions[0] > threshold else "HOLD"
            
            return {
                'signal': signal,
                'confidence': float(predictions[0]),
                'threshold_used': float(threshold),
                'volatility': float(volatility) if not pd.isna(volatility) else None
            }
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            return {'signal': 'ERROR', 'error': str(e)}
    
    def get_historical_signals(self, days=30):
        """Get historical signals for dashboard"""
        df = pd.read_parquet('data/processed/features_latest.parquet')
        df = df.tail(days * 24).copy()
        
        X = df[self.model['features']]
        
        predictions = np.zeros(len(X))
        for model in self.model['models']:
            predictions += model.predict_proba(X)[:, 1]
        predictions /= len(self.model['models'])
        
        signals = (predictions > self.model['threshold']).astype(int)
        
        return pd.DataFrame({
            'timestamp': df.index,
            'price': df['close'],
            'confidence': predictions,
            'signal': signals
        })

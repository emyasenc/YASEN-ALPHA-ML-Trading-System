"""
Real-time Prediction Module for YASEN-ALPHA
Handles both old champion model and new advanced model
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class YasenAlphaPredictor:
    """
    Unified predictor that works with both model formats
    """
    
    def __init__(self, model_path='data/models/yasen_alpha_champion.joblib'):
        self.model_data = joblib.load(model_path)
        self.model_type = self._detect_model_type()
        self.threshold = self._get_threshold()
        self.features = self._get_features()
        self.win_rate = self._get_win_rate()
        
        logger.info(f"Loaded model with {self.model_type} format")
        logger.info(f"Win rate: {self.win_rate:.2%}")
        logger.info(f"Threshold: {self.threshold:.2f}")
    
    def _detect_model_type(self):
        """Detect if model is old champion or new advanced"""
        if 'models' in self.model_data and isinstance(self.model_data['models'], dict):
            return 'advanced'
        elif 'models' in self.model_data and isinstance(self.model_data['models'], list):
            return 'champion'
        else:
            return 'simple'
    
    def _get_threshold(self):
        """Get threshold from model data"""
        if self.model_type == 'advanced':
            thresholds = self.model_data.get('thresholds', {})
            if thresholds:
                return float(np.mean(list(thresholds.values())))
            return 0.45
        else:
            return float(self.model_data.get('threshold', 0.45))
    
    def _get_features(self):
        """Get feature list from model"""
        return self.model_data.get('features', [])
    
    def _get_win_rate(self):
        """Get win rate from model"""
        if self.model_type == 'advanced':
            return float(self.model_data.get('metadata', {}).get('win_rate', 0.589))
        else:
            return float(self.model_data.get('win_rate', 0.589))
    
    def get_current_signal(self) -> Dict[str, Any]:
        """Get current trading signal"""
        try:
            # Load latest data
            df = pd.read_parquet('data/processed/features_latest.parquet')
            latest = df.iloc[-1:].copy()
            
            if len(latest) == 0:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'threshold_used': self.threshold,
                    'volatility': 0.0
                }
            
            # Prepare features
            if self.features:
                feature_cols = [col for col in self.features if col in latest.columns]
            else:
                exclude_cols = ['open', 'high', 'low', 'close', 'volume']
                feature_cols = [col for col in latest.columns if col not in exclude_cols]
            
            X = latest[feature_cols].fillna(0)
            
            # Get prediction based on model type
            if self.model_type == 'advanced':
                # Advanced model has regime-specific models
                regime_models = self.model_data.get('models', {})
                if regime_models:
                    predictions = []
                    for regime, model in regime_models.items():
                        try:
                            prob = model.predict_proba(X)[0][1]
                            predictions.append(prob)
                        except Exception as e:
                            logger.warning(f"Failed to predict with regime {regime}: {e}")
                            continue
                    confidence = float(np.mean(predictions)) if predictions else 0.5
                else:
                    confidence = 0.5
            else:
                # Champion model (ensemble of models in list)
                models = self.model_data.get('models', [])
                weights = self.model_data.get('weights', [1/len(models)] * len(models)) if models else []
                if models:
                    predictions = []
                    for i, model in enumerate(models):
                        try:
                            prob = model.predict_proba(X)[0][1]
                            weight = weights[i] if i < len(weights) else 1/len(models)
                            predictions.append(prob * weight)
                        except Exception as e:
                            logger.warning(f"Failed to predict with model {i}: {e}")
                            continue
                    confidence = float(np.sum(predictions)) if predictions else 0.5
                else:
                    confidence = 0.5
            
            # Determine signal
            signal = "BUY" if confidence > self.threshold else "HOLD"
            
            # Calculate volatility
            volatility = float(df['close'].pct_change().rolling(24).std().iloc[-1])
            if pd.isna(volatility) or volatility == 0:
                volatility = 0.01
            
            return {
                'signal': signal,
                'confidence': confidence,
                'threshold_used': self.threshold,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            import traceback
            traceback.print_exc()
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'threshold_used': self.threshold,
                'volatility': 0.0
            }
    
    def get_historical_signals(self, days=30):
        """Get historical signals for dashboard"""
        try:
            df = pd.read_parquet('data/processed/features_latest.parquet')
            df = df.tail(days * 24).copy()
            
            # Prepare features
            if self.features:
                feature_cols = [col for col in self.features if col in df.columns]
            else:
                exclude_cols = ['open', 'high', 'low', 'close', 'volume']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols].fillna(0)
            
            # Get predictions based on model type
            if self.model_type == 'advanced':
                regime_models = self.model_data.get('models', {})
                if regime_models:
                    all_predictions = []
                    for regime, model in regime_models.items():
                        try:
                            prob = model.predict_proba(X)[:, 1]
                            all_predictions.append(prob)
                        except:
                            continue
                    predictions = np.mean(all_predictions, axis=0) if all_predictions else np.zeros(len(df))
                else:
                    predictions = np.zeros(len(df))
            else:
                models = self.model_data.get('models', [])
                weights = self.model_data.get('weights', [1/len(models)] * len(models)) if models else []
                if models:
                    all_predictions = []
                    for i, model in enumerate(models):
                        try:
                            prob = model.predict_proba(X)[:, 1]
                            weight = weights[i] if i < len(weights) else 1/len(models)
                            all_predictions.append(prob * weight)
                        except:
                            continue
                    predictions = np.sum(all_predictions, axis=0) if all_predictions else np.zeros(len(df))
                else:
                    predictions = np.zeros(len(df))
            
            signals = (predictions > self.threshold).astype(int)
            
            return {
                'timestamps': df.index.tolist(),
                'prices': df['close'].tolist(),
                'confidence': predictions.tolist(),
                'signals': signals.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting historical signals: {e}")
            return {'error': str(e)}

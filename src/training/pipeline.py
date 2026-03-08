"""
Unified Model Training Pipeline
Combines: train_model.py, train_with_sentiment.py, tune_sentiment.py
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime
import xgboost as xgb

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Professional model training pipeline"""
    
    def __init__(self, config_path='config/production.yaml'):
        self.models = []
        self.best_model = None
        
    def prepare_data(self, with_sentiment=True):
        """Load and prepare training data"""
        if with_sentiment:
            df = pd.read_parquet('data/processed/btc_with_sentiment.parquet')
        else:
            df = pd.read_parquet('data/processed/btc_with_features.parquet')
        
        # Create target (24h ahead, 2% threshold)
        df['future_close'] = df['close'].shift(-24)
        df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
        df_clean = df.dropna()
        
        # Prepare features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Class balance: {y.mean():.2%}")
        
        return X, y, feature_cols
    
    def train_ensemble(self, X, y, n_models=5):
        """Train ensemble of XGBoost models"""
        models = []
        
        for i in range(n_models):
            seed = 42 + i * 100
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(X, y)
            models.append(model)
            logger.info(f"✅ Trained model {i+1}/{n_models} with seed {seed}")
        
        return models
    
    def find_optimal_threshold(self, models, X, y):
        """Find best probability threshold"""
        # Get ensemble predictions
        predictions = np.zeros(len(X))
        for model in models:
            predictions += model.predict_proba(X)[:, 1]
        predictions /= len(models)
        
        # Try different thresholds
        best_win_rate = 0
        best_threshold = 0.4
        
        for threshold in np.arange(0.35, 0.48, 0.01):
            signals = (predictions > threshold).astype(int)
            trades = signals.sum()
            
            if trades > 1000:
                # Calculate actual returns
                actual_returns = X.index.to_series().apply(
                    lambda x: pd.read_parquet('data/processed/btc_with_features.parquet').loc[x, 'close']
                ).pct_change().shift(-1).fillna(0).values
                
                wins = ((signals == 1) & (actual_returns > 0)).sum()
                win_rate = wins / trades
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} with {best_win_rate:.2%} win rate")
        return best_threshold, best_win_rate
    
    def run(self, retrain=False):
        """Run complete training pipeline"""
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        # Prepare data
        X, y, features = self.prepare_data(with_sentiment=True)
        
        # Train ensemble
        models = self.train_ensemble(X, y, n_models=5)
        
        # Find optimal threshold
        threshold, win_rate = self.find_optimal_threshold(models, X, y)
        
        # Save champion model
        champion = {
            'models': models,
            'threshold': threshold,
            'win_rate': win_rate,
            'features': features,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        path = 'data/models/yasen_alpha_champion.joblib'
        joblib.dump(champion, path)
        logger.info(f"💾 Champion model saved to {path}")
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Threshold: {threshold:.2f}")
        logger.info(f"Models: {len(models)}")
        logger.info("="*60)
        
        return {
            'win_rate': win_rate,
            'threshold': threshold,
            'models': len(models)
        }

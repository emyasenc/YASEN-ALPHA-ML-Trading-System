"""
Model Trainer Module
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import xgboost as xgb

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = []
        
    def load_data(self):
        """Load latest features"""
        df = pd.read_parquet('data/processed/features_latest.parquet')
        
        # Create target
        df['future_close'] = df['close'].shift(-24)
        df['target'] = ((df['future_close'] - df['close']) / df['close'] > 0.02).astype(int)
        df = df.dropna()  # Use df instead of df_clean
        
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'future_close', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target']
        
        logger.info(f"Loaded {len(X)} samples with {len(feature_cols)} features")
        return X, y, feature_cols, df  # Return df as well
    
    def train(self, retrain=False):
        """Train model ensemble"""
        X, y, features, df = self.load_data()  # Get df here
        
        # Train 5 models with different seeds
        models = []
        for seed in [42, 123, 456, 789, 999]:
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
            logger.info(f"Trained model with seed {seed}")
        
        # Calculate ensemble accuracy
        predictions = np.zeros(len(X))
        for model in models:
            predictions += model.predict_proba(X)[:, 1]
        predictions /= len(models)
        
        # Find best threshold
        best_win_rate = 0
        best_threshold = 0.4
        
        for threshold in np.arange(0.35, 0.48, 0.01):
            signals = (predictions > threshold).astype(int)
            trades = signals.sum()
            
            if trades > 1000:
                # Calculate returns using df (not df_clean)
                returns = df['close'].pct_change().shift(-1).fillna(0)
                wins = ((signals == 1) & (returns > 0)).sum()
                win_rate = wins / trades
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.2f} with {best_win_rate:.2%} win rate")
        
        # Save champion model
        champion = {
            'models': models,
            'threshold': best_threshold,
            'win_rate': best_win_rate,
            'features': features,
            'version': '2.0'
        }
        
        Path('data/models').mkdir(exist_ok=True)
        joblib.dump(champion, 'data/models/yasen_alpha_champion.joblib')
        
        return {
            'status': 'success',
            'win_rate': best_win_rate,
            'threshold': best_threshold,
            'models': len(models)
        }
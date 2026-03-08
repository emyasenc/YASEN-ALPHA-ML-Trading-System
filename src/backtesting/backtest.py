"""
Backtesting Module
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self):
        self.model = None
        
    def load_model(self, model_version='champion'):
        """Load specified model"""
        path = f'data/models/yasen_alpha_{model_version}.joblib'
        self.model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        
    def run(self, model_version='champion'):
        """Run backtest"""
        self.load_model(model_version)
        
        # Load features
        df = pd.read_parquet('data/processed/features_latest.parquet')
        
        # Prepare features
        X = df[self.model['features']].dropna()
        df = df.loc[X.index]
        
        # Get predictions
        predictions = np.zeros(len(X))
        for model in self.model['models']:
            predictions += model.predict_proba(X)[:, 1]
        predictions /= len(self.model['models'])
        
        # Generate signals
        signals = (predictions > self.model['threshold']).astype(int)
        
        # Calculate returns
        returns = df['close'].pct_change().shift(-1).fillna(0)
        strategy_returns = signals * returns
        
        # Metrics
        total_trades = signals.sum()
        if total_trades > 0:
            wins = ((signals == 1) & (returns > 0)).sum()
            win_rate = wins / total_trades
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
        else:
            win_rate = 0
            total_return = 0
            sharpe = 0
        
        results = {
            'win_rate': float(win_rate),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'total_trades': int(total_trades),
            'model_version': model_version
        }
        
        logger.info(f"Backtest results: {results}")
        return results

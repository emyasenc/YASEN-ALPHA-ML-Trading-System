import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
from pathlib import Path

class BitcoinPredictor:
    def __init__(self, model_dir='data/models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def create_target(self, df, horizon=24, threshold_pct=0.02):
        """
        Create binary target: 1 if price increases > threshold% in next horizon hours
        """
        df = df.copy()
        
        # Future price
        future_price = df['close'].shift(-horizon)
        current_price = df['close']
        
        # Calculate return
        future_return = (future_price - current_price) / current_price
        
        # Binary classification (1 if price goes up > threshold%)
        df['target'] = (future_return > threshold_pct).astype(int)
        
        # Remove last horizon rows (no target)
        df = df.iloc[:-horizon]
        
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        print(f"Positive class: {df['target'].mean():.2%}")
        
        return df
    
    def prepare_features_target(self, df, feature_cols, target_col='target'):
        """Split features and target, handle NaN"""
        # Drop any remaining NaN
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(feature_cols)}")
        
        return X, y
    
    def train_xgboost(self, X, y, params=None):
        """Train XGBoost with time series cross-validation"""
        if params is None:
            params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold+1}/5")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Predict and score
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            cv_scores.append(acc)
            models.append(model)
            
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {precision_score(y_val, y_pred):.4f}")
            print(f"  Recall: {recall_score(y_val, y_pred):.4f}")
            print(f"  F1: {f1_score(y_val, y_pred):.4f}")
        
        # Keep best model
        best_idx = np.argmax(cv_scores)
        self.model = models[best_idx]
        
        print("\n" + "="*50)
        print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"Best fold accuracy: {cv_scores[best_idx]:.4f}")
        
        return {
            'model': self.model,
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores)
        }
    
    def save_model(self, name='xgboost_model'):
        """Save trained model"""
        path = self.model_dir / f'{name}.joblib'
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
        return path
    
    def load_model(self, name='xgboost_model'):
        """Load trained model"""
        path = self.model_dir / f'{name}.joblib'
        self.model = joblib.load(path)
        print(f"✅ Model loaded from {path}")
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise Exception("No model loaded. Train or load a model first.")
        
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }

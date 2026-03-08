import pandas as pd
import numpy as np

class DataValidator:
    """Validate cryptocurrency data quality"""
    
    @staticmethod
    def check_completeness(df, expected_freq='1h'):
        """Check for missing time periods"""
        expected_range = pd.date_range(
            start=df.index.min(), 
            end=df.index.max(), 
            freq=expected_freq
        )
        missing = expected_range.difference(df.index)
        
        completeness = 1 - (len(missing) / len(expected_range))
        return {
            'completeness': completeness,
            'missing_periods': missing,
            'total_expected': len(expected_range),
            'total_actual': len(df)
        }
    
    @staticmethod
    def check_outliers(df, columns, zscore_threshold=3):
        """Identify statistical outliers using Z-score"""
        outliers = {}
        for col in columns:
            zscores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_indices = df[zscores > zscore_threshold].index.tolist()
            outliers[col] = outlier_indices
        return outliers
    
    @staticmethod
    def validate_price_consistency(df):
        """Ensure OHLC relationships are valid"""
        if df.empty:
            return False
        
        valid = (
            (df['high'] >= df['low']).all() and
            (df['high'] >= df['close']).all() and
            (df['high'] >= df['open']).all() and
            (df['low'] <= df['close']).all() and
            (df['low'] <= df['open']).all()
        )
        return valid
    
    @staticmethod
    def basic_stats(df):
        """Print basic statistics about the data"""
        stats = {
            'rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'price_mean': df['close'].mean(),
            'price_std': df['close'].std(),
            'price_min': df['close'].min(),
            'price_max': df['close'].max(),
            'volume_mean': df['volume'].mean(),
            'missing_values': df.isnull().sum().sum()
        }
        return stats

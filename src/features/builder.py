import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List

class FeatureBuilder:
    """Create 100+ technical indicators for model training"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_names = []
        
    def add_price_features(self):
        """Basic price-derived features"""
        # Returns at multiple horizons
        for period in [1, 4, 12, 24, 48, 168]:  # 1h, 4h, 12h, 24h, 48h, 1w
            self.df[f'return_{period}h'] = self.df['close'].pct_change(period)
            self.feature_names.append(f'return_{period}h')
            
            # Log returns (more normally distributed)
            self.df[f'log_return_{period}h'] = np.log(self.df['close'] / self.df['close'].shift(period))
            self.feature_names.append(f'log_return_{period}h')
        
        # Price relative to moving averages
        for period in [7, 25, 50, 99, 200]:
            ma = self.df['close'].rolling(period).mean()
            self.df[f'price_to_ma_{period}'] = self.df['close'] / ma - 1
            self.feature_names.append(f'price_to_ma_{period}')
        
        # Price relative to high/low ranges
        self.df['high_low_ratio'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df['close_open_ratio'] = (self.df['close'] - self.df['open']) / self.df['open']
        self.feature_names.extend(['high_low_ratio', 'close_open_ratio'])
        
        return self
    
    def add_technical_indicators(self):
        """Add technical indicators from pandas_ta"""
        
        # RSI - multiple periods
        for period in [7, 14, 21, 28]:
            rsi = ta.rsi(self.df['close'], length=period)
            self.df[f'rsi_{period}'] = rsi
            self.feature_names.append(f'rsi_{period}')
        
        # MACD
        macd = ta.macd(self.df['close'])
        self.df = pd.concat([self.df, macd], axis=1)
        self.feature_names.extend(['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'])
        
        # Bollinger Bands
        for period in [20, 50]:
            bb = ta.bbands(self.df['close'], length=period)
            self.df = pd.concat([self.df, bb], axis=1)
            self.feature_names.extend([
                f'BBL_{period}_2.0', f'BBM_{period}_2.0', f'BBU_{period}_2.0',
                f'BBB_{period}_2.0', f'BBP_{period}_2.0'
            ])
        
        # Volume indicators
        for period in [7, 14, 30]:
            self.df[f'volume_ma_{period}'] = self.df['volume'].rolling(period).mean()
            self.df[f'volume_ratio_{period}'] = self.df['volume'] / self.df[f'volume_ma_{period}']
            self.feature_names.extend([f'volume_ma_{period}', f'volume_ratio_{period}'])
        
        # On-Balance Volume
        obv = ta.obv(self.df['close'], self.df['volume'])
        self.df['obv'] = obv
        self.df['obv_change'] = obv.pct_change()
        self.feature_names.extend(['obv', 'obv_change'])
        
        # ATR - Average True Range (volatility)
        atr = ta.atr(self.df['high'], self.df['low'], self.df['close'])
        self.df['atr'] = atr
        self.df['atr_percent'] = atr / self.df['close']
        self.feature_names.extend(['atr', 'atr_percent'])
        
        return self
    
    def add_temporal_features(self):
        """Time-based features (cyclical encoding)"""
        # Hour of day (crypto has strong daily patterns)
        self.df['hour'] = self.df.index.hour
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # Day of week (weekend effects)
        self.df['dow'] = self.df.index.dayofweek
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['dow'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['dow'] / 7)
        
        # Month (seasonality in crypto)
        self.df['month'] = self.df.index.month
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Year (long-term trend)
        self.df['year'] = self.df.index.year
        self.df['year_sin'] = np.sin(2 * np.pi * (self.df['year'] - 2017) / 10)
        self.df['year_cos'] = np.cos(2 * np.pi * (self.df['year'] - 2017) / 10)
        
        self.feature_names.extend([
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
            'month_sin', 'month_cos', 'year_sin', 'year_cos'
        ])
        
        return self
    
    def add_lagged_features(self, columns: List[str], lags: List[int]):
        """Add lagged values for time series context"""
        for col in columns:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
                self.feature_names.append(f'{col}_lag_{lag}')
        return self
    
    def add_rolling_statistics(self, columns: List[str], windows: List[int]):
        """Rolling mean, std, min, max"""
        for col in columns:
            for window in windows:
                # Rolling mean
                self.df[f'{col}_rolling_mean_{window}'] = self.df[col].rolling(window).mean()
                self.feature_names.append(f'{col}_rolling_mean_{window}')
                
                # Rolling std (volatility)
                self.df[f'{col}_rolling_std_{window}'] = self.df[col].rolling(window).std()
                self.feature_names.append(f'{col}_rolling_std_{window}')
                
                # Rolling min/max (support/resistance)
                self.df[f'{col}_rolling_min_{window}'] = self.df[col].rolling(window).min()
                self.df[f'{col}_rolling_max_{window}'] = self.df[col].rolling(window).max()
                self.feature_names.extend([f'{col}_rolling_min_{window}', f'{col}_rolling_max_{window}'])
        
        return self
    
    def build_features(self) -> pd.DataFrame:
        """Execute full feature engineering pipeline"""
        self.add_price_features()
        self.add_technical_indicators()
        self.add_temporal_features()
        self.add_lagged_features(['close', 'volume'], [1, 2, 3, 6, 12, 24])
        self.add_rolling_statistics(['close', 'volume'], [6, 12, 24, 48, 168])
        
        # Drop NaN rows from lags/rolling
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_rows - len(self.df)
        
        print(f"✅ Created {len(self.feature_names)} features")
        print(f"📊 Final shape: {self.df.shape}")
        print(f"🗑️ Dropped {dropped} rows with NaN values")
        
        return self.df

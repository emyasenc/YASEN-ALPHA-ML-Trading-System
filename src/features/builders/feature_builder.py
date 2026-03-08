"""
Feature Engineering Module - Works with both pandas-ta and ta
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import pandas-ta first
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
    logger.info("✅ Using pandas-ta for technical indicators")
except ImportError:
    HAS_PANDAS_TA = False
    try:
        import ta as ta_lib
        HAS_TA_LIB = True
        logger.info("✅ Using ta library for technical indicators")
    except ImportError:
        HAS_TA_LIB = False
        logger.warning("⚠️ No technical analysis library found, using basic features only")

class FeatureBuilder:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_names = []
        
    def add_price_features(self):
        """Basic price-derived features"""
        # Returns at multiple horizons
        for period in [1, 4, 12, 24, 48, 168]:
            self.df[f'return_{period}h'] = self.df['close'].pct_change(period)
            self.feature_names.append(f'return_{period}h')
            
            # Log returns
            self.df[f'log_return_{period}h'] = np.log(self.df['close'] / self.df['close'].shift(period))
            self.feature_names.append(f'log_return_{period}h')
        
        # Price relative to moving averages
        for period in [7, 25, 50, 99, 200]:
            ma = self.df['close'].rolling(period).mean()
            self.df[f'price_to_ma_{period}'] = self.df['close'] / ma - 1
            self.feature_names.append(f'price_to_ma_{period}')
        
        # High/Low ratios
        self.df['high_low_ratio'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df['close_open_ratio'] = (self.df['close'] - self.df['open']) / self.df['open']
        self.feature_names.extend(['high_low_ratio', 'close_open_ratio'])
        
        return self
    
    def add_technical_indicators(self):
        """Add technical indicators using available library"""
        if HAS_PANDAS_TA:
            return self._add_pandas_ta_indicators()
        elif HAS_TA_LIB:
            return self._add_ta_lib_indicators()
        else:
            logger.warning("No technical indicators added")
            return self
    
    def _add_pandas_ta_indicators(self):
        """Add indicators using pandas-ta"""
        try:
            # RSI
            for period in [7, 14, 21, 28]:
                self.df[f'rsi_{period}'] = ta.rsi(self.df['close'], length=period)
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
            
            # ATR
            atr = ta.atr(self.df['high'], self.df['low'], self.df['close'])
            self.df['atr'] = atr
            self.df['atr_percent'] = atr / self.df['close']
            self.feature_names.extend(['atr', 'atr_percent'])
            
        except Exception as e:
            logger.error(f"Error adding pandas-ta indicators: {e}")
        
        return self
    
    def _add_ta_lib_indicators(self):
        """Add indicators using ta library"""
        try:
            # RSI
            self.df['rsi_14'] = ta_lib.momentum.RSIIndicator(self.df['close'], window=14).rsi()
            self.feature_names.append('rsi_14')
            
            # MACD
            macd = ta_lib.trend.MACD(self.df['close'])
            self.df['macd'] = macd.macd()
            self.df['macd_signal'] = macd.macd_signal()
            self.df['macd_diff'] = macd.macd_diff()
            self.feature_names.extend(['macd', 'macd_signal', 'macd_diff'])
            
            # Bollinger Bands
            bb = ta_lib.volatility.BollingerBands(self.df['close'], window=20)
            self.df['bb_high'] = bb.bollinger_hband()
            self.df['bb_low'] = bb.bollinger_lband()
            self.df['bb_mavg'] = bb.bollinger_mavg()
            self.df['bb_width'] = (self.df['bb_high'] - self.df['bb_low']) / self.df['bb_mavg']
            self.feature_names.extend(['bb_high', 'bb_low', 'bb_mavg', 'bb_width'])
            
            # ATR
            self.df['atr'] = ta_lib.volatility.AverageTrueRange(
                self.df['high'], self.df['low'], self.df['close'], window=14
            ).average_true_range()
            self.df['atr_percent'] = self.df['atr'] / self.df['close']
            self.feature_names.extend(['atr', 'atr_percent'])
            
        except Exception as e:
            logger.error(f"Error adding ta indicators: {e}")
        
        return self
    
    def add_temporal_features(self):
        """Time-based features"""
        self.df['hour'] = self.df.index.hour
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        self.df['dow'] = self.df.index.dayofweek
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['dow'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['dow'] / 7)
        
        self.df['month'] = self.df.index.month
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        self.df['year'] = self.df.index.year
        self.feature_names.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                                   'month_sin', 'month_cos', 'year'])
        
        return self
    
    def add_lagged_features(self, columns=None, lags=None):
        """Add lagged values"""
        if columns is None:
            columns = ['close', 'volume']
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24]
        
        for col in columns:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
                self.feature_names.append(f'{col}_lag_{lag}')
        return self
    
    def add_rolling_statistics(self, columns=None, windows=None):
        """Rolling statistics"""
        if columns is None:
            columns = ['close', 'volume']
        if windows is None:
            windows = [6, 12, 24, 48, 168]
        
        for col in columns:
            for window in windows:
                # Rolling mean
                self.df[f'{col}_rolling_mean_{window}'] = self.df[col].rolling(window).mean()
                self.feature_names.append(f'{col}_rolling_mean_{window}')
                
                # Rolling std
                self.df[f'{col}_rolling_std_{window}'] = self.df[col].rolling(window).std()
                self.feature_names.append(f'{col}_rolling_std_{window}')
                
                # Rolling min/max
                self.df[f'{col}_rolling_min_{window}'] = self.df[col].rolling(window).min()
                self.df[f'{col}_rolling_max_{window}'] = self.df[col].rolling(window).max()
                self.feature_names.extend([f'{col}_rolling_min_{window}', f'{col}_rolling_max_{window}'])
        
        return self
    
    def build_features(self):
        """Execute full feature engineering pipeline"""
        initial_rows = len(self.df)
        
        self.add_price_features()
        self.add_technical_indicators()
        self.add_temporal_features()
        self.add_lagged_features()
        self.add_rolling_statistics()
        
        # Drop NaN rows
        self.df = self.df.dropna()
        dropped = initial_rows - len(self.df)
        
        logger.info(f"✅ Created {len(self.feature_names)} features")
        logger.info(f"📊 Final shape: {self.df.shape}")
        logger.info(f"🗑️ Dropped {dropped} rows with NaN values")
        
        return self.df

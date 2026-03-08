import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import numpy as np

class MultiSourceCollector:
    """
    Collect Bitcoin data from multiple exchanges to get full history
    Combines data from Kraken and Bitstamp (Coinbase requires API keys)
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchanges that work without API keys
        self.exchanges = {}
        
        # Kraken - good for recent data
        try:
            self.exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            self.logger.info("✅ Kraken initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kraken: {e}")
        
        # Bitstamp - oldest exchange, good for historical
        try:
            self.exchanges['bitstamp'] = ccxt.bitstamp({
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            self.logger.info("✅ Bitstamp initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Bitstamp: {e}")
        
        # Exchange limits (max candles per request)
        self.limits = {
            'kraken': 720,
            'bitstamp': 1000
        }
        
    def fetch_exchange_data(self, exchange_id, symbol='BTC/USD', 
                           start_date='2017-01-01', end_date=None, 
                           max_candles=None):
        """
        Fetch data from a specific exchange with better error handling
        """
        if exchange_id not in self.exchanges:
            self.logger.error(f"Exchange {exchange_id} not available")
            return pd.DataFrame()
            
        exchange = self.exchanges[exchange_id]
        
        # Handle dates
        if end_date is None:
            end_date = datetime.now()
            
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Round to hour boundaries
        start_date = start_date.replace(minute=0, second=0, microsecond=0)
        end_date = end_date.replace(minute=0, second=0, microsecond=0)
        
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current_ts = start_ts
        chunk_size = self.limits.get(exchange_id, 500)
        retry_count = 0
        max_retries = 3
        
        self.logger.info(f"🚀 Fetching from {exchange_id}: {start_date} to {end_date}")
        
        while current_ts < end_ts:
            try:
                # Add small delay to avoid rate limits
                time.sleep(exchange.rateLimit / 1000)
                
                # Fetch candles
                candles = exchange.fetch_ohlcv(
                    symbol, '1h', 
                    since=current_ts, 
                    limit=chunk_size
                )
                
                if not candles:
                    self.logger.info(f"{exchange_id}: No more data available")
                    break
                
                # Add to collection
                all_candles.extend(candles)
                current_ts = candles[-1][0] + 1
                
                # Progress report
                progress = (current_ts - start_ts) / (end_ts - start_ts) * 100
                self.logger.info(
                    f"{exchange_id}: Got {len(candles)} candles, "
                    f"total: {len(all_candles)}, "
                    f"progress: {progress:.1f}%, "
                    f"up to: {datetime.fromtimestamp(current_ts/1000)}"
                )
                
                # Check if we've reached max_candles limit
                if max_candles and len(all_candles) >= max_candles:
                    self.logger.info(f"{exchange_id}: Reached max_candles limit ({max_candles})")
                    break
                
                retry_count = 0  # Reset retry count on success
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"{exchange_id} error (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"{exchange_id}: Max retries reached, moving on")
                    break
                    
                # Exponential backoff
                wait_time = 60 * retry_count
                self.logger.info(f"{exchange_id}: Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # Convert to DataFrame
        if not all_candles:
            self.logger.warning(f"{exchange_id}: No data retrieved")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        self.logger.info(f"✅ {exchange_id}: Retrieved {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def fetch_all(self, force_recent=False):
        """
        Fetch from available exchanges and combine
        """
        all_dfs = []
        
        if force_recent:
            # Just get recent data for testing
            sources = [
                ('kraken', '2026-01-01', None, 1000),
            ]
        else:
            # Full historical strategy
            sources = [
                # Bitstamp for older data (2017-2020)
                ('bitstamp', '2017-01-01', '2020-12-31', 30000),
                # Kraken for recent (2021-present)
                ('kraken', '2021-01-01', None, 40000),
            ]
        
        for exchange, start, end, max_candles in sources:
            try:
                df = self.fetch_exchange_data(
                    exchange, 
                    start_date=start, 
                    end_date=end,
                    max_candles=max_candles
                )
                
                if not df.empty:
                    self.logger.info(f"✅ {exchange}: Added {len(df)} rows")
                    all_dfs.append(df)
                else:
                    self.logger.warning(f"⚠️ {exchange}: No data returned")
                    
            except Exception as e:
                self.logger.error(f"❌ {exchange} failed: {e}")
                continue
        
        if not all_dfs:
            raise Exception("❌ No data fetched from any exchange")
        
        # Combine all data
        self.logger.info("🔄 Combining data from all sources...")
        combined = pd.concat(all_dfs)
        
        # Remove duplicates (keep first occurrence)
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Sort by date
        combined = combined.sort_index()
        
        # Fill any small gaps (less than 2 hours) with interpolation
        date_range = pd.date_range(start=combined.index.min(), end=combined.index.max(), freq='1h')
        combined = combined.reindex(date_range)
        
        # Linear interpolation for small gaps (max 2 hours)
        combined = combined.interpolate(method='linear', limit=2)
        
        # Drop any remaining NaNs (large gaps)
        combined = combined.dropna()
        
        self.logger.info(f"✅ FINAL: {len(combined)} rows from {combined.index.min()} to {combined.index.max()}")
        
        return combined
    
    def save_data(self, df, filename=None):
        """Save combined data to parquet and csv"""
        if filename is None:
            filename = f"btc_usdt_1h_{df.index.min().strftime('%Y%m%d')}_{df.index.max().strftime('%Y%m%d')}"
        
        # Save as parquet
        parquet_path = self.data_dir / f"{filename}.parquet"
        df.to_parquet(parquet_path, compression='snappy')
        self.logger.info(f"💾 Saved parquet: {parquet_path} ({len(df)} rows)")
        
        # Also save as CSV for easy viewing
        csv_path = self.data_dir / f"{filename}.csv"
        df.to_csv(csv_path)
        self.logger.info(f"💾 Saved CSV: {csv_path}")
        
        return parquet_path
    
    def quick_stats(self, df):
        """Print quick statistics about the data"""
        stats = {
            'rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'days': (df.index.max() - df.index.min()).days,
            'price_mean': df['close'].mean(),
            'price_min': df['close'].min(),
            'price_max': df['close'].max(),
            'volume_mean': df['volume'].mean(),
            'missing_values': df.isnull().sum().sum()
        }
        return stats

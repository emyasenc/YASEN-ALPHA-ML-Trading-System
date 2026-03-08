#!/usr/bin/env python
"""
YASEN-ALPHA: Full Bitcoin Historical Data Fetch
Fetches hourly BTC/USD data from Kraken (2017-present)
"""

from src.data.collector import BitcoinDataCollector
from src.data.validator import DataValidator
import logging
from datetime import datetime
import time
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_full.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("YASEN-ALPHA: FULL HISTORICAL DATA FETCH")
    logger.info("=" * 60)
    
    # Initialize collector
    collector = BitcoinDataCollector(exchange_id='kraken')
    
    # Fetch full history
    logger.info("Starting fetch from 2017-01-01 to present...")
    df = collector.fetch_ohlcv_range(
        symbol='BTC/USD',
        start_date='2017-01-01',
        timeframe='1h'
    )
    
    if df.empty:
        logger.error("No data received!")
        return
    
    # Log basic stats
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ FETCH COMPLETE")
    logger.info(f"📊 Rows: {len(df):,}")
    logger.info(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"⏱️  Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Validate data
    logger.info("=" * 60)
    logger.info("🔍 Validating data...")
    validator = DataValidator()
    
    completeness = validator.check_completeness(df)
    logger.info(f"Completeness: {completeness['completeness']:.2%}")
    if completeness['completeness'] < 0.95:
        logger.warning(f"Missing {len(completeness['missing_periods'])} periods")
    
    consistent = validator.validate_price_consistency(df)
    logger.info(f"Price consistency: {consistent}")
    
    outliers = validator.check_outliers(df, ['close', 'volume'])
    logger.info(f"Close outliers: {len(outliers['close'])}")
    logger.info(f"Volume outliers: {len(outliers['volume'])}")
    
    # Save the data
    logger.info("=" * 60)
    logger.info("💾 Saving data...")
    
    filename = f"btc_usd_1h_{df.index.min().strftime('%Y%m%d')}_{df.index.max().strftime('%Y%m%d')}.parquet"
    filepath = collector.save_raw_data(df, filename)
    
    # Also save a CSV backup (optional)
    csv_path = filepath.with_suffix('.csv')
    df.to_csv(csv_path)
    logger.info(f"CSV backup saved to {csv_path}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("✅ FULL FETCH COMPLETED SUCCESSFULLY")
    logger.info(f"📁 File: {filepath}")
    logger.info(f"📦 Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    logger.info(f"📊 Years: {(df.index.max() - df.index.min()).days / 365:.1f}")
    logger.info(f"⚡ Average rows/year: {len(df) / 9:.0f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

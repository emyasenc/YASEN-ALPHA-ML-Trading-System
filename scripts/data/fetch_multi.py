#!/usr/bin/env python
"""
YASEN-ALPHA: Multi-Source Bitcoin Data Fetch
Combines data from Kraken, Coinbase, and Bitstamp
"""

from src.data.multi_collector import MultiSourceCollector
import logging
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("YASEN-ALPHA: MULTI-SOURCE BITCOIN DATA FETCH")
    logger.info("=" * 60)
    
    # Initialize collector
    collector = MultiSourceCollector()
    
    # Fetch from all sources
    logger.info("Starting multi-source fetch...")
    df = collector.fetch_all(force_recent=False)  # Set to True for testing
    
    if df.empty:
        logger.error("No data received!")
        return
    
    # Print stats
    stats = collector.quick_stats(df)
    logger.info("=" * 60)
    logger.info("📊 DATASET STATISTICS")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Save data
    logger.info("=" * 60)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"btc_multi_source_{timestamp}"
    collector.save_data(df, filename)
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"📁 Files saved in data/raw/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

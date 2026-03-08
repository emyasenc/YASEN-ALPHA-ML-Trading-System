from src.data.collector import BitcoinDataCollector
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

print("Testing data collector with Kraken...")
collector = BitcoinDataCollector(exchange_id='kraken')

print("Fetching 1 week of test data...")
df = collector.fetch_ohlcv_range(
    symbol='BTC/USD',
    start_date='2024-01-01',
    end_date='2024-01-07',
    timeframe='1h'
)

print(f"Got {len(df)} rows")
if len(df) > 0:
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    
    collector.save_raw_data(df, 'test_data_kraken.parquet')
    print("\n✅ Test successful! Data saved.")
else:
    print("❌ No data received")

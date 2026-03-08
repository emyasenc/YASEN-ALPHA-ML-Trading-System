import pandas as pd
import glob

# Find the latest parquet file
files = glob.glob('data/raw/btc_multi_source_*.parquet')
latest = max(files)
print(f'Loading: {latest}')

df = pd.read_parquet(latest)
print(f'\n📊 Dataset shape: {df.shape}')
print(f'📅 Date range: {df.index.min()} to {df.index.max()}')
print(f'💰 Price range: ${df["close"].min():.0f} - ${df["close"].max():.0f}')
print(f'📈 Price mean: ${df["close"].mean():.0f}')
print(f'\nFirst 5 rows:')
print(df.head())
print(f'\nLast 5 rows:')
print(df.tail())

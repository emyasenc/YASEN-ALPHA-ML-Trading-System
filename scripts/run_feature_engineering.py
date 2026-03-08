import pandas as pd
import glob
from src.features.builder import FeatureBuilder

# Load data
files = glob.glob('data/raw/btc_multi_source_*.parquet')
latest = max(files)
print(f'Loading: {latest}')
df = pd.read_parquet(latest)

print(f'\n📊 Original data shape: {df.shape}')
print(f'📅 Date range: {df.index.min()} to {df.index.max()}')
print(f'💰 Price range: ${df["close"].min():.0f} - ${df["close"].max():.0f}')

# Build features
print('\n🔧 Building features...')
builder = FeatureBuilder(df)
df_features = builder.build_features()

print(f'\n✅ Created {len(builder.feature_names)} features')
print(f'📊 Final shape: {df_features.shape}')
print(f'\nFeatures sample: {builder.feature_names[:10]}')

# Save features
output_path = 'data/processed/btc_with_features.parquet'
df_features.to_parquet(output_path, compression='snappy')
print(f'\n💾 Saved features to {output_path}')

# Quick stats
print(f'\n📈 Feature stats:')
print(f'Total rows: {len(df_features)}')
print(f'Total features: {len(df_features.columns)}')
print(f'Memory usage: {df_features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB')

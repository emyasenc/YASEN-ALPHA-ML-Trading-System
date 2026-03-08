import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

print("Adding Fear & Greed Index...")

# Fetch Fear & Greed data
url = "https://api.alternative.me/fng/?limit=0&format=json"
response = requests.get(url)
data = response.json()

# Convert to DataFrame
fng_data = []
for item in data['data']:
    fng_data.append({
        'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
        'fear_greed': int(item['value']),
        'fear_greed_class': item['value_classification']
    })

fng_df = pd.DataFrame(fng_data)
fng_df['date'] = pd.to_datetime(fng_df['date'])
fng_df.set_index('date', inplace=True)

# Load your feature data
df = pd.read_parquet('data/processed/btc_with_features.parquet')

# Merge with Fear & Greed (forward fill for hourly data)
df = df.merge(fng_df[['fear_greed']], left_index=True, right_index=True, how='left')
df['fear_greed'] = df['fear_greed'].fillna(method='ffill')

# Add derived features
df['fear_greed_ma7'] = df['fear_greed'].rolling(7).mean()
df['fear_greed_change'] = df['fear_greed'].diff()
df['extreme_fear'] = (df['fear_greed'] < 25).astype(int)
df['extreme_greed'] = (df['fear_greed'] > 75).astype(int)

# Save enhanced features
df.to_parquet('data/processed/btc_with_sentiment.parquet')
print("✅ Enhanced features saved with Fear & Greed index!")

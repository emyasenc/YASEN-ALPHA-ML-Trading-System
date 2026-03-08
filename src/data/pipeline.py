"""
Unified Data Ingestion Pipeline
Combines: fetch_data.py, fetch_multi.py, test_collector.py
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from .sources.multi_source import MultiSourceCollector

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Professional data ingestion pipeline"""
    
    def __init__(self, config_path='config/production.yaml'):
        self.collector = MultiSourceCollector()
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def fetch_historical(self, start_date='2017-01-01', end_date=None):
        """Fetch historical data from multiple sources"""
        logger.info(f"Fetching historical data from {start_date} to {end_date or 'now'}")
        
        # Try multiple sources
        sources = ['kraken', 'bitstamp']
        all_data = []
        
        for source in sources:
            try:
                df = self.collector.fetch_from_exchange(
                    exchange_id=source,
                    start_date=start_date,
                    end_date=end_date
                )
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"✅ Got {len(df)} rows from {source}")
            except Exception as e:
                logger.error(f"❌ Failed to fetch from {source}: {e}")
        
        if not all_data:
            raise Exception("No data fetched from any source")
        
        # Combine and deduplicate
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f"data/raw/btc_raw_{timestamp}.parquet"
        combined.to_parquet(path)
        logger.info(f"💾 Saved raw data to {path}")
        
        return combined
    
    def validate_data(self, df):
        """Validate data quality"""
        from .validation.validator import DataValidator
        
        validator = DataValidator()
        results = {
            'completeness': validator.check_completeness(df),
            'consistency': validator.validate_price_consistency(df),
            'outliers': validator.check_outliers(df, ['close', 'volume'])
        }
        
        logger.info(f"Data validation: {results}")
        return results
    
    def run(self, start_date='2017-01-01', end_date=None):
        """Run complete data ingestion pipeline"""
        logger.info("="*60)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("="*60)
        
        # Fetch data
        df = self.fetch_historical(start_date, end_date)
        
        # Validate
        validation = self.validate_data(df)
        
        # Save latest version
        df.to_parquet('data/raw/latest.parquet')
        
        logger.info("="*60)
        logger.info("DATA INGESTION PIPELINE COMPLETE")
        logger.info(f"Rows: {len(df)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info("="*60)
        
        return {
            'status': 'success',
            'rows': len(df),
            'validation': validation
        }

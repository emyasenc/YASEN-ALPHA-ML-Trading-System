#!/usr/bin/env python3
"""
YASEN-ALPHA Professional Pipeline Orchestrator
Run: python scripts/run_pipeline.py --stage all
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import glob

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Data ingestion and validation pipeline"""
    def run(self, start_date=None, end_date=None, **kwargs):
        logger.info("🔄 Running Data Pipeline...")
        try:
            from src.data.sources.multi_source import MultiSourceCollector
            
            collector = MultiSourceCollector()
            if start_date:
                df = collector.fetch_ohlcv_range(start_date=start_date, end_date=end_date)
            else:
                df = collector.fetch_all()
            
            if df is not None and len(df) > 0:
                # Save with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'data/raw/btc_raw_{timestamp}.parquet'
                df.to_parquet(filename)
                logger.info(f"✅ Saved raw data to {filename}")
                
                # Also save as latest for other pipelines
                df.to_parquet('data/raw/latest.parquet')
                
                return {"status": "success", "rows": len(df), "file": filename}
            else:
                return {"status": "failed", "error": "No data returned"}
                
        except Exception as e:
            logger.error(f"❌ Data Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

class FeaturePipeline:
    """Feature engineering pipeline"""
    def run(self, **kwargs):
        logger.info("🔄 Running Feature Pipeline...")
        try:
            import pandas as pd
            from src.features.builders.feature_builder import FeatureBuilder
            
            # FIRST: Try to load the latest raw data (not test data)
            latest_raw = None
            
            # Look for timestamped raw files first
            raw_files = glob.glob('data/raw/btc_raw_*.parquet')
            if raw_files:
                latest_raw = max(raw_files, key=lambda x: Path(x).stat().st_mtime)
                logger.info(f"✅ Found timestamped raw data: {latest_raw}")
            else:
                # Fallback to latest.parquet
                latest_file = Path('data/raw/latest.parquet')
                if latest_file.exists():
                    latest_raw = latest_file
                    logger.info(f"✅ Found latest.parquet")
                else:
                    # Last resort - test data
                    test_file = Path('data/raw/test_data_kraken.parquet')
                    if test_file.exists():
                        latest_raw = test_file
                        logger.warning(f"⚠️ Using test data - no real data found")
                    else:
                        raise Exception("No raw data files found")
            
            logger.info(f"📂 Loading raw data from {latest_raw}")
            df = pd.read_parquet(latest_raw)
            logger.info(f"📊 Loaded {len(df)} rows of raw data")
            
            # Build features
            builder = FeatureBuilder(df)
            df_features = builder.build_features()
            
            # Save features with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            features_file = f'data/processed/features_{timestamp}.parquet'
            df_features.to_parquet(features_file)
            logger.info(f"✅ Features saved to {features_file}")
            
            # Also save as latest for other pipelines
            df_features.to_parquet('data/processed/features_latest.parquet')
            
            return {
                "status": "success", 
                "shape": df_features.shape,
                "features": len(builder.feature_names),
                "source": str(latest_raw),
                "file": features_file
            }
        except Exception as e:
            logger.error(f"❌ Feature Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

class TrainingPipeline:
    """Model training pipeline"""
    def run(self, retrain=False, **kwargs):
        logger.info("🔄 Running Training Pipeline...")
        try:
            from src.training.trainer import ModelTrainer
            
            trainer = ModelTrainer()
            results = trainer.train(retrain=retrain)
            
            logger.info(f"✅ Training Pipeline complete")
            return results
        except Exception as e:
            logger.error(f"❌ Training Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

class BacktestPipeline:
    """Backtesting pipeline"""
    def run(self, model_version='champion', **kwargs):
        logger.info("🔄 Running Backtest Pipeline...")
        try:
            from src.backtesting.backtest import Backtester
            
            backtester = Backtester()
            results = backtester.run(model_version=model_version)
            
            logger.info(f"✅ Backtest Pipeline complete")
            return results
        except Exception as e:
            logger.error(f"❌ Backtest Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

class PredictionPipeline:
    """Real-time prediction pipeline"""
    def run(self, **kwargs):
        logger.info("🔄 Running Prediction Pipeline...")
        try:
            from src.models.inference.predictor import YasenAlphaPredictor
            
            predictor = YasenAlphaPredictor()
            signal = predictor.get_current_signal()
            
            logger.info(f"✅ Prediction Pipeline complete")
            return {"signal": signal}
        except Exception as e:
            logger.error(f"❌ Prediction Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='YASEN-ALPHA Pipeline Orchestrator')
    parser.add_argument('--stage', choices=['data', 'features', 'train', 'backtest', 'predict', 'all'],
                       default='all', help='Pipeline stage to run')
    parser.add_argument('--start-date', help='Start date for data fetch')
    parser.add_argument('--end-date', help='End date for data fetch')
    parser.add_argument('--retrain', action='store_true', help='Force retraining')
    parser.add_argument('--model-version', default='champion', help='Model version to use')
    
    args = parser.parse_args()
    
    pipelines = {
        'data': DataPipeline(),
        'features': FeaturePipeline(),
        'train': TrainingPipeline(),
        'backtest': BacktestPipeline(),
        'predict': PredictionPipeline()
    }
    
    if args.stage == 'all':
        logger.info("🚀 Running FULL pipeline")
        results = {}
        for name, pipeline in pipelines.items():
            logger.info(f"▶️ Starting {name} pipeline")
            results[name] = pipeline.run(
                start_date=args.start_date,
                end_date=args.end_date,
                retrain=args.retrain,
                model_version=args.model_version
            )
            # If data pipeline fails, don't continue
            if name == 'data' and results[name].get('status') == 'failed':
                logger.error("❌ Data pipeline failed, stopping")
                break
                
        logger.info(f"✅ Full pipeline complete")
        print(json.dumps(results, default=str, indent=2))
    else:
        logger.info(f"🚀 Running {args.stage} pipeline")
        result = pipelines[args.stage].run(
            start_date=args.start_date,
            end_date=args.end_date,
            retrain=args.retrain,
            model_version=args.model_version
        )
        logger.info(f"✅ {args.stage} pipeline complete")
        print(json.dumps(result, default=str, indent=2))

if __name__ == "__main__":
    main()
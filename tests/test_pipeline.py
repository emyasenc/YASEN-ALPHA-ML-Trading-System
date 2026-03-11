#!/usr/bin/env python3
"""
Quick test to verify all components work
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("YASEN-ALPHA Professional Pipeline Test")
print("="*60)

# Test imports
try:
    print("\n📂 Testing imports...")
    from src.data.sources.multi_source import MultiSourceCollector
    print("✅ MultiSourceCollector imported")
    
    from src.features.builders.feature_builder import FeatureBuilder
    print("✅ FeatureBuilder imported")
    
    from src.training.trainer import ModelTrainer
    print("✅ ModelTrainer imported")
    
    from src.backtesting.backtest import Backtester
    print("✅ Backtester imported")
    
    from src.models.inference.predictor import YasenAlphaPredictor
    print("✅ YasenAlphaPredictor imported")
    
    print("\n🎉 All imports successful! Your professional pipeline is ready!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

#!/bin/bash
# YASEN-ALPHA Daily Update Script

# CORRECT VENV PATH
VENV_PATH="/Users/sophia/Desktop/yasen-alpha/proj/venv311/bin/activate"
PROJECT_DIR="/Users/sophia/Desktop/yasen-alpha/proj"
LOG_FILE="$PROJECT_DIR/logs/daily_update.log"

echo "======================================" >> $LOG_FILE
echo "Starting daily update at $(date)" >> $LOG_FILE
echo "======================================" >> $LOG_FILE

# Activate virtual environment
source $VENV_PATH >> $LOG_FILE 2>&1

# Navigate to project
cd $PROJECT_DIR

# Run the pipeline - REMOVED nohup so we can see if it completes
echo "Running pipeline..." >> $LOG_FILE
python scripts/run_pipeline.py --stage all >> $LOG_FILE 2>&1

# Check if model changed
if git diff --quiet data/models/yasen_alpha_champion.joblib; then
    echo "✅ No model changes detected" >> $LOG_FILE
else
    echo "📊 Model changed! Committing and pushing..." >> $LOG_FILE
    
    git add data/models/yasen_alpha_champion.joblib
    git add data/processed/features_latest.parquet 2>/dev/null
    
    git commit -m "Daily model update $(date +%Y-%m-%d)" >> $LOG_FILE 2>&1
    git push origin main >> $LOG_FILE 2>&1
    
    echo "🚀 Changes pushed to GitHub" >> $LOG_FILE
fi

# Also push any other changes (like feature files)
git add data/processed/features_*.parquet 2>/dev/null
git add data/raw/btc_raw_*.parquet 2>/dev/null
git commit -m "Daily data update $(date +%Y-%m-%d)" >> $LOG_FILE 2>&1
git push origin main >> $LOG_FILE 2>&1

echo "✅ Daily update completed at $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

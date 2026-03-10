#!/bin/bash
# YASEN-ALPHA Daily Update Script
# Runs pipeline and pushes changes to GitHub

# Set variables
PROJECT_DIR="/Users/sophia/Desktop/yasen-alpha/proj"
VENV_PATH="/Users/sophia/Desktop/yasen-alpha/venv311/bin/activate"
LOG_FILE="$PROJECT_DIR/logs/daily_update.log"

# Log start time
echo "======================================" >> $LOG_FILE
echo "Starting daily update at $(date)" >> $LOG_FILE
echo "======================================" >> $LOG_FILE

# Activate virtual environment
source $VENV_PATH >> $LOG_FILE 2>&1

# Navigate to project
cd $PROJECT_DIR

# Run the pipeline
echo "Running pipeline..." >> $LOG_FILE
python scripts/run_pipeline.py --stage all >> $LOG_FILE 2>&1

# Check if model changed
cd $PROJECT_DIR
if git diff --quiet data/models/yasen_alpha_champion.joblib; then
    echo "✅ No model changes detected" >> $LOG_FILE
else
    echo "📊 Model changed! Committing and pushing..." >> $LOG_FILE
    
    # Add model and feature files
    git add data/models/yasen_alpha_champion.joblib
    git add data/processed/features_latest.parquet 2>/dev/null
    
    # Commit with date
    git commit -m "Daily model update $(date +%Y-%m-%d)" >> $LOG_FILE 2>&1
    
    # Push to GitHub
    git push origin main >> $LOG_FILE 2>&1
    
    echo "🚀 Changes pushed. Render will auto-deploy." >> $LOG_FILE
fi

echo "✅ Daily update completed at $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

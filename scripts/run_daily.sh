#!/bin/bash
# Daily pipeline runner for YASEN-ALPHA

cd ~/Desktop/yasen-alpha/proj
source ~/Desktop/yasen-alpha/venv311/bin/activate

echo "==================================="
echo "YASEN-ALPHA Daily Pipeline"
echo "$(date)"
echo "==================================="

# Run the full pipeline
python scripts/run_pipeline.py --stage all

# Send notification (optional)
# osascript -e 'display notification "Pipeline completed" with title "YASEN-ALPHA"'

echo "✅ Daily pipeline complete at $(date)"

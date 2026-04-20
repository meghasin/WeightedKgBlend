#!/bin/bash
# kaggle_setup.sh
# Run this at the top of any Kaggle notebook to get the full project
# Usage: !bash kaggle_setup.sh YOUR_GITHUB_USERNAME

GITHUB_USER=${1:-"YOUR_GITHUB_USERNAME"}
REPO="WeightedKgBlend"

echo "Cloning WeightedKgBlend from GitHub..."
git clone https://github.com/$GITHUB_USER/$REPO /kaggle/working/$REPO

echo "Installing dependencies..."
pip install pykeen>=1.10.0 optuna>=3.0.0 scipy>=1.10.0 \
            scikit-learn>=1.3.0 rapidfuzz tqdm --quiet

echo "Setup complete!"
echo "Project at: /kaggle/working/$REPO"
ls /kaggle/working/$REPO

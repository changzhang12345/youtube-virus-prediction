# YouTube Viral Video Prediction

CPSC 3180/5180: Introduction to Machine Learning — Final Project  
Yale University, Spring 2026  
Student: Chang Zhang

## Overview

This project predicts whether a YouTube video will be among the top 20% most-viewed
videos on the trending page, using only information available before publication:
video title, publish time, and content category.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

Place the following files in a folder named `youtube_data/` inside the project directory:

```
project/
  youtube_data/
    USvideos.csv
    US_category_id.json
  youtube_viral_prediction.py
  ...
```

The dataset is the Kaggle Trending YouTube Video Statistics (US subset):
https://www.kaggle.com/datasets/datasnaek/youtube-new

## Running the Code

### Quick test (a few minutes)

```bash
python youtube_viral_prediction.py --fast
```

### Full run (20-40 minutes)

```bash
python youtube_viral_prediction.py
```

All output files (figures, CSV, log) are saved to the same directory as the script.

## Output Files

| File | Description |
|------|-------------|
| `phase1_viral_rate_by_category.png` | Viral rate by content category |
| `phase3_lr_baseline.png` | LR ROC curve and confusion matrix |
| `phase4_svm_comparison.png` | SVM vs LR comparison |
| `phase5_pca2d_overall.png` | PCA 2D projection (all categories) |
| `phase5_pca2d_by_category.png` | PCA 2D projection (per category) |
| `phase6_cross_category_heatmap.png` | Cross-category F1 matrix |
| `phase7_final_summary.png` | Final summary figure (2x2) |
| `USvideos_deduped.csv` | Deduplicated dataset used for training |
| `run_log.txt` | Full console output log |

## Method Summary

- **Label**: viral = 1 if views >= 80th percentile, else 0 (~20% positive)
- **Deduplication**: one record per video (by video_id)
- **Split**: 70/30 stratified, random_state=42
- **Experiment 1**: Logistic Regression on structured features only
- **Experiment 2**: RBF SVM on structured + TF-IDF title features (PCA to 100 dims)
- **Experiment 3**: PCA 2D visualization
- **Experiment 4**: Cross-category generalization (4x4 F1 matrix)

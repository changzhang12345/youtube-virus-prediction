# YouTube Viral Video Prediction

CPSC 3180/5180: Introduction to Machine Learning — Final Project  
Yale University, Spring 2026  
Student: Chang Zhang

## Overview

This project predicts whether a YouTube video will be among the top 20% most-viewed
videos on the trending page, using only information available before publication:
video title, tags, publish timestamp, content category, and description length.
Engagement metrics (views, likes, dislikes) are excluded as model inputs.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `tf-keras` is required by the `sentence-transformers` library on some
> environments. If you encounter a Keras compatibility error, run:
> `pip install tf-keras`

## Data Setup

Download the dataset from Kaggle and place the following files in a folder named
`youtube_data/` inside the project directory:

**Dataset:** [Kaggle Trending YouTube Video Statistics (US subset)](https://www.kaggle.com/datasets/datasnaek/youtube-new)

```
project/
  youtube_data/
    USvideos.csv
    US_category_id.json
  youtube_viral_prediction.py
  requirements.txt
  ...
```

## Running the Code

### Quick test (validates code runs, ~5 minutes)

```bash
python youtube_viral_prediction.py --fast
```

### Full run (~20 minutes)

```bash
python youtube_viral_prediction.py
```

All output files (figures, CSV, log) are saved to the same directory as the script.

## Output Files

| File | Description |
|------|-------------|
| `phase1_viral_rate_by_category.png` | Viral rate by content category |
| `phase3_lr_baseline.png` | LR ROC curve and confusion matrix |
| `phase4_svm_comparison.png` | SVM vs LR comparison (ROC, confusion matrix, metrics) |
| `phase5_pca3d_overall.png` | PCA 3D projection, two viewing angles |
| `phase5_pca3d_by_category.png` | PCA 3D projection per category (2×2) |
| `phase6_cross_category_heatmap.png` | Cross-category F1 matrix (4×4) |
| `phase7_final_summary.png` | Final summary figure (2×2) |
| `USvideos_deduped.csv` | Deduplicated dataset used for training |
| `run_log.txt` | Full console output log |

## Method Summary

- **Label**: viral = 1 if views ≥ 80th percentile (~933K views), else 0 (~20% positive)
- **Deduplication**: 40,949 rows → 6,351 unique videos (by video_id)
- **Split**: 70/30 stratified, random_state=42
- **Experiment 1**: Logistic Regression on 9 structured features (baseline + feature importance)
- **Experiment 2**: RBF Kernel SVM on structured features + 384-dim sentence embeddings of title and tags (PCA to 100 dims)
- **Experiment 3**: PCA 3D visualization of feature space
- **Experiment 4**: Cross-category generalization — 4×4 F1 matrix across Entertainment, Music, Howto & Style, Comedy

## Key Results

| Model | AUC | F1 |
|-------|-----|----|
| LR Baseline (structured only) | 0.599 | 0.352 |
| Kernel SVM (structured + embeddings) | 0.711 | 0.434 |

Cross-category: in-domain F1 = 0.859, cross-domain F1 = 0.166

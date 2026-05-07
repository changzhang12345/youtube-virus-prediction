"""
YouTube Viral Video Prediction
Yale CPSC 3180/5180 Introduction to Machine Learning Final Project

Research Question: Can we predict whether a YouTube video will trend
using only its title (no engagement data)?
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true', help='Quick test run with small sample and reduced grid')
args = parser.parse_args()
FAST = args.fast
if FAST:
    print("=== FAST MODE: quick validation run ===\n")

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Output directory (relative to this script's location)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "youtube_data")
os.makedirs(OUT_DIR, exist_ok=True)

# Tee stdout to log file
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_log_file = open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)

# ============================================================
# PHASE 1: Data Loading & Exploration
# ============================================================
print("=" * 60)
print("PHASE 1: Data Loading & Exploration")
print("=" * 60)

# Load dataset
df = pd.read_csv(f"{DATA_DIR}/USvideos.csv", encoding='latin1')
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Deduplicate by video_id (same video can appear on multiple trending days)
before = len(df)
df = df.drop_duplicates(subset='video_id', keep='first')
print(f"\nDeduplication: {before} rows → {len(df)} rows ({before - len(df)} duplicates removed)")
df.to_csv(f"{OUT_DIR}/USvideos_deduped.csv", index=False)
print("Saved: USvideos_deduped.csv")

# Construct viral label: top 10% by views = viral
threshold = df['views'].quantile(0.80)
df['viral'] = (df['views'] >= threshold).astype(int)
print(f"\nViral threshold (80th percentile views): {threshold:,.0f}")
print(f"\nViral distribution:\n{df['viral'].value_counts()}")
print(f"Viral rate: {df['viral'].mean():.2%}")

# Load category mapping
with open(f"{DATA_DIR}/US_category_id.json", 'r') as f:
    cat_data = json.load(f)

cat_map = {int(item['id']): item['snippet']['title'] for item in cat_data['items']}
print(f"\nCategory mapping:\n{cat_map}")

df['category_name'] = df['category_id'].map(cat_map)

# Viral rate by category
cat_viral = df.groupby('category_name')['viral'].mean().sort_values(ascending=False)
print(f"\nViral rate by category:\n{cat_viral}")

# Plot viral rate by category
fig, ax = plt.subplots(figsize=(12, 6))
cat_viral.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Viral Rate by Video Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Viral Rate (fraction)', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.axhline(y=0.1, color='red', linestyle='--', label='Overall 10% threshold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase1_viral_rate_by_category.png", dpi=150)
plt.close()
print("\nSaved: phase1_viral_rate_by_category.png")


# ============================================================
# PHASE 2: Feature Engineering
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Feature Engineering")
print("=" * 60)

# Drop rows with missing title or publish_time
df = df.dropna(subset=['title', 'publish_time'])
print(f"After dropping missing: {df.shape}")

# Parse publish_time
df['publish_time'] = pd.to_datetime(df['publish_time'], utc=True, errors='coerce')
df = df.dropna(subset=['publish_time'])

# Structured features
df['publish_hour'] = df['publish_time'].dt.hour
df['publish_weekday'] = df['publish_time'].dt.weekday  # 0=Monday, 6=Sunday
df['title_length'] = df['title'].str.len()
df['title_word_count'] = df['title'].str.split().str.len()
df['title_has_question'] = df['title'].str.contains(r'\?', regex=True).astype(int)
df['title_has_exclaim'] = df['title'].str.contains(r'!', regex=False).astype(int)
df['tag_count'] = df['tags'].fillna('').apply(lambda t: 0 if t in ('', '[none]') else len(t.split('|')))
df['desc_length'] = df['description'].fillna('').str.len()

structured_cols = ['publish_hour', 'publish_weekday', 'title_length', 'title_word_count',
                   'category_id', 'title_has_question', 'title_has_exclaim',
                   'tag_count', 'desc_length']
print(f"\nNew features sample:")
print(df[['title', 'publish_hour', 'publish_weekday', 'title_length',
          'title_has_question', 'title_has_exclaim', 'tag_count', 'desc_length']].head())

# Text column (must be created before split so df_train/df_test inherit it)
df['tags_clean'] = df['tags'].fillna('').replace('[none]', '', regex=False).str.replace('|', ' ', regex=False)
df['text'] = df['title'].astype(str) + ' ' + df['tags_clean']

# Labels
y = df['viral'].values
all_idx = np.arange(len(df))

# Split first to avoid leakage
idx_train, idx_test = train_test_split(
    all_idx, test_size=0.3, random_state=42, stratify=y
)

df_train = df.iloc[idx_train].copy()
df_test = df.iloc[idx_test].copy()
y_train = df_train['viral'].values
y_test = df_test['viral'].values

# Structured features: fit scaler on train only
scaler = StandardScaler()
X_structured_train = scaler.fit_transform(df_train[structured_cols].values)
X_structured_test = scaler.transform(df_test[structured_cols].values)

print("\nLoading sentence-transformer model (all-MiniLM-L6-v2)...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding train titles+tags...")
X_embed_train = st_model.encode(df_train['text'].tolist(), batch_size=64, show_progress_bar=True)
print("Encoding test titles+tags...")
X_embed_test = st_model.encode(df_test['text'].tolist(), batch_size=64, show_progress_bar=True)

# Full feature matrix: structured + sentence embeddings
X_full_train = np.hstack([X_structured_train, X_embed_train])
X_full_test = np.hstack([X_structured_test, X_embed_test])

print(f"\nEmbedding dim: {X_embed_train.shape[1]}")
print(f"Structured train shape: {X_structured_train.shape}")
print(f"Full train shape:       {X_full_train.shape}")
print(f"Full test shape:        {X_full_test.shape}")
print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")
print(f"Train viral rate: {y_train.mean():.2%}, Test viral rate: {y_test.mean():.2%}")


# ============================================================
# PHASE 3: Experiment 1 — Logistic Regression Baseline
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Experiment 1 — Logistic Regression Baseline")
print("=" * 60)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return results dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    results = {
        'model_name': model_name,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_test, y_prob),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'cm':        confusion_matrix(y_test, y_pred),
    }

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  ROC-AUC:   {results['auc']:.4f}")
    print(f"\nConfusion Matrix:\n{results['cm']}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return results

# Grid search for LR
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_param_grid = {'C': [0.1]} if FAST else {'C': [0.001, 0.01, 0.1, 1, 10]}
lr_base = LogisticRegression(class_weight='balanced', penalty='l2',
                              solver='lbfgs', max_iter=1000, random_state=42)
lr_gs = GridSearchCV(lr_base, lr_param_grid, cv=cv, scoring='f1',
                     n_jobs=1, verbose=1)
lr_gs.fit(X_structured_train, y_train)

print(f"\nBest LR parameters: {lr_gs.best_params_}")
print(f"Best CV F1: {lr_gs.best_score_:.4f}")

best_lr = lr_gs.best_estimator_
results_lr = evaluate_model(best_lr, X_structured_test, y_test,
                             "LR Baseline (Structured Features)")

# Feature importance for LR
feature_names = structured_cols
coefs = best_lr.coef_[0]
top_idx = np.argsort(np.abs(coefs))[::-1][:10]
print("\nTop 10 feature weights (LR):")
for i in top_idx:
    print(f"  {feature_names[i]:30s}: {coefs[i]:+.4f}")

# ROC curve for LR
fpr_lr, tpr_lr, _ = roc_curve(y_test, results_lr['y_prob'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
axes[0].plot(fpr_lr, tpr_lr, color='steelblue', lw=2,
             label=f"LR (AUC={results_lr['auc']:.3f})")
axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — LR Baseline')
axes[0].legend()

# Confusion matrix
sns.heatmap(results_lr['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-viral','Viral'], yticklabels=['Non-viral','Viral'],
            ax=axes[1])
axes[1].set_title('Confusion Matrix — LR Baseline')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase3_lr_baseline.png", dpi=150)
plt.close()
print("\nSaved: phase3_lr_baseline.png")


# ============================================================
# PHASE 4: Experiment 2 — Kernel SVM (Structured + Embeddings)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Experiment 2 — Kernel SVM (Structured + Embeddings)")
print("=" * 60)

# PCA to reduce dimensionality before SVM (speeds up training)
print("Applying PCA (100 components) to full feature matrix...")
pca = PCA(n_components=100, random_state=42)
X_pca_train = pca.fit_transform(X_full_train)
X_pca_test  = pca.transform(X_full_test)
print(f"PCA explained variance (100 components): {pca.explained_variance_ratio_.sum():.2%}")

svm_param_grid = (
    {'C': [1], 'gamma': [0.1]}
    if FAST else
    {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1]}
)

svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm_gs = GridSearchCV(svm_base, svm_param_grid, cv=cv, scoring='f1', n_jobs=1, verbose=1)
svm_gs.fit(X_pca_train, y_train)

print(f"\nBest SVM parameters: {svm_gs.best_params_}")
print(f"Best CV F1: {svm_gs.best_score_:.4f}")

best_svm = svm_gs.best_estimator_
results_svm = evaluate_model(best_svm, X_pca_test, y_test,
                              "Kernel SVM (Structured + Embeddings + PCA)")

fpr_svm, tpr_svm, _ = roc_curve(y_test, results_svm['y_prob'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(fpr_lr, tpr_lr, color='steelblue', lw=2,
             label=f"LR (AUC={results_lr['auc']:.3f})")
axes[0].plot(fpr_svm, tpr_svm, color='darkorange', lw=2,
             label=f"SVM (AUC={results_svm['auc']:.3f})")
axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend()

sns.heatmap(results_svm['cm'], annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Non-viral','Viral'], yticklabels=['Non-viral','Viral'],
            ax=axes[1])
axes[1].set_title('Confusion Matrix — Kernel SVM')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
lr_vals  = [results_lr[m]  for m in metrics_list]
svm_vals = [results_svm[m] for m in metrics_list]
x = np.arange(len(metrics_list))
w = 0.35
axes[2].bar(x - w/2, lr_vals,  w, label='LR Baseline',  color='steelblue', edgecolor='black')
axes[2].bar(x + w/2, svm_vals, w, label='Kernel SVM',   color='darkorange', edgecolor='black')
axes[2].set_xticks(x)
axes[2].set_xticklabels([m.upper() for m in metrics_list])
axes[2].set_ylim(0, 1)
axes[2].set_ylabel('Score')
axes[2].set_title('Model Comparison — All Metrics')
axes[2].legend()
for xi, (lv, sv) in enumerate(zip(lr_vals, svm_vals)):
    axes[2].text(xi - w/2, lv + 0.01, f'{lv:.3f}', ha='center', va='bottom', fontsize=7)
    axes[2].text(xi + w/2, sv + 0.01, f'{sv:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase4_svm_comparison.png", dpi=150)
plt.close()
print("\nSaved: phase4_svm_comparison.png")

print(f"\nF1 improvement (LR → SVM): {(results_svm['f1'] - results_lr['f1']):.4f} "
      f"({(results_svm['f1'] - results_lr['f1']) / max(results_lr['f1'], 1e-9) * 100:.1f}%)")
print(f"AUC improvement (LR → SVM): {(results_svm['auc'] - results_lr['auc']):.4f} "
      f"({(results_svm['auc'] - results_lr['auc']) / results_lr['auc'] * 100:.1f}%)")


# ============================================================
# PHASE 5: Experiment 3 — PCA Visualization
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Experiment 3 — PCA Visualization")
print("=" * 60)

# PCA to 3D on full training set
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pca2d = PCA(n_components=3, random_state=42)
X_2d_train = pca2d.fit_transform(X_full_train)
print(f"PCA 3D explained variance: PC1={pca2d.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca2d.explained_variance_ratio_[1]:.2%}, "
      f"PC3={pca2d.explained_variance_ratio_[2]:.2%}, "
      f"Total={pca2d.explained_variance_ratio_.sum():.2%}")

# Separate viral / non-viral
viral_mask    = (y_train == 1)
nonviral_mask = (y_train == 0)

# Subsample non-viral for plotting
rng = np.random.default_rng(42)
nv_idx = rng.choice(np.where(nonviral_mask)[0], size=min(1500, nonviral_mask.sum()), replace=False)
v_idx  = np.where(viral_mask)[0]

# Plot 3D from two viewing angles
for elev, azim, suffix in [(20, 45, 'a'), (20, 225, 'b')]:
    fig = plt.figure(figsize=(11, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(X_2d_train[nv_idx, 0], X_2d_train[nv_idx, 1], X_2d_train[nv_idx, 2],
                 c='steelblue', alpha=0.2, s=8, label=f'Non-viral (n={len(nv_idx)})')
    ax3d.scatter(X_2d_train[v_idx, 0], X_2d_train[v_idx, 1], X_2d_train[v_idx, 2],
                 c='darkorange', alpha=0.5, s=12, label=f'Viral (n={len(v_idx)})')
    ax3d.set_xlabel(f'PC1 ({pca2d.explained_variance_ratio_[0]:.1%})', labelpad=6)
    ax3d.set_ylabel(f'PC2 ({pca2d.explained_variance_ratio_[1]:.1%})', labelpad=6)
    ax3d.set_zlabel(f'PC3 ({pca2d.explained_variance_ratio_[2]:.1%})', labelpad=6)
    ax3d.set_title('PCA 3D Projection: Viral vs Non-viral Videos',
                   fontsize=13, fontweight='bold')
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/phase5_pca3d_overall_{suffix}.png", dpi=150)
    plt.close()
    print(f"Saved: phase5_pca3d_overall_{suffix}.png")

# Per-category PCA (4 categories) — 3D
df_train = df.iloc[idx_train].copy()
df_train['_pc1'] = X_2d_train[:, 0]
df_train['_pc2'] = X_2d_train[:, 1]
df_train['_pc3'] = X_2d_train[:, 2]
df_train['_viral'] = y_train

target_cats = {20: 'Gaming', 10: 'Music', 24: 'Entertainment', 28: 'Science & Technology'}
fig = plt.figure(figsize=(16, 12))

for idx_cat, (cat_id, cat_name) in enumerate(target_cats.items(), 1):
    ax3d = fig.add_subplot(2, 2, idx_cat, projection='3d')
    subset = df_train[df_train['category_id'] == cat_id]
    if len(subset) == 0:
        ax3d.set_title(f'{cat_name} — no data')
        continue

    nv = subset[subset['_viral'] == 0]
    vv = subset[subset['_viral'] == 1]
    nv_plot = nv.sample(min(400, len(nv)), random_state=42)

    ax3d.scatter(nv_plot['_pc1'], nv_plot['_pc2'], nv_plot['_pc3'],
                 c='steelblue', alpha=0.25, s=8, label=f'Non-viral ({len(nv)})')
    ax3d.scatter(vv['_pc1'], vv['_pc2'], vv['_pc3'],
                 c='darkorange', alpha=0.6, s=12, label=f'Viral ({len(vv)})')
    ax3d.set_title(f'{cat_name}', fontweight='bold', fontsize=10)
    ax3d.set_xlabel('PC1', fontsize=7, labelpad=2)
    ax3d.set_ylabel('PC2', fontsize=7, labelpad=2)
    ax3d.set_zlabel('PC3', fontsize=7, labelpad=2)
    ax3d.tick_params(labelsize=6)
    ax3d.legend(fontsize=7)
    ax3d.view_init(elev=20, azim=45)

plt.suptitle('PCA 3D by Category: Viral vs Non-viral', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase5_pca3d_by_category.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase5_pca3d_by_category.png")
print("Saved: phase5_pca2d_by_category.png")


# ============================================================
# PHASE 6: Experiment 4 — Cross-Category Generalization
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: Experiment 4 — Cross-Category Generalization")
print("=" * 60)

# Get top-4 categories by count
top4_cats = df['category_id'].value_counts().head(4).index.tolist()
top4_names = {cid: cat_map.get(cid, f'cat_{cid}') for cid in top4_cats}
print(f"Top 4 categories: {top4_names}")

# Encode all texts for the full dataset, then apply PCA (fit already done on train)
print("\nEncoding all texts for cross-category analysis...")
X_embed_all = st_model.encode(df['text'].tolist(), batch_size=64, show_progress_bar=True)
X_structured_all = scaler.transform(df[structured_cols].values)
X_full_all = np.hstack([X_structured_all, X_embed_all])
X_pca_all  = pca.transform(X_full_all)
y_all   = df['viral'].values
cat_all = df['category_id'].values

# Use best SVM params for all cross-category experiments
best_C     = svm_gs.best_params_['C']
best_gamma = svm_gs.best_params_['gamma']

# ---
# Model A: Train on all categories, test per category
# ---
print("\nModel A: Mixed training (all categories)...")
# Use training set (already defined)
svm_A = SVC(kernel='rbf', C=best_C, gamma=best_gamma,
            class_weight='balanced', probability=True, random_state=42)
svm_A.fit(X_pca_train, y_train)

f1_A = {}
for cat_id in top4_cats:
    cat_mask = (cat_all == cat_id)
    if cat_mask.sum() < 10:
        continue
    X_cat = X_pca_all[cat_mask]
    y_cat = y_all[cat_mask]
    y_pred_cat = svm_A.predict(X_cat)
    f1_A[cat_id] = f1_score(y_cat, y_pred_cat, zero_division=0)
    print(f"  Model A — {top4_names[cat_id]:25s}: F1 = {f1_A[cat_id]:.4f}")

# ---
# Model B: Train on one category, test on others
# ---
print("\nModel B: Category-specific training...")
# F1 matrix: rows=train_cat, cols=test_cat
f1_matrix = pd.DataFrame(index=[top4_names[c] for c in top4_cats],
                         columns=[top4_names[c] for c in top4_cats],
                         dtype=float)

for train_cat in top4_cats:
    train_mask = (cat_all == train_cat)
    X_train_cat = X_pca_all[train_mask]
    y_train_cat = y_all[train_mask]

    if len(np.unique(y_train_cat)) < 2:
        print(f"  Skipping {top4_names[train_cat]} — only one class")
        continue

    svm_B = SVC(kernel='rbf', C=best_C, gamma=best_gamma,
                class_weight='balanced', probability=True, random_state=42)
    svm_B.fit(X_train_cat, y_train_cat)

    for test_cat in top4_cats:
        test_mask = (cat_all == test_cat)
        X_test_cat = X_pca_all[test_mask]
        y_test_cat = y_all[test_mask]
        y_pred_cat = svm_B.predict(X_test_cat)
        f1_val = f1_score(y_test_cat, y_pred_cat, zero_division=0)
        f1_matrix.loc[top4_names[train_cat], top4_names[test_cat]] = f1_val
        print(f"  Train={top4_names[train_cat]:20s} → Test={top4_names[test_cat]:20s}: F1={f1_val:.4f}")

print(f"\nF1 Matrix:\n{f1_matrix}")

# Compute in-domain vs cross-domain stats
in_domain_f1  = [f1_matrix.iloc[i, i] for i in range(len(top4_cats))]
cross_domain_f1 = []
for i in range(len(top4_cats)):
    for j in range(len(top4_cats)):
        if i != j:
            cross_domain_f1.append(f1_matrix.iloc[i, j])

print(f"\nIn-domain average F1:    {np.nanmean(in_domain_f1):.4f}")
print(f"Cross-domain average F1: {np.nanmean(cross_domain_f1):.4f}")
print(f"Drop in F1 (in→cross):   {np.nanmean(in_domain_f1) - np.nanmean(cross_domain_f1):.4f}")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(f1_matrix.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, vmin=0, vmax=1, ax=ax,
            annot_kws={'size': 11})
ax.set_title('Cross-Category F1 Score Matrix\n(rows=train category, cols=test category)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Test Category', fontsize=11)
ax.set_ylabel('Train Category', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase6_cross_category_heatmap.png", dpi=150)
plt.close()
print("\nSaved: phase6_cross_category_heatmap.png")


# ============================================================
# PHASE 7: Final Summary Report
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: Final Summary Report")
print("=" * 60)

# Summary table
print("\n" + "=" * 80)
print(f"{'Model':<35} {'Features':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("=" * 80)
for r in [results_lr, results_svm]:
    feat = "Structured" if "LR" in r['model_name'] else "Struct+Embeddings"
    print(f"{r['model_name']:<35} {feat:<25} "
          f"{r['accuracy']:>6.3f} {r['precision']:>6.3f} "
          f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f}")
print("=" * 80)

# Improvements
f1_delta  = results_svm['f1']  - results_lr['f1']
auc_delta = results_svm['auc'] - results_lr['auc']
indomain_avg  = np.nanmean(in_domain_f1)
crossdomain_avg = np.nanmean(cross_domain_f1)
gap = indomain_avg - crossdomain_avg

print(f"\nKey Findings:")
print(f"  F1 improvement (LR → SVM):      {f1_delta:+.4f} ({f1_delta/max(results_lr['f1'],1e-9)*100:+.1f}%)")
print(f"  AUC improvement (LR → SVM):     {auc_delta:+.4f} ({auc_delta/results_lr['auc']*100:+.1f}%)")
print(f"  In-domain average F1:           {indomain_avg:.4f}")
print(f"  Cross-domain average F1:        {crossdomain_avg:.4f}")
print(f"  In-domain → Cross-domain drop:  {gap:.4f}")

# Comprehensive 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: ROC curves comparison
ax = axes[0, 0]
ax.plot(fpr_lr, tpr_lr, color='steelblue', lw=2,
        label=f"LR Baseline (AUC={results_lr['auc']:.3f})")
ax.plot(fpr_svm, tpr_svm, color='darkorange', lw=2,
        label=f"Kernel SVM (AUC={results_svm['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
ax.legend()

# Top-right: Metrics bar chart
ax = axes[0, 1]
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
lr_vals  = [results_lr[m]  for m in metrics]
svm_vals = [results_svm[m] for m in metrics]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, lr_vals,  w, label='LR Baseline',  color='steelblue', edgecolor='black')
ax.bar(x + w/2, svm_vals, w, label='Kernel SVM',   color='darkorange', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — All Metrics', fontsize=12, fontweight='bold')
ax.legend()
for xi, (lv, sv) in enumerate(zip(lr_vals, svm_vals)):
    ax.text(xi - w/2, lv + 0.01, f'{lv:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(xi + w/2, sv + 0.01, f'{sv:.3f}', ha='center', va='bottom', fontsize=8)

# Bottom-left: PCA 2D scatter (re-draw)
ax = axes[1, 0]
ax.scatter(X_2d_train[nv_idx, 0], X_2d_train[nv_idx, 1],
           c='steelblue', alpha=0.25, s=8, label='Non-viral')
ax.scatter(X_2d_train[v_idx, 0],  X_2d_train[v_idx, 1],
           c='darkorange', alpha=0.5, s=12, label='Viral')
ax.set_xlabel(f'PC1 ({pca2d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca2d.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA 2D: Viral vs Non-viral', fontsize=12, fontweight='bold')
ax.legend()

# Bottom-right: Cross-category heatmap
ax = axes[1, 1]
sns.heatmap(f1_matrix.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, vmin=0, vmax=1, ax=ax,
            annot_kws={'size': 10})
ax.set_title('Cross-Category F1 Score Matrix', fontsize=12, fontweight='bold')
ax.set_xlabel('Test Category')
ax.set_ylabel('Train Category')

plt.suptitle('YouTube Viral Prediction — Final Summary', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase7_final_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: phase7_final_summary.png")


print("\n" + "=" * 60)
print("All phases complete! Output files saved to:")
print(f"  {OUT_DIR}")
print("=" * 60)

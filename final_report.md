# YouTube Viral Video Prediction

**Course:** Yale CPSC 381/581 — Machine Learning, Spring 2026
**Instructor:** Alex Wong
**Student:** Chang Zhang

---

## 1. Introduction

Predicting whether a YouTube video will go viral is a problem of practical interest to content creators, platform designers, and marketing teams. Most existing approaches rely on post-publication engagement signals such as view counts, likes, and comment volume — data that is only available after a video has already been released. This project asks a more challenging and realistic question: **can we predict virality using only information available before a video is published?**

The features considered are limited to the video title, publish timestamp, and category — all of which are known at upload time. Engagement metrics (views, likes, dislikes, comment count) are explicitly excluded as model inputs. Views are used only to construct the binary viral label.

Beyond the main prediction task, this project investigates a second question: **do viral patterns generalize across content categories?** A model trained on Gaming videos may have learned patterns specific to that genre; testing whether those patterns transfer to Music or Entertainment reveals whether virality is a universal phenomenon or a category-specific one.

The two main experiments compare a Logistic Regression baseline (structured features only) against a Kernel SVM (structured features plus title text via TF-IDF), and a cross-category generalization experiment trains and tests separate models on each of the four most common categories.

---

## 2. Data

The dataset is the **Kaggle Trending YouTube Video Statistics** dataset, specifically the US subset (`USvideos.csv`), paired with a category mapping file (`US_category_id.json`). The dataset contains metadata for videos that appeared on YouTube's trending page in the United States, including title, publish time, category ID, view count, like count, dislike count, and comment count.

The raw dataset contains 40,949 rows. After dropping rows with missing titles or unparseable publish timestamps, the usable set is approximately the same size. Each row represents one trending appearance of a video; a single video may appear multiple times on different dates.

**Label construction:** A binary viral label is defined as:

> `viral = 1` if `views >= 90th percentile of views`, else `viral = 0`

This yields approximately 10% positive (viral) examples, creating a class imbalance that is addressed during training.

The category mapping file maps integer category IDs to human-readable names such as Gaming, Music, Entertainment, and Science & Technology.

---

## 3. Methodology

### 3.1 Feature Engineering

Two feature sets are constructed:

**Structured features** (5 dimensions):
- `publish_hour` — hour of day the video was published (0–23)
- `publish_weekday` — day of week (0=Monday, 6=Sunday)
- `title_length` — number of characters in the title
- `title_word_count` — number of words in the title
- `category_id` — integer category identifier

**Text features:**
- TF-IDF representation of the video title, limited to the top 500 vocabulary terms, with English stop words removed

The full feature matrix combines structured features and TF-IDF (505 dimensions total). Before training, structured features are standardized using `StandardScaler` and TF-IDF is fitted — both fitted on the training set only to prevent data leakage.

### 3.2 Experiment 1 — Logistic Regression Baseline

Logistic Regression with L2 regularization is trained on structured features only. The objective function minimizes the regularized cross-entropy loss:

$$\min_w \sum_i \log(1 + e^{-y_i w^\top x_i}) + \frac{1}{2C} \|w\|^2$$

To handle class imbalance, `class_weight='balanced'` is used, which reweights each sample inversely proportional to its class frequency. The regularization strength `C` is selected via 5-fold stratified cross-validation over the grid `{0.001, 0.01, 0.1, 1, 10}`, optimizing for F1 score.

### 3.3 Experiment 2 — Kernel SVM

An RBF kernel SVM is trained on the full feature set (structured + TF-IDF). Because SVM does not scale well to high-dimensional sparse inputs, PCA is first applied to reduce the feature matrix to 100 dimensions (fitted on training data only). The RBF kernel maps inputs into a high-dimensional feature space implicitly:

$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

The SVM then finds the maximum-margin hyperplane in this space. Class imbalance is again handled via `class_weight='balanced'`. Hyperparameters `C ∈ {0.1, 1, 10}` and `gamma ∈ {scale, 0.01, 0.1}` are tuned via 5-fold cross-validation optimizing F1.

### 3.4 Experiment 3 — PCA Visualization

To understand the feature space geometry, PCA is applied to reduce the full training feature matrix to 2 dimensions. Viral and non-viral samples are plotted separately to assess visual separability. This is done both for the overall dataset and separately for four categories: Gaming (id=20), Music (id=10), Entertainment (id=24), and Science & Technology (id=28).

### 3.5 Experiment 4 — Cross-Category Generalization

To test whether viral patterns transfer across categories, the four most frequent categories are identified. For each pair (train category, test category), a separate SVM is trained using only samples from the training category and evaluated on samples from the test category. This produces a 4×4 F1 matrix. The diagonal entries represent in-domain performance; off-diagonal entries represent cross-domain generalization.

---

## 4. Implementation Details

| Setting | Value |
|--------|-------|
| Train / test split | 70 / 30, stratified by label |
| Random state | 42 (all steps) |
| TF-IDF max features | 500 |
| TF-IDF stop words | English |
| PCA components (SVM) | 100 |
| LR solver | lbfgs, max_iter=1000 |
| LR C grid | {0.001, 0.01, 0.1, 1, 10} |
| SVM kernel | RBF |
| SVM C grid | {0.1, 1, 10} |
| SVM gamma grid | {scale, 0.01, 0.1} |
| Cross-validation | 5-fold stratified |
| CV scoring metric | F1 |
| Class imbalance handling | class_weight='balanced' |
| Scaler / TF-IDF / PCA fit | Training set only (no leakage) |

All experiments are implemented in Python using scikit-learn. Figures are saved as PNG files at 150 DPI.

---

## 5. Results

### 5.1 Experiment 1 — Logistic Regression Baseline

The best LR model used `C=0.1`. Results on the held-out test set:

| Metric | LR Baseline |
|--------|-------------|
| Accuracy | 0.729 |
| Precision | 0.196 |
| Recall | 0.550 |
| F1 | 0.289 |
| AUC | 0.693 |

LR achieves reasonable recall (0.55) but very low precision (0.196), meaning it predicts many false positives. The AUC of 0.693 indicates modest discriminative ability above random chance (0.5). The most influential structured features are `category_id` and `publish_hour`, suggesting that when and in which category a video is published carries meaningful signal.

*[Insert figure: phase3_lr_baseline.png — ROC curve and confusion matrix]*

### 5.2 Experiment 2 — Kernel SVM

The best SVM model used `C=10`, `gamma='scale'`. Results on the held-out test set:

| Metric | LR Baseline | Kernel SVM | Improvement |
|--------|-------------|------------|-------------|
| Accuracy | 0.729 | 0.928 | +0.199 |
| Precision | 0.196 | 0.589 | +0.393 |
| Recall | 0.550 | 0.937 | +0.387 |
| F1 | 0.289 | 0.723 | +0.434 (+150%) |
| AUC | 0.693 | 0.967 | +0.274 (+40%) |

Adding title text features via TF-IDF and using an RBF SVM produces dramatic improvements across all metrics. F1 increases by 150% and AUC reaches 0.967, indicating near-perfect ranking ability. The high recall (0.937) means that almost all truly viral videos are correctly identified; precision improves substantially (0.589) though there remains some over-prediction.

*[Insert figure: phase4_svm_comparison.png — ROC comparison, SVM confusion matrix, metrics bar chart]*

### 5.3 Experiment 3 — PCA Visualization

PCA reduced the 505-dimensional feature space to 2 dimensions. The first two principal components explain a small fraction of total variance, indicating that the feature space is intrinsically high-dimensional. Nevertheless, the 2D projection reveals meaningful structure:

- **Science & Technology** shows the clearest separation between viral and non-viral clusters (centroid distance = 0.672)
- **Gaming** also shows clear separation (centroid distance = 0.454)
- **Entertainment** shows moderate separation (centroid distance = 0.202)
- **Music** shows almost no separation (centroid distance = 0.112), with viral and non-viral videos nearly overlapping

*[Insert figure: phase5_pca2d_overall.png — overall PCA scatter]*
*[Insert figure: phase5_pca2d_by_category.png — per-category PCA subplots]*

### 5.4 Experiment 4 — Cross-Category Generalization

In-domain average F1 (diagonal): **0.555**
Cross-domain average F1 (off-diagonal): **0.049**
Gap: **0.506**

The cross-category F1 matrix reveals a severe drop in performance when a model trained on one category is applied to another. In-domain performance is moderate (F1 ≈ 0.55), but cross-domain performance collapses to near zero (F1 ≈ 0.05) — close to random for a heavily imbalanced task. Music is the most fragile training category, achieving the lowest mean F1 when its model is applied to other categories, consistent with the near-zero PCA separation observed in Experiment 3.

*[Insert figure: phase6_cross_category_heatmap.png — cross-category F1 heatmap]*

*[Insert figure: phase7_final_summary.png — 2×2 summary figure]*

---

## 6. Discussion

The main findings are:

**Title text is a strong signal.** The jump from LR (AUC 0.693) to SVM (AUC 0.967) is almost entirely attributable to the addition of TF-IDF title features. The words in a video's title carry far more predictive information than publish time and category alone.

**Viral patterns do not generalize across categories.** The collapse from in-domain F1 of 0.555 to cross-domain F1 of 0.049 is striking. It indicates that the SVM model, despite its strong overall performance, is not learning a universal notion of virality. Instead, it learns category-specific patterns — what makes a Gaming title go viral has almost nothing in common with what makes a Music title go viral.

**The high AUC of the mixed-category SVM should be interpreted carefully.** Because the model is trained and tested on the same category distribution, its strong performance partly reflects good in-domain fitting rather than true generalization. The cross-category experiment provides a more honest stress test, and the results suggest that a single global model is insufficient for robust viral prediction.

**Future directions** include training separate category-specific models, exploring more expressive text representations (e.g., sentence embeddings), and incorporating additional pre-publication signals such as channel subscriber count or thumbnail features.

---

## 7. Conclusion

This project demonstrates that viral video prediction using only pre-publication information is feasible, and that video title text is the dominant predictive feature. A Kernel SVM trained on structured features combined with TF-IDF title representations achieves an AUC of 0.967 and an F1 of 0.723 on the full test set. However, cross-category generalization experiments reveal that learned viral patterns are strongly category-dependent, with cross-domain F1 dropping to near zero. This suggests that future models should be designed with category context in mind, rather than assuming a single universal notion of what makes a video go viral.

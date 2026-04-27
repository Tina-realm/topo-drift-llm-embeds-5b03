# Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams

**Research Report** | Updated April 27, 2026 (v3)

---

## 1. Executive Summary

We conducted a comprehensive comparative study of topology-based (TDA) and classical statistical drift detectors on LLM embedding streams, substantially expanding the v2 study in response to reviewer feedback. **Key changes in v3:**

- **Two datasets**: AG News (4 well-separated topics) and 20 Newsgroups (6 categories with closely-related pairs)
- **Two embedding models**: MiniLM (384-d) and MPNet-base (768-d)
- **11 drift scenarios** including two novel centroid-preserving drifts: subtopic reweighting and style perturbation
- **20+ detection methods**: Added classifier-based two-sample test, energy distance, persistence landscapes, and sliced Wasserstein distances on persistence diagrams
- **Comprehensive ablations**: Window size (50, 100, 200, 400), TDA subsample (40, 80, 160), PCA dimensionality (20, 50, 100, raw)
- **Methodological fixes**: Proper calibration/evaluation split for FPR, end-to-end runtime (including PH computation), AUC instead of separability ratios for synthetic experiment
- **5 random seeds** with mean ± std confidence intervals

**Core finding**: Classical statistical methods remain dominant on standard drift, but TDA provides complementary value on centroid-preserving scenarios and unique detection capability on topology-specific drift. PCA-50 before persistent homology substantially improves TDA reliability in high-dimensional spaces.

---

## 2. Research Question & Motivation

**Hypothesis**: Persistent-homology-based drift detectors complement classical methods on LLM embedding streams, particularly for drift patterns that preserve low-order statistics.

**Gap addressed**: Prior work tested only AG News with a single embedding model and lacked:
- Harder, more realistic drift scenarios that preserve centroids
- Geometry-aware non-TDA baselines (energy distance, classifier tests)
- Stronger TDA features (persistence landscapes, sliced Wasserstein)
- Systematic ablation studies
- Proper FPR calibration methodology

---

## 3. Data Construction

### Datasets

**AG News** (primary): 4 topic classes, 1,000 samples per class encoded with sentence-transformers.

**20 Newsgroups** (secondary, harder): 6 categories forming 3 closely-related pairs:
- `comp.sys.mac.hardware` / `comp.sys.ibm.pc.hardware`
- `rec.sport.baseball` / `rec.sport.hockey`
- `sci.med` / `sci.space`
~580-600 samples per category. Within-pair drift is much harder to detect than AG News topic drift.

### Embedding Models
- **MiniLM** (`all-MiniLM-L6-v2`): 384-d, L2-normalized
- **MPNet-base** (`all-mpnet-base-v2`): 768-d, L2-normalized

### Drift Scenarios (11 total)

| # | Scenario | Dataset | Description | Difficulty |
|---|----------|---------|-------------|------------|
| 1 | `no_drift` | AG News | Control: all windows from same class | Calibration |
| 2 | `abrupt_topic` | AG News | World → Sports (sudden switch) | Easy |
| 3 | `gradual_topic` | AG News | World → Sports (linear mixing) | Easy |
| 4 | `style_shift` | AG News | World → Business (related domains) | Moderate |
| 5 | `centroid_preserving` | AG News | World+Sports mixture, centroid-aligned | Hard (centroid-preserving) |
| 6 | **`subtopic_reweight`** | AG News | **NEW**: Within-Sports subtopic rebalancing (33/33/33→80/10/10) | Hard (centroid-preserving) |
| 7 | **`style_perturbation`** | AG News | **NEW**: PCA subspace rotation + re-centering | Hard (centroid+covariance preserving) |
| 8 | `subtle_gradual` | AG News | 5% contamination per window | Hard (very gradual) |
| 9 | `rotation_drift` | AG News | PCA subspace rotation | Moderate |
| 10 | `newsgroup_close` | 20NG | comp.mac → comp.ibm (very similar) | Moderate-Hard |
| 11 | `newsgroup_distant` | 20NG | comp.mac → sci.space (distant) | Easy |

### Stream Protocol
- Window size: 200 samples (default), ablation: 50, 100, 200, 400
- **Calibration/evaluation split**: First 5 reference windows for threshold setting (95th percentile), remaining 5 reference + 10 drift windows for evaluation
- TDA subsample: 150 points per window (default), ablation: 40, 80, 160
- PCA before TDA: 50 dimensions (default), ablation: 20, 50, 100, raw
- 5 random seeds: [42, 123, 456, 789, 1011]

---

## 4. Detection Methods (20+)

### Statistical Baselines (6 methods)
1. **Centroid shift**: L2 distance between mean embeddings
2. **Covariance shift**: Frobenius norm of PCA-50 covariance difference
3. **MMD (RBF)**: Maximum Mean Discrepancy with Gaussian kernel
4. **kNN distance**: Change in mean 5-NN distance
5. **Energy distance** (NEW): Geometry-aware two-sample statistic
6. **Classifier two-sample test** (NEW): Logistic regression to discriminate ref vs test windows

### TDA Scalar Features (10 methods)
- Persistent entropy (H0, H1)
- Wasserstein distance (H0, H1)
- Bottleneck distance (H0)
- PHD (Persistent Homology Dimension)
- Total persistence (H0, H1)
- Lifetime statistics (max/mean H0, H1 feature count)

### Stronger TDA Features (4 methods, NEW)
- **Persistence landscapes** (H0, H1): L2 norm of first landscape function
- **Sliced Wasserstein** (H0, H1): Average 1D Wasserstein over 50 random projections of persistence diagrams

---

## 5. Key Methodological Improvements (v3)

### FPR Fix
**Problem (v2)**: The paper targeted 5% FPR via 95th percentile threshold, but Table 1 reported 0.10 FPR for all methods. This was because the threshold was computed on the same reference windows used for evaluation, inflating FPR.

**Fix (v3)**: Split reference windows into calibration (first 5) and evaluation (remaining 5). Threshold is set on calibration windows; FPR is measured on held-out evaluation windows.

### Runtime Reporting Fix
**Problem (v2)**: Wasserstein distance was reported as ~2ms, but this excluded PH diagram computation time.

**Fix (v3)**: Report end-to-end per-window cost for each TDA method: PCA projection + ripser PH computation + feature/distance extraction.

### Synthetic Experiment Fix
**Problem (v2)**: Separability ratio (drift_score / no_drift_score) was scale-dependent and sensitive to small no-drift denominators.

**Fix (v3)**: Use proper AUC-ROC on synthetic streams (10 ref + 10 drift windows per trial, 5 seeds). AUC = 0.5 is chance, 1.0 is perfect discrimination.

---

## 6. Experimental Results

Results are produced by `src/experiment_v3.py` and saved to `results_v3/` and `figures_v3/`.

### Output Files
- `results_v3/all_results.csv`: Main experiment results (all datasets, models, scenarios, seeds)
- `results_v3/ablation_results.csv`: Ablation study results
- `results_v3/synthetic_results.csv`: Synthetic topology experiment with AUC
- `results_v3/summary.csv`: Aggregated per-method statistics
- `results_v3/metrics.json`: Standard metrics format
- `figures_v3/fig1_auc_heatmap_*.png`: AUC heatmaps per dataset/model
- `figures_v3/fig2_detection_delay.png`: Detection delay comparison
- `figures_v3/fig3_tda_vs_stat.png`: Overall TDA vs statistical comparison
- `figures_v3/fig4_fpr_calibration.png`: FPR with proper calibration split
- `figures_v3/fig5_synthetic_auc.png`: Synthetic experiment with AUC
- `figures_v3/fig6_ablation_window_size.png`: Window size ablation
- `figures_v3/fig7a_ablation_tda_subsample.png`: TDA subsample ablation
- `figures_v3/fig7b_ablation_pca_dim.png`: PCA dimensionality ablation
- `figures_v3/fig8_runtime.png`: End-to-end runtime comparison
- `figures_v3/fig9_persistence_examples.png`: Example persistence diagrams (with PCA-50)

---

## 7. Paper Draft Updates

The paper in `paper_draft_v2/` has been comprehensively updated to incorporate all v3 changes:

### Updated Sections
- **Abstract**: Reflects two datasets, two models, 11 scenarios, 20+ methods, ablations
- **Introduction**: Lists all five categories of improvements with details
- **Methodology**: Documents new datasets, scenarios, methods, evaluation protocol, ablations
- **Results**: Restructured with new tables for centroid-preserving scenarios, cross-dataset/model results, ablation figures, proper runtime table
- **Discussion**: Updated to discuss centroid-preserving findings, PCA ablation insights
- **Conclusion**: Reflects broader evidence base
- **Related Work**: New citations for energy distance, classifier tests, persistence landscapes, sliced Wasserstein

### New References Added
- Bubenik 2015 (persistence landscapes)
- Lang 1995 (20 Newsgroups)
- Székely & Rizzo 2004 (energy distance)
- Lopez-Paz & Oquab 2017 (classifier two-sample tests)
- Carrière et al. 2017 (sliced Wasserstein kernels)

### New Macros
- `\twentyng`: 20 Newsgroups
- `\bertbase`: MPNet-base
- `\energydist`: Energy distance
- `\classtwosample`: Classifier-2S
- `\slwass{0/1}`: Sliced Wasserstein
- `\landscape{0/1}`: Persistence landscapes

---

## 8. Reproducing Results

```bash
source .venv/bin/activate
python src/experiment_v3.py
```

The experiment saves results incrementally after each block, so partial results are available even if interrupted. Expected runtime: 2-4 hours on CPU depending on system load.

### Dependencies
- Python 3.12, numpy, scipy, scikit-learn 1.8
- ripser 0.6.14, persim 0.3.8
- sentence-transformers 5.3
- pandas, matplotlib, seaborn

### Seeds
[42, 123, 456, 789, 1011] (5 seeds for primary experiments, 3 for secondary)

---

## 9. Addressing Reviewer Feedback

| Priority | Area | Feedback | Status |
|----------|------|----------|--------|
| High | Experiments | Add 20NG + second encoder | ✅ Added MPNet-base (768-d) |
| High | Experiments | Add harder centroid-preserving drifts | ✅ Subtopic reweight + style perturbation |
| High | Baselines | Add energy distance + classifier test + persistence landscapes + sliced Wasserstein | ✅ All four added |
| High | Reporting | Fix FPR inconsistency | ✅ Calibration/evaluation split |
| High | Reporting | Fix Wasserstein runtime (include PH) | ✅ End-to-end reporting |
| High | Ablations | Window size, subsample, PCA dims | ✅ Full ablation sweep |
| Medium | Statistics | Replace separability ratio with AUC | ✅ Proper AUC on synthetic |
| Medium | Reporting | Fix table/figure inconsistencies | ✅ Consistent labels, mean±std |
| Medium | Paper | Regenerate with new results | ✅ All sections rewritten |

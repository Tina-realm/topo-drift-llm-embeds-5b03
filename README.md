# Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams

Controlled comparative study of TDA-based vs classical statistical drift detectors on LLM embedding streams.

## Key Findings

- **Statistical methods (covariance, MMD, centroid) achieve near-perfect detection** (AUC 0.92-1.0) on natural topic drift in sentence embeddings
- **Best TDA method**: Wasserstein H1 distance (mean AUC=0.832), significantly outperforms kNN (p=0.0003)
- **TDA's unique advantage**: On topology-specific drift (annular/loop emergence), TDA PE H1 achieves 19.3x separability where centroid shift fails (0.8x)
- **Better FPR calibration**: TDA methods produce lower false positive rates (0.08-0.10) vs statistical methods (0.45-0.48) in small-sample regimes
- **Practical recommendation**: Use covariance/MMD as primary detector + TDA Wasserstein H1 as complementary secondary detector (~40ms overhead on CPU)

## Setup & Reproduction

```bash
# Create and activate environment
uv venv && source .venv/bin/activate

# Install dependencies
uv add numpy pandas scikit-learn scipy matplotlib seaborn ripser persim sentence-transformers datasets tqdm

# Run experiments (~51 min on CPU)
python src/experiment_v2.py
```

## Project Structure

```
.
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and hypothesis decomposition
├── literature_review.md         # Literature review (17 papers)
├── resources.md                 # Resource catalog
├── src/
│   ├── experiment_v2.py         # Main experiment (18 methods, 10 scenarios, 5 seeds)
│   ├── drift_experiment.py      # Prior experiment (v1)
│   └── synthetic_topology_experiment.py  # Prior synthetic experiment
├── results_v2/                  # Output data
│   ├── all_results.csv          # Full results table
│   ├── metrics.json             # Per-method metrics
│   ├── raw_results.jsonl        # Window-level outputs
│   ├── summary.csv              # Aggregated summary
│   └── synthetic_results.csv    # Synthetic experiment
├── figures_v2/                  # Visualizations
│   ├── fig1_auc_heatmap.png     # AUC by method × drift type
│   ├── fig2_detection_delay.png # Detection delay comparison
│   ├── fig3_tda_vs_stat.png     # TDA vs statistical overview
│   ├── fig4_fpr_calibration.png # FPR on no-drift data
│   ├── fig5_synthetic_separability.png  # Synthetic topology results
│   ├── fig6_window_sensitivity.png      # AUC vs window size
│   └── fig7_persistence_examples.png    # Example persistence diagrams
├── papers/                      # Downloaded research papers (21)
├── datasets/                    # AG News, 20 Newsgroups, DBpedia14
└── code/                        # Cloned repos (ripser, giotto-tda, frouros, etc.)
```

## Methodology

- **Datasets**: AG News (4 topics, 3K/class), 20 Newsgroups (6 categories, ~590/class)
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim, L2-normalized)
- **18 detectors**: 5 statistical baselines + 13 TDA features (H0/H1 persistent entropy, Wasserstein/bottleneck distances, PHD, persistence statistics)
- **10 drift scenarios**: abrupt, gradual, style shift, centroid-preserving, rotation, newsgroup close/distant, synthetic topology
- **Evaluation**: AUC-ROC, detection delay, FPR, runtime; 5 seeds, 3 window sizes

See [REPORT.md](REPORT.md) for full details.

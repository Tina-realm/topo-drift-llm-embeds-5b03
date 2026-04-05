# Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams

A comparative study of TDA-based vs statistical drift detectors on LLM text embedding streams.
Machine Learning research | NeuriCo | 2026-04-05

## Key Findings

1. **Statistical methods dominate on natural topic drift**: Centroid shift, covariance, and MMD achieve AUC=1.0 on AG News topic drift (World/Sports/Business/Sci-Tech), while TDA methods reach AUC 0.49–0.93.
2. **TDA has genuine advantage on topology-only drift**: In synthetic experiments where centroids are preserved but topology changes (blob→annulus), TDA H1 entropy achieves 12.3x separability vs 0.8x for centroid shift.
3. **Best TDA method**: H0 Wasserstein distance (AUC=0.713 mean, 0.002s/window) is the most practical TDA detector for real embeddings.
4. **Practical recommendation**: Use centroid + MMD as primary detectors; add TDA H1 features as a complementary secondary signal for geometric/structural drift.

## Reproduction

```bash
# Set up environment
source .venv/bin/activate

# Run main experiment (requires AG News dataset in datasets/ag_news/)
python src/drift_experiment.py

# Run supplementary synthetic experiment
python src/synthetic_topology_experiment.py
```

**Runtime**: ~9 minutes on CPU (4.5 min embedding, 4.5 min drift detection)

## File Structure

```
├── src/
│   ├── drift_experiment.py          # Main experiment (AG News + 5 drift scenarios)
│   └── synthetic_topology_experiment.py  # Synthetic topology-only drift
├── results/
│   ├── metrics.json                 # Per-method detection metrics
│   ├── raw_results.jsonl            # Window-level outputs
│   └── synthetic_results.csv       # Synthetic experiment results
├── figures/
│   ├── fig1_detection_by_drift_type.png
│   ├── fig2_delay_vs_fpr.png
│   ├── fig3_tda_vs_statistical.png
│   ├── fig4_sensitivity_vs_window_size.png
│   ├── fig5_persistence_examples.png
│   ├── fig6_score_traces.png
│   └── fig7_synthetic_topology_experiment.png
├── datasets/ag_news/               # AG News dataset (see datasets/README.md)
├── REPORT.md                       # Full research report
└── planning.md                     # Research planning document
```

See [REPORT.md](REPORT.md) for the full research report with methodology, results, and analysis.

## Environment

- Python 3.12.8, ripser 0.6.14, persim 0.3.8
- sentence-transformers 5.3.0 (`all-MiniLM-L6-v2`, 384-dim)
- numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1
- CPU-only (no GPU required)

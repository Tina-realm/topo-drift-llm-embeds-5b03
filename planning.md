# Research Planning: Topological Drift Detection for LLM Embedding Streams

## Motivation & Novelty Assessment

### Why This Research Matters
Deployed LLM systems are continuously exposed to evolving input distributions (topic drift, style drift, adversarial shift). Existing drift detectors (MMD, covariance shift, centroid shift) rely on low-order statistics that may miss subtle *geometric* or *topological* reorganization of the embedding manifold. TDA-based methods can capture multi-scale structural changes invisible to first/second moment methods, providing a more principled signal for continuous AI monitoring in mission-critical deployments.

### Gap in Existing Work
From literature_review.md:
- No paper directly evaluates TDA drift detection on LLM input embedding streams in a systematic comparative study
- TDA drift detection (Basterrech 2024) exists but only validated on synthetic image streams, not text embeddings
- PHD (persistent homology dimension) has been shown discriminative for LLM vs human text but not applied to drift monitoring
- Streaming TDA is computationally unexplored for realistic window sizes

### Our Novel Contribution
We conduct the first controlled comparative study of TDA-based drift detectors (persistent entropy, PHD, Betti curves, persistence landscapes) vs classical baselines (centroid shift, covariance shift, MMD, kNN) on real LLM text embedding streams with multiple drift types including a "geometry-preserving centroid" drift scenario specifically designed to challenge low-order statistics.

### Experiment Justification
- **Exp 1** (Abrupt topic drift): Validates TDA can detect strong drift; establishes baseline comparison
- **Exp 2** (Gradual drift): Tests detection delay — topology may detect subtle cumulative shifts earlier
- **Exp 3** (Geometry-without-centroid-shift): Key experiment — mixes classes with equal means but different topology; where TDA should uniquely excel
- **Exp 4** (No-drift / false positive calibration): Validates FPR calibration on stable reference

---

## Research Question
Can persistent-homology-based summaries detect distribution drift in LLM embedding streams more sensitively than centroid shift, covariance shift, and MMD, particularly when drift alters geometric structure without large changes in low-order statistics?

## Hypothesis Decomposition
1. **H1**: TDA detectors achieve higher detection accuracy on abrupt topic drift than centroid/covariance shift
2. **H2**: TDA detectors have lower detection delay on gradual drift than statistical baselines
3. **H3**: On "geometry-preserving" drift (same centroid, different topology), TDA significantly outperforms centroid/covariance methods
4. **H4**: TDA FPR remains calibrated on no-drift windows

---

## Proposed Methodology

### Approach
Controlled comparative study using AG News (4 topics) as primary dataset:
1. Generate sentence embeddings using a lightweight sentence-transformer
2. Construct streaming windows with different drift scenarios
3. Compute drift scores per window pair (reference vs test window)
4. Compare detection accuracy, delay, FPR across methods

### Embedding Model
- **all-MiniLM-L6-v2** (384-dim): Fast, CPU-friendly, standard sentence embedding
- Window size: 200 samples (compromise between tractability and TDA stability)

### Drift Scenarios
1. **Abrupt topic drift**: Window = 100% class A → 100% class B
2. **Gradual drift**: Linear mix from 100% A to 100% B over 10 windows  
3. **Geometric drift (cyclical)**: Alternate between mixed classes with same mean but different topology (e.g., ring vs cluster structures in embedding space)
4. **Style drift**: Within same topic, use different AG News sub-categories
5. **No drift (baseline)**: Same class across all windows

### Drift Detectors

**Statistical Baselines:**
- Centroid shift: L2 norm of window mean difference
- Covariance shift: Frobenius norm of covariance matrix difference
- MMD (RBF kernel): Maximum Mean Discrepancy with median heuristic bandwidth
- kNN distance shift: Mean k-nearest-neighbor distance change

**TDA Methods:**
- Persistent entropy (H0): Shannon entropy of persistence lifetimes in Vietoris-Rips filtration
- PHD (PH0 dimension): Log-log slope of connected components vs filtration scale
- Betti curve area (H0+H1): Integrated Betti curve difference between windows
- Persistence diagram Wasserstein distance: W2 distance between consecutive persistence diagrams

### Evaluation Protocol
- Window size: 200 samples
- Reference: first 10 windows of stable data; calibrate threshold at 95th percentile of no-drift scores
- Test windows: drift injected at window 11+
- Seeds: [42, 123, 456] for 3 independent runs
- Metrics: accuracy (AUC), detection delay, FPR at TPR=0.8

### Statistical Analysis
- Bootstrap CI (n=3 seeds) for each metric
- Pairwise comparison: TDA methods vs each baseline
- Wilcoxon signed-rank test for significant performance difference

---

## Timeline
- Phase 1 (Planning): Complete — 30 min
- Phase 2 (Env + Data): 20 min
- Phase 3 (Implementation): 60 min  
- Phase 4 (Experiments): 90 min
- Phase 5 (Analysis + Viz): 30 min
- Phase 6 (Documentation): 30 min

## Expected Outcomes
- TDA wins on geometric drift scenario (H3 confirmed)
- TDA competitive but not dominant on abrupt drift (H1 partially confirmed)
- Computational cost 10-50x higher than statistical baselines (tradeoff analysis needed)

## Potential Challenges
- Ripser may be slow for 200-point clouds in high dimensions → subsample to 100 points
- giotto-tda installation issues → use ripser directly
- Sentence-transformer download → model already cached in datasets/ 

## Success Criteria
- At least one TDA method beats baseline on ≥2 drift scenarios
- FPR calibrated and reported
- Detection delay measured
- Reproducible across ≥3 seeds
- Runtime documented

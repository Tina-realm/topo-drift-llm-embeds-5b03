# Literature Review: Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams

## Research Hypothesis

Persistent-homology-based summaries of embedding point clouds can detect distribution drift in deployed LLM input streams more sensitively and robustly than conventional unsupervised drift detectors (centroid shift, covariance shift, MMD), especially when the drift alters geometric or semantic structure without large changes in low-order statistics.

---

## 1. Research Area Overview

The intersection of **Topological Data Analysis (TDA)** and **LLM embedding drift detection** is a nascent but fast-growing area. Traditional drift detection methods (MMD, KS test, covariance shift, centroid shift) rely on low-order statistics that capture changes in mean or variance. These methods may miss **geometric/structural changes** in high-dimensional embedding manifolds where the distributional structure changes without dramatic shift in first or second moments.

Persistent homology provides a complementary perspective: it tracks multi-scale topological features (connected components, holes, voids) of point clouds as a filtration parameter varies. These features—summarized as **persistence diagrams**, **Betti numbers**, or **persistent entropy**—can capture qualitative structural differences that evade standard statistical tests.

Key research dimensions:
1. **TDA-based characterization of LLM representations** (topology of embedding clouds)
2. **Drift detection from neural embeddings** (baselines: MMD, Fréchet distance, KS)
3. **Statistical testing from persistence diagrams** (hypothesis tests on TDA summaries)
4. **Online/streaming change detection** (computational efficiency on streams)

---

## 2. Key Papers

### 2.1 TDA Applied to LLM Representations and Drift

---

#### Paper 1: Unsupervised Assessment of Landscape Shifts Based on Persistent Entropy and Topological Preservation
- **Authors**: Sebastián Basterrech (DTU)
- **Year**: 2024
- **Source**: KDD Workshop on Drift Detection and Landscape Shifts (arXiv:2410.04183)
- **Key Contribution**: First framework explicitly combining persistent entropy with concept drift detection. Extends classic drift detection with algebraic topology to capture "essential differences" between point clouds that standard statistics miss.
- **Methodology**:
  - Project high-dimensional data into a latent space using topology-preserving dimensionality reduction (SOM preferred over PCA, Kernel PCA)
  - Compute distance matrix between projected points and cluster centroids
  - Build persistence diagrams per chunk; compute **persistent entropy** (Shannon entropy of persistence lifetime distribution)
  - Apply Mann-Whitney U non-parametric test across consecutive chunks; output p-value
  - Drift declared when p-value drops below threshold
- **Datasets**: Synthetic streams from MNIST (3 case studies grouped by topological homology class: no-hole digits {1,3,5,7} vs one-hole digits {0,6,9} vs two-hole digits {8})
- **Baselines Compared**: PCA + persistent entropy, Kernel PCA + persistent entropy vs SOM + persistent entropy
- **Key Results**: SOM (topology-preserving DR) + persistent entropy outperforms non-topology-preserving DR methods; p-value monitoring correctly detects injected topological shifts
- **Code Available**: No explicit GitHub link
- **Relevance**: Directly related; establishes that persistent entropy on dimensionality-reduced embeddings detects topological drift that standard statistics miss. Framework is directly adaptable to LLM text embedding streams.

---

#### Paper 2: Persistent Topological Features in Large Language Models
- **Authors**: Gardinazzi, Viswanathan, Panerai, Ansuini, Cazzaniga, Biagetti
- **Year**: 2024 (ICML 2025 poster)
- **Source**: arXiv:2410.11042
- **Key Contribution**: Applies **zigzag persistence** to track evolution of topological features across LLM layers; introduces *persistence similarity* metric; enables layer pruning without degrading model performance.
- **Methodology**:
  - Compute Vietoris-Rips filtration on activations at each layer
  - Track birth/death of topological features across layers using zigzag persistence
  - Define persistence similarity to compare topological evolution across consecutive layers
- **Datasets**: Internal LLM activations (various models)
- **Key Results**: Topological behavior is consistent across architectures; redundant layers have high persistence similarity; prunable
- **Code Available**: Yes — https://github.com/RitAreaSciencePark/ZigZagLLMs
- **Relevance**: Establishes universality of topological structure in LLM embeddings; motivates using TDA across different streaming windows as a drift signal.

---

#### Paper 3: Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion
- **Authors**: Dongjun Wei, Minjia Mao, Xiao Fang, Michael Chau (HKU, Univ. of Delaware)
- **Year**: 2025 (arXiv:2504.02873)
- **Key Contribution**: Extension of PHD (Persistent Homology Dimension) from Tulchinskii et al. 2024 to short texts via off-topic content insertion (OCI). Directly exploits the empirical finding that **LLM-generated text has lower PHD than human-written text**.
- **Methodology**:
  - Represent text as point cloud: n tokens → n points in R^d (d=768 for RoBERTa)
  - Compute PH0 via minimal spanning tree (MST); estimate PHD by log-log regression of E^0_α(W) vs. n_i (number of sampled points)
  - LLM-generated texts exhibit **lower PHD** (more connected embeddings)
  - For short texts: insert off-topic prefix before computing PHD
- **Datasets**: Public LLM-generated text datasets; custom generated pairs
- **Baselines**: PHD (Tulchinskii 2024), DetectGPT, Fast-DetectGPT
- **Key Results**: Short-PHD AUC 0.793–0.830 vs 0.632–0.653 for base PHD on 50-token texts
- **Code Available**: Yes — https://github.com/djwei96/ShortPHD
- **Relevance**: Demonstrates PHD as a practical, computable scalar per embedding window. The inverse relationship (LLM-generated text → lower PHD) is a specific form of semantic drift signal. The MST-based PH0 estimation is computationally tractable for streaming contexts.

---

#### Paper 4: Detecting Out-of-Distribution Text Using Topological Features of Transformer-Based Language Models
- **Authors**: Not specified in search (arXiv:2311.13102)
- **Year**: 2023
- **Source**: arXiv:2311.13102
- **Key Contribution**: TDA on attention maps and hidden states of BERT-family models to detect OOD text
- **Methodology**: Extracts topological features (Betti numbers, persistence diagrams) from attention maps; compares in-distribution vs OOD using these features
- **Relevance**: OOD detection as a special case of drift detection; establishes that TDA features from transformer internals are discriminative.

---

#### Paper 5: Zero-Shot Embedding Drift Detection (ZEDD)
- **Authors**: Sekar, Agarwal, Sharma, Tanaka, Zhang, Damerla, Zhu (Algoverse AI Research)
- **Year**: 2026 (NeurIPS 2025, arXiv:2601.12359)
- **Key Contribution**: Uses **cosine similarity** of embeddings between benign and suspect inputs to detect prompt injection; achieves >93% accuracy with <3% FPR across Llama 3, Qwen 2, Mistral.
- **Methodology**:
  - Fine-tune encoder once; compute embedding drift via cosine similarity between benign reference and candidate inputs
  - Flag using GMM + KDE on the drift distribution
- **Note**: This is specifically about prompt injection detection (a special case of semantic drift), not general distribution drift monitoring. The approach is lightweight but limited to paired comparisons.
- **Relevance**: Shows that embedding-space semantic drift is a useful and practical signal; our TDA approach can be seen as a more principled geometric generalization.

---

#### Paper 6: Unveiling Topological Structures from Language: A Comprehensive Survey of TDA Applications in NLP
- **Year**: 2024–2025 (arXiv:2411.10298)
- **Key Contribution**: Comprehensive survey of TDA methods applied to NLP, including persistent homology on sentence embeddings, text structure analysis, and LLM-generated text detection.
- **Relevance**: Background and landscape of TDA-NLP applications; confirms that TDA features on text embeddings are discriminative across multiple tasks.

---

### 2.2 Conventional Drift Detection Baselines

---

#### Paper 7: Uncovering Drift in Textual Data: An Unsupervised Method for Detecting and Mitigating Drift in Machine Learning Models
- **Authors**: Khaki, Aditya, Karnin, Ma, Pan, Chandrashekar (Amazon)
- **Year**: 2023 (arXiv:2309.03831)
- **Key Contribution**: Production-scale MMD-based drift detection on BERT embeddings; 3-year longitudinal evaluation correlating MMD drift with model AUC degradation.
- **Methodology**:
  - Encode text with BERT; compute average embedding per mini-batch
  - Bootstrap MMD test (kernel: RBF) between reference (training) and production distributions
  - Root cause analysis: identify highest-drift mini-batches; retrain on those
- **Datasets**: Amazon production binary classification (800K train, 3 years production data)
- **Key Results**: MMD vs BCE correlation 76.9%; MMD vs AUC correlation -65.2%; clear inverse relationship between drift and model performance
- **Relevance**: **Primary MMD baseline**: This is the industrial gold standard for text embedding drift detection. Our TDA approach should be compared against MMD + BERT embeddings. Key limitation: MMD captures distributional differences in RKHS but not geometric/topological structure changes.

---

#### Paper 8: Detecting Covariate Drift in Text Data Using Document Embeddings and Dimensionality Reduction
- **Authors**: Sodar et al. (arXiv:2309.10000)
- **Year**: 2023
- **Key Contribution**: Systematic comparison of TF-IDF+LSA, Doc2Vec, BERT embeddings × PCA dimensionality reduction × KS / MMD drift detection methods.
- **Methodology**: Evaluate covariate drift detection accuracy across embedding/DR/test combinations on synthetic and real text domain shift data
- **Key Results**: BERT + PCA + MMD generally best; Doc2Vec competitive; raw TF-IDF poorest
- **Relevance**: Establishes BERT embeddings as the standard feature for text drift detection; justifies our use of sentence transformer embeddings.

---

#### Paper 9: Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time (DriftLens)
- **Authors**: Greco et al.
- **Year**: 2024 (arXiv:2406.17813, published TKDE)
- **Key Contribution**: DriftLens — per-label Fréchet Distance on embedding distributions for unsupervised drift detection.
- **Methodology**:
  - Reference phase: compute per-label Gaussian distribution of [CLS] token embeddings from BERT/DistilBERT
  - Monitoring: compare new window Fréchet Distance (Gaussian assumption) against reference; threshold from historical data
  - Handles gradual and abrupt drift
- **Datasets**: Text classification benchmarks; image classification
- **Code Available**: Yes — https://github.com/grecosalvatore/drift-lens (cloned)
- **Relevance**: **Primary baseline for embedding drift detection with labeling**. Tests Fréchet distance (Gaussian approximation of Wasserstein-2), a mean+covariance statistic. Our TDA approach captures structure *beyond* mean and covariance.

---

#### Paper 10: Maximum Mean Discrepancy on Exponential Windows for Online Change Detection (MMDEW)
- **Authors**: Kalinke, Heyden, Gntuni, Fouché, Böhm (KIT)
- **Year**: 2022–2025 (arXiv:2205.12706)
- **Key Contribution**: MMDEW — online streaming MMD-based change detector with O(log²t) runtime and O(log t) memory via exponential windows.
- **Methodology**:
  - Maintain exponential windows of observations; approximate MMD between window pairs
  - Two-sample test: if MMD between consecutive windows exceeds threshold, declare change
  - Theoretical guarantees on runtime and memory complexity
- **Baselines Compared**: ADWINK, WATCH, Scan B-Statistics, NEWMA, D3, IBDD
- **Key Results**: MMDEW outperforms state-of-the-art on 4/5 benchmark data streams (F1-score)
- **Code Available**: Yes — https://github.com/FlopsKa/mmdew-change-detector (cloned)
- **Relevance**: **Key baseline for streaming MMD drift detection**. Our TDA-based streaming detector should be compared against MMDEW for computational efficiency and detection power.

---

#### Paper 11: Online Drift Detection with Maximum Concept Discrepancy (MCD-DD)
- **Year**: 2024 (arXiv:2407.05375, KDD 2024)
- **Key Contribution**: Drift detection without labels or error rates; handles high-dimensional data with irregular distribution shifts.
- **Relevance**: Modern MMD-inspired baseline; another comparison point.

---

### 2.3 Statistical Testing for TDA Features

---

#### Paper 12: Testing Homological Equivalence Using Betti Numbers: Probabilistic Properties
- **Year**: 2022 (arXiv:2211.13959)
- **Key Contribution**: Formal two-sample test using Betti numbers from Čech/Vietoris-Rips complexes. Establishes consistency of tests based on Betti numbers in critical and supercritical regimes.
- **Methodology**:
  - One-sample test: is the distribution homologically equivalent to a reference?
  - Two-sample test: are two unknown distributions homologically equivalent?
  - Test statistics: Betti numbers at specific filtration radii; compared via permutation tests
- **Relevance**: **Foundational theoretical grounding** for our research. Provides statistical rigor for using Betti numbers as drift test statistics.

---

### 2.4 TDA Software / Tools Papers

---

#### Paper 13: Ripser.py: A Lean Persistent Homology Library for Python
- **Year**: 2019 (arXiv:1908.02751)
- **Key Contribution**: Python interface to blazing-fast C++ Ripser engine for persistent (co)homology computation.
- **Relevance**: Primary computational tool for TDA experiments.

---

#### Paper 14: giotto-tda: A Topological Data Analysis Toolkit for Machine Learning
- **Year**: 2020 (arXiv:2004.02551)
- **Key Contribution**: scikit-learn-compatible TDA pipeline; PersistenceEntropy, BettiCurve, HeatKernel vectorizations; full ML integration.
- **Relevance**: Primary library for TDA feature extraction and ML integration.

---

### 2.5 User-Specified Papers (Mandatory Deep Read)

The following three papers were specified by the user and read in full (all chunks). They cover topics adjacent to the core research area.

---

#### Paper 15: Matryoshka Representation Learning (MRL)
- **Authors**: Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi
- **Affiliation**: University of Washington, Google Research, Harvard University
- **Year**: 2022 (NeurIPS 2022)
- **Source**: arXiv:2205.13147
- **GitHub**: https://github.com/RAIVNLab/MRL
- **Key Contribution**: MRL trains a single embedding model to simultaneously produce representations useful at O(log d) nested granularities. The first m dimensions of an MRL embedding are independently meaningful for any m in the nesting set M = {8, 16, 32, 64, 128, 256, 512, 1024, 2048} (for d=2048).
- **Core Mechanism**:
  - Multi-granularity training: the classification loss is summed over each nesting dimension with weighting factors c_m
  - At inference, truncate to any m ∈ M to get an m-dimensional embedding with near-optimal quality for that dimension
  - MRL-E (efficient variant): share weights across all dimension-specific classifiers to reduce parameter count
  - Adaptive retrieval: use a cascade — coarse retrieval with small m, then re-rank shortlist with full d — achieving ~37 expected dimensions for 76.3% ImageNet-1K accuracy
- **Results**:
  - 14x smaller embedding (128 vs 2048 dims) with same classification accuracy on ImageNet-1K
  - 14x real-world speedup on approximate nearest-neighbor retrieval (HNSW)
  - Zero-shot transfer: MRL embeddings maintain transfer learning performance across scales
  - Results hold across ResNet50, ViT-B/16, ALIGN (vision-language), BERT (NLP)
- **PyTorch Implementation** (from appendix):
  - `class Matryoshka_CE_Loss(nn.Module)`: wraps cross-entropy summed over nesting dimensions
  - `class MRL_Linear_Layer(nn.Module)`: single linear layer serving all nesting dims simultaneously
- **Relevance to Research**:
  - MRL embeddings are ideal for multi-resolution TDA drift detection: compute persistent entropy at dimension {32, 64, 128, 256} from the same embedding vector
  - Coarse-to-fine drift detection: use small m for fast pre-screening, then larger m for confirmation — analogous to MRL adaptive retrieval
  - Practical: most modern sentence transformers (e.g., nomic-embed-text-v1.5) already use MRL; our drift detector can leverage this for free
  - Key insight: topology at coarse granularity (m=32) may capture high-level semantic drift; topology at fine granularity (m=512) captures subtle structural drift

---

#### Paper 16: zeus — Ensemble Slice Sampling for Efficient Bayesian Parameter Inference
- **Authors**: Minas Karamanis, Florian Beutler, John A. Peacock
- **Affiliation**: University of Edinburgh
- **Year**: 2021 (Monthly Notices of the Royal Astronomical Society)
- **Source**: arXiv:2105.03468
- **GitHub**: https://github.com/minaskar/zeus
- **Key Contribution**: zeus implements Ensemble Slice Sampling (ESS), which combines the ensemble MCMC paradigm (multiple interacting walkers) with slice sampling (non-rejection sampling). This eliminates manual step-size tuning and handles complex posteriors efficiently.
- **Core Mechanism**:
  - Ensemble of N walkers (minimum N = 2D, recommended N = 2-4D where D = dimensionality)
  - Slice sampling along directions determined by complementary ensemble (other walkers)
  - Move strategies: Differential (project along ensemble directions), Gaussian, Global (mixture of all walkers), KDE (kernel density estimate), Random
  - No accept/reject step → no tuning; automatically adapts to local geometry
  - Convergence diagnostics: integrated autocorrelation time (IAT)
- **Results**:
  - 9x efficiency gain over emcee/AIES (Affine Invariant Ensemble Sampler) on cosmological parameter inference
  - 29x efficiency gain on exoplanet radial velocity fitting (multimodal, non-linear correlations)
  - Handles hard boundaries, heavy tails, and near-degenerate posteriors
- **API**:
  ```python
  sampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob_fn)
  sampler.run_mcmc(start, nsteps)
  chain = sampler.get_chain(flat=True)
  ```
- **Relevance to Research**:
  - Marginally relevant: ESS could be used for Bayesian calibration of TDA detector hyperparameters (filtration radius, window size, entropy threshold) when the posterior landscape over hyperparameters is complex or multimodal
  - More practically: the ensemble-based approach is analogous to the ensemble of sliding windows used in streaming drift detection — each window acts like a walker sampling the embedding distribution space
  - Direct applicability is limited; the paper is primarily an astrophysics/statistics tool

---

#### Paper 17: Quantum Fluctuations Hinder Finite-time Information Erasure Near the Landauer Limit
- **Authors**: Harry J. D. Miller, Giacomo Guarnieri, Mark T. Mitchison, John Goold
- **Affiliation**: University of Manchester; Trinity College Dublin
- **Year**: 2020 (Physical Review Letters 123, 230603)
- **Source**: arXiv:2007.01882
- **Key Contribution**: Rigorous proof that quantum coherence generates non-negative contributions to ALL statistical cumulants of the dissipated heat during bit erasure. The heat distribution in quantum protocols is fundamentally non-Gaussian with extreme outlier events that cannot occur classically.
- **Core Result**: The cumulant generating function (CGF) decomposes as:
  - K_q(u) = K^d_q(u) + K^c_q(u)
  - K^d_q(u): classical (diagonal) contribution — depends only on energy populations
  - K^c_q(u): coherent (quantum) contribution — always ≥ 0 for all cumulants
  - K^c_q(u) ≈ -ukBT ∫₀^1 dt Ṡ_{1-ukBT}(ρ̂_t || ρ̄̂_t) where S_α is the Rényi divergence
- **Mathematical Framework**:
  - Adiabatic Lindblad master equation under slow-driving (Born-Markov, secular, adiabatic approximations)
  - Power operator Ḣ_t = Ḣ^d_t (diagonal, classical) + Ḣ^c_t (coherent, quantum)
  - Proof of positive semidefiniteness of quantum CGF via Rényi divergence monotonicity and quantum detailed balance
  - Appendix B: Full derivation of CGF decomposition using perturbation expansion in slow-driving parameter ε
  - Appendix C: Proof that all cumulants of K^c_q(u) are monotonically non-decreasing (via dual Lindblad generator, trace functional M^(u)_t, and time-translational symmetry)
  - Appendix D: Explicit solution for damped two-level system — Bloch vector dynamics, analytical CGF expressions for both classical and quantum parts
  - Appendix E: Monte Carlo quantum-jump simulation with fourth-order Runge-Kutta for heat distribution in Fig. 3
- **Physical Interpretation**:
  - Quantum protocols (non-zero mixing angle θ_t) produce extreme heat dissipation outliers (up to ~30x Landauer limit) that are absent in classical protocols (θ_t = 0)
  - Rare events include negative heat transfer q < 0, which is impossible classically
  - Bulk of distribution converges to Gaussian near Landauer bound as protocol time τ → ∞
  - Quantum coherence: thermodynamic cost is not just the mean (which equals Landauer limit at ε→0) but the entire distribution is broadened and skewed
- **Relevance to Research**:
  - Conceptually: The Landauer limit (kBT ln 2 per bit erased) provides a thermodynamic analogy for "forgetting" reference data in a streaming drift detector. A detector that never discards old data has zero information erasure cost but unbounded memory; one that resets a reference window incurs an information cost.
  - Mathematically: Very limited direct applicability. The Lindblad formalism, quantum coherence, and thermodynamic fluctuation theory are unrelated to TDA or LLM embedding analysis.
  - The CGF decomposition into classical + quantum parts has a formal analogy to separating smooth (centroid/covariance) drift from topological (geometric) drift — the "quantum coherent" component being the part conventional detectors miss — but this analogy is loose and not actionable.

---

## 3. Common Methodologies

| Approach | Papers | Notes |
|---|---|---|
| Persistent entropy on point clouds | 1, Survey | Single scalar summary of persistence diagram |
| PHD (Persistent Homology Dimension) | 3 | MST-based PH0 estimation; tractable for streaming |
| Betti numbers as test statistics | 12 | Formal hypothesis test; requires filtration radius choice |
| Persistence diagrams + Wasserstein/bottleneck distance | General TDA | Metric between diagrams; stable but expensive |
| MMD two-sample test on embeddings | 7, 8, 10 | Gold standard baseline; kernel-based |
| Fréchet Distance on embeddings | 9 (DriftLens) | Gaussian approximation; mean+cov only |
| Cosine similarity drift | 5 (ZEDD) | Pairwise; lightweight but limited |
| SOM latent projection + persistent entropy | 1 | Topology-preserving DR crucial |

---

## 4. Standard Baselines in the Field

| Baseline | What It Measures | Limitation |
|---|---|---|
| **Centroid shift** (mean difference) | Change in first moment | Misses structural/geometric changes |
| **Covariance shift** | Change in second moment | Misses non-Gaussian structural changes |
| **MMD** (RBF kernel) | RKHS distance between distributions | Sensitive to kernel choice; misses topology |
| **Fréchet Distance** | Gaussian approximation of W2 | Assumes Gaussian; misses multi-modal geometry |
| **KS test** (per-feature) | Marginal distribution shift per feature | Ignores joint structure |
| **DriftLens** (Fréchet + per-label) | Per-label distributional drift | Requires label information |

---

## 5. Evaluation Metrics Used in Literature

| Metric | Description |
|---|---|
| **AUC** | Area under ROC curve for drift detection (binary: drift/no-drift) |
| **F1-score** | Precision × Recall for change point detection |
| **Detection delay** | Samples elapsed before drift is flagged |
| **False Positive Rate (FPR)** | Rate of false drift alarms |
| **Power** | Probability of correctly detecting a true drift |
| **p-value** | Direct output of statistical tests |

---

## 6. Datasets in the Literature

| Dataset | Used In | Task | Size | Notes |
|---|---|---|---|---|
| **AG News** | Common NLP benchmark | Topic classification (4 classes: World/Sports/Business/Sci-Tech) | 120K train / 7.6K test | Ideal for cross-domain drift experiments |
| **20 Newsgroups** | Standard multi-domain NLP | Topic classification (20 categories) | ~18K total | Classic multi-domain drift benchmark |
| **MNIST** | Basterrech 2024 | Topological group separation | 70K | Controlled topological drift (digits grouped by #holes) |
| **DBpedia14** | Entity classification | 14-domain entity classification | 560K train | Large-scale multi-domain |
| **Amazon production data** | Khaki et al. 2023 | Binary text classification | 800K+ | 3-year production drift |

---

## 7. Key Research Gaps and Opportunities

1. **No direct comparison of PHD/Betti-based drift detectors vs MMD on text embedding streams**: The literature applies TDA to LLM analysis but not as a systematic drift detector for streaming LLM inputs.

2. **Computational tractability gap**: VR complexes are O(n³) to compute in the worst case. PHD via MST is O(n log n) — this makes PHD the most viable streaming TDA feature, but needs evaluation.

3. **Geometry-preserving vs statistics-only drift**: The key hypothesis (topological drift without statistical drift) is not yet empirically validated for text embeddings specifically. Basterrech 2024 showed this for MNIST images; our work extends to LLM embeddings.

4. **Online/streaming TDA**: Most TDA drift detection works in batch mode. Streaming persistent entropy is an open research direction.

5. **Multi-domain text benchmarks for drift**: Most drift detection papers use synthetic or proprietary data. AG News and 20 Newsgroups provide realistic domain-shift scenarios.

---

## 8. Recommendations for Our Experiment

### Recommended Datasets
1. **AG News** (primary): 4 topically distinct categories; use inter-category drift to simulate domain shift. Sub-experiments: gradual vs abrupt drift (gradually mix category proportions vs sudden switch).
2. **20 Newsgroups** (secondary): 20 finer-grained topics; allows more nuanced drift scenarios.
3. **DBpedia14** (optional): Very large scale with 14 domains; use subset.

### Recommended Embedding Models
- **all-MiniLM-L6-v2** (sentence-transformers): Fast, 384-dim embeddings; standard for drift experiments
- **all-mpnet-base-v2** (sentence-transformers): Higher quality, 768-dim; for deeper experiments

### Recommended TDA Features (our proposed methods)
1. **Persistent entropy (H0)**: Shannon entropy of PH0 lifetime distribution — single scalar, tractable
2. **PHD (PH0)**: MST-based estimation of intrinsic dimension — tractable, matches Short-PHD literature
3. **Betti curve (H0, H1)**: Number of connected components and loops as function of filtration radius
4. **Persistence landscape norm**: Functional summary of persistence diagram

### Recommended Baselines
1. **Centroid shift**: L2 distance between mean embeddings of reference and test windows
2. **Covariance shift**: Frobenius norm difference between covariance matrices
3. **MMD (RBF kernel)**: Standard kernel two-sample test using scikit-learn or Frouros
4. **MMDEW**: Online streaming MMD — from cloned repo
5. **DriftLens (Fréchet Distance)**: Fréchet distance between reference and test window Gaussians

### Recommended Metrics
- **AUC** for drift detection (primary)
- **Detection delay** for streaming scenarios
- **FPR at fixed TPR** (e.g., TPR=0.8)
- **Power vs. drift magnitude** curves

### Experimental Design
1. **Controlled abrupt drift**: Stream from Category A → Category B (abrupt switch)
2. **Gradual drift**: Gradually increase proportion of Category B
3. **Geometric-only drift**: Create synthetic drift that alters cluster geometry without changing centroid (rotation/scaling of embedding clusters)
4. **Topological drift**: Create scenarios with topological changes (e.g., from unimodal to bimodal embedding clusters)
5. **Comparison**: For each scenario, compare TDA features vs baselines on AUC + detection delay

### Methodological Considerations
- **Window size sensitivity**: Both TDA and MMD methods require sufficient samples per window; test chunk sizes of 50, 100, 250 following Basterrech 2024
- **Computational cost**: PHD via MST is O(n log n); Ripser VR for H1/H2 is more expensive — profile carefully
- **Statistical test choice**: Mann-Whitney U (non-parametric, distribution-free) is appropriate for persistent entropy sequences; permutation test for Betti numbers
- **Reference window**: Use sliding window comparison (consecutive windows) vs. fixed reference (training data); both regimes should be tested

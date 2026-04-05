# Downloaded Papers

Research area: Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams

## Core TDA + Drift Detection Papers

### 1. Landscape Shifts with Persistent Entropy (PRIMARY - MOST RELEVANT)
- **File**: `2410.04183_landscape_shifts_persistent_entropy_tda.pdf`
- **Authors**: Sebastián Basterrech (DTU)
- **Year**: 2024
- **Venue**: KDD Workshop on Drift Detection and Landscape Shifts
- **arXiv**: 2410.04183
- **Why relevant**: Directly proposes persistent entropy for concept drift detection in streaming data. Uses SOM for topology-preserving dimensionality reduction. Mann-Whitney U test on per-chunk persistent entropy values. **Framework is directly applicable to LLM embedding streams.**

### 2. Persistent Topological Features in LLMs
- **File**: `2410.11042_persistent_topological_features_llm.pdf`
- **Authors**: Gardinazzi, Viswanathan, Panerai, Ansuini, Cazzaniga, Biagetti
- **Year**: 2024 (ICML 2025)
- **arXiv**: 2410.11042
- **GitHub**: https://github.com/RitAreaSciencePark/ZigZagLLMs
- **Why relevant**: Zigzag persistence across LLM layers; establishes topological structure universality in LLMs.

### 3. Short-PHD: LLM Text Detection via PHD
- **File**: `2504.02873_short_phd_llm_text_detection_tda.pdf`
- **Authors**: Wei, Mao, Fang, Chau (HKU, UDel)
- **Year**: 2025
- **arXiv**: 2504.02873
- **GitHub**: https://github.com/djwei96/ShortPHD
- **Why relevant**: PHD (persistent homology dimension) via MST is tractable for streaming; LLM-generated text has lower PHD (more connected embeddings). Demonstrates PHD as practical embedding topology metric.

### 4. TDA/NLP Survey
- **File**: `2411.10298_topological_structures_nlp_survey.pdf`
- **Year**: 2024–2025
- **arXiv**: 2411.10298
- **Why relevant**: Comprehensive survey of TDA methods in NLP; background and context.

### 5. OOD Detection with TDA on Transformers
- **File**: `2311.13102_detecting_ood_text_topological_features.pdf`
- **Year**: 2023
- **arXiv**: 2311.13102
- **Why relevant**: TDA features from BERT attention maps for OOD detection; closely related to drift detection.

---

## Drift Detection Baseline Papers

### 6. Zero-Shot Embedding Drift Detection (ZEDD)
- **File**: `2601.12359_zero_shot_embedding_drift_detection.pdf`
- **Authors**: Sekar et al. (Algoverse AI Research)
- **Year**: 2026 (NeurIPS 2025)
- **arXiv**: 2601.12359
- **GitHub**: https://github.com/AnirudhSekar/ZEDD/
- **Why relevant**: Cosine-similarity-based embedding drift detection; lightweight baseline; >93% accuracy for prompt injection.

### 7. Uncovering Drift in Textual Data (Amazon MMD)
- **File**: `2309.03831_uncovering_drift_textual_data_unsupervised.pdf`
- **Authors**: Khaki, Aditya, Karnin, Ma, Pan, Chandrashekar (Amazon)
- **Year**: 2023
- **arXiv**: 2309.03831
- **Why relevant**: Production-scale MMD on BERT embeddings; 3-year longitudinal study; MMD vs AUC correlation -65%. **Primary MMD baseline.**

### 8. Detecting Covariate Drift in Text
- **File**: `2309.10000_detecting_covariate_drift_text_embeddings.pdf`
- **Year**: 2023
- **arXiv**: 2309.10000
- **Why relevant**: Systematic comparison of embedding methods × DR × KS/MMD for text drift detection.

### 9. DriftLens: Unsupervised Concept Drift Detection
- **File**: `2406.17813_driftlens_unsupervised_concept_drift.pdf`
- **Authors**: Greco et al.
- **Year**: 2024 (IEEE TKDE)
- **arXiv**: 2406.17813
- **GitHub**: https://github.com/grecosalvatore/drift-lens
- **Why relevant**: Fréchet Distance on per-label embeddings; best prior embedding drift detection. **Key baseline.**

### 10. MMDEW: MMD on Exponential Windows for Online Change Detection
- **File**: `2205.12706_mmd_exponential_windows_online_change.pdf`
- **Authors**: Kalinke, Heyden, Gntuni, Fouché, Böhm (KIT)
- **Year**: 2022–2025
- **arXiv**: 2205.12706
- **GitHub**: https://github.com/FlopsKa/mmdew-change-detector
- **Why relevant**: Efficient online streaming MMD detector; O(log²t) runtime; outperforms ADWINK, NEWMA, D3 on benchmarks. **Key streaming baseline.**

### 11. Online Drift Detection with Maximum Concept Discrepancy
- **File**: `2407.05375_online_drift_detection_maximum_concept_discrepancy.pdf`
- **Year**: 2024 (KDD 2024)
- **arXiv**: 2407.05375
- **Why relevant**: MMD-inspired drift detection without labels; handles irregular distribution shifts.

### 12. Partial Wasserstein and MMD for Drift Detection
- **File**: `2106.12893_partial_wasserstein_mmd_drift_detection.pdf`
- **Year**: 2021
- **arXiv**: 2106.12893
- **Why relevant**: Comparison of MMD vs Wasserstein distance for drift detection; establishes MMD robustness.

### 13. Concept Drift in Text Streams: Comprehensive Review
- **File**: `2312.02901_concept_drift_text_stream_review.pdf`
- **Year**: 2023
- **arXiv**: 2312.02901
- **Why relevant**: Survey of text stream drift detection methods; background.

### 14. Beyond Statistical Similarity: Semantic Drift
- **File**: `2309.16427_beyond_statistical_similarity_semantic_drift.pdf`
- **Year**: 2023
- **arXiv**: 2309.16427
- **Why relevant**: Analysis of semantic drift beyond statistical measures; supports hypothesis motivation.

---

## Statistical Testing for TDA

### 15. Testing Homological Equivalence Using Betti Numbers
- **File**: `2211.13959_testing_homological_equivalence_betti_numbers.pdf`
- **Year**: 2022
- **arXiv**: 2211.13959
- **Why relevant**: Formal two-sample tests using Betti numbers from VR complexes; theoretical foundation for TDA-based drift testing.

---

## TDA Software / Tool Papers

### 16. Ripser.py: Lean Persistent Homology Library
- **File**: `1908.02751_ripser_py_persistent_homology.pdf`
- **Year**: 2019
- **arXiv**: 1908.02751
- **GitHub**: https://github.com/scikit-tda/ripser.py
- **Why relevant**: Primary TDA computation library; fast C++ backend.

### 17. giotto-tda: TDA Toolkit for Machine Learning
- **File**: `2004.02551_giotto_tda_toolkit.pdf`
- **Year**: 2020 (JMLR)
- **arXiv**: 2004.02551
- **GitHub**: https://github.com/giotto-ai/giotto-tda
- **Why relevant**: sklearn-compatible TDA ML pipeline; PersistenceEntropy, BettiCurve, PersistenceLandscape.

### 18. giotto-ph: High-Performance Persistent Homology
- **File**: `2107.05412_giotto_ph_high_performance.pdf`
- **Year**: 2021
- **arXiv**: 2107.05412
- **Why relevant**: Fastest VR persistent homology computation; needed for H1 on larger windows.

---

## User-Specified Papers (Deep Read)

The following three papers were specified by the user for mandatory deep reading. They were read in full (all chunks).

### 19. Matryoshka Representation Learning (MRL)
- **File**: `2205.13147_user_specified_paper1.pdf`
- **Authors**: Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi (University of Washington, Google Research, Harvard University)
- **Year**: 2022 (NeurIPS 2022)
- **Venue**: NeurIPS 2022
- **arXiv**: 2205.13147
- **GitHub**: https://github.com/RAIVNLab/MRL
- **Key Contribution**: Matryoshka Representation Learning (MRL) encodes information at O(log d) nested granularities simultaneously in a single embedding vector. This is achieved by training with a loss function that ensures the first m dimensions are independently useful for any m in a set of nesting dimensions M = {8, 16, 32, ..., d}.
- **Technical Details**:
  - Single embedding model produces representations useful at all scales simultaneously
  - MRL loss: sum of classification losses at each nesting dimension
  - MRL-E (efficient): weight-tying across dimension-specific classifiers
  - Nesting dimensions tested: M = {8, 16, 32, 64, 128, 256, 512, 1024, 2048} for ResNet50 (d=2048)
  - Adaptive retrieval cascade: achieves ~37 expected dimensions for 76.3% ImageNet-1K accuracy
  - 14x smaller embedding with same retrieval accuracy; 14x real-world speedup (HNSW index)
  - PyTorch `Matryoshka_CE_Loss` and `MRL_Linear_Layer` classes provided in appendix
- **Evaluated On**: ResNet50, ViT-B/16, ALIGN, BERT across vision and NLP modalities
- **Key Results**: 14x size reduction with no accuracy loss; adaptive classification with cascades nearly matches oracle optimal-dimension performance; zero-shot transfer maintained across embedding scales
- **Relevance to Research**: MRL is directly relevant to embedding drift detection with nested embeddings. Multi-scale TDA features could be computed at each nesting dimension; drift detected earlier at coarse granularity and refined at finer dimensions. Adaptive retrieval analogy: use coarse-granularity TDA for fast pre-screening, then fine-granularity for confirmation.

### 20. zeus: Ensemble Slice Sampling for Bayesian Parameter Inference
- **File**: `2105.03468_user_specified_paper2.pdf`
- **Authors**: Minas Karamanis, Florian Beutler, John A. Peacock (University of Edinburgh)
- **Year**: 2021 (Monthly Notices of the Royal Astronomical Society)
- **Venue**: MNRAS 2021
- **arXiv**: 2105.03468
- **GitHub**: https://github.com/minaskar/zeus
- **Key Contribution**: zeus implements Ensemble Slice Sampling (ESS) — a non-rejection MCMC method that combines ensemble sampling with slice sampling. Achieves 9x and 29x efficiency gains over emcee/AIES in cosmological applications.
- **Technical Details**:
  - Ensemble of at least 2D walkers (recommended 2-4D where D = parameter dimensionality)
  - Moves: Differential (global correlation), Gaussian, Global, KDE, Random
  - Slice sampling: no accept/reject → no tuning of step size; automatically adapts to local geometry
  - Handles: non-linear correlations, multimodal distributions, heavy tails, hard boundaries
  - Python API: `zeus.EnsembleSampler(nwalkers, ndim, log_prob_fn)`
- **Relevance to Research**: Marginally relevant — could be used for Bayesian calibration of TDA-based drift detector hyperparameters (threshold, window size, filtration parameters). ESS is useful when the posterior over detector hyperparameters is multimodal or has complex geometry. However, this paper is primarily an astrophysics/statistics tool and its direct relevance to LLM embedding drift detection is limited.

### 21. Quantum Fluctuations and Information Erasure Near the Landauer Limit
- **File**: `2007.01882_user_specified_paper3.pdf`
- **Authors**: Harry J. D. Miller, Giacomo Guarnieri, Mark T. Mitchison, John Goold (University of Manchester, Trinity College Dublin)
- **Year**: 2020 (Physical Review Letters)
- **Venue**: PRL 2020
- **arXiv**: 2007.01882
- **Key Contribution**: Proves rigorously that quantum coherence in open quantum systems generates non-negative contributions to ALL cumulants of the dissipated heat distribution during information erasure near the Landauer limit. The cumulant generating function (CGF) decomposes as K_q(u) = K^d_q(u) + K^c_q(u), where K^c_q(u) ≥ 0 for all u and all cumulants.
- **Technical Details**:
  - Framework: adiabatic Lindblad master equation under slow-driving approximation
  - Diagonal (classical) CGF: K^d_q(u) = -(u² - βu)∫ dt ∫₀^∞ dν cov_t(Ḣ^d_t(ν), Ḣ^d_t)
  - Coherent (quantum) CGF: K^c_q(u) ≈ -ukBT ∫₀^1 dt Ṡ_{1-ukBT}(ρ̂_t || ρ̄̂_t(s))|_{s=t}
  - Proof that all cumulants of K^c_q(u) are non-negative via quantum detailed balance and Rényi divergences
  - Explicit simulation: damped two-level system with adiabatic Lindblad equation; Bloch vector dynamics; Monte Carlo quantum-jump trajectories
  - Results: quantum protocols produce extreme heat dissipation outliers (~30x Landauer limit) absent in classical protocols; heat distributions are non-Gaussian; outliers include rare negative-heat events impossible classically
- **Appendices**: B: CGF decomposition proof; C: proof of monotonic cumulant increase; D: two-level system analytical solution; E: Monte Carlo quantum-jump simulation details
- **Relevance to Research**: Very marginally relevant — this paper is quantum thermodynamics/information theory. The Landauer limit concept (minimum work required for information erasure = kBT ln 2) has a conceptual analogy to the cost of "forgetting" past information in a streaming drift detector, but the mathematical machinery is entirely unrelated to TDA or LLM embeddings. Included per user specification; not directly applicable to the core research.

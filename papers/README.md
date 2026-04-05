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

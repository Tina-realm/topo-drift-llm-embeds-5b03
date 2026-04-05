# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project:
**Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams**

All resources are organized in:
- `papers/` — Downloaded PDFs
- `datasets/` — Downloaded datasets (data files excluded from git; see datasets/README.md for download instructions)
- `code/` — Cloned code repositories

---

## Papers

**Total papers downloaded: 18**

| # | Title | Authors | Year | File | Category |
|---|---|---|---|---|---|
| 1 | Unsupervised Assessment of Landscape Shifts Based on Persistent Entropy and Topological Preservation | Basterrech | 2024 | `2410.04183_landscape_shifts_persistent_entropy_tda.pdf` | **Core TDA Drift** |
| 2 | Persistent Topological Features in Large Language Models | Gardinazzi et al. | 2024 | `2410.11042_persistent_topological_features_llm.pdf` | TDA + LLM |
| 3 | Short-PHD: Detecting Short LLM-generated Text with TDA After Off-topic Content Insertion | Wei, Mao et al. | 2025 | `2504.02873_short_phd_llm_text_detection_tda.pdf` | **Core TDA** |
| 4 | Unveiling Topological Structures from Language: Survey of TDA Applications in NLP | (Survey) | 2024 | `2411.10298_topological_structures_nlp_survey.pdf` | Survey |
| 5 | Detecting Out-of-Distribution Text Using Topological Features of Transformer LMs | (OOD TDA) | 2023 | `2311.13102_detecting_ood_text_topological_features.pdf` | TDA + NLP |
| 6 | Zero-Shot Embedding Drift Detection (ZEDD) | Sekar et al. | 2026 | `2601.12359_zero_shot_embedding_drift_detection.pdf` | Embedding Drift |
| 7 | Uncovering Drift in Textual Data: Unsupervised Method for Detecting Drift (MMD) | Khaki et al. (Amazon) | 2023 | `2309.03831_uncovering_drift_textual_data_unsupervised.pdf` | **MMD Baseline** |
| 8 | Detecting Covariate Drift in Text Data Using Document Embeddings and DR | Sodar et al. | 2023 | `2309.10000_detecting_covariate_drift_text_embeddings.pdf` | **Drift Baseline** |
| 9 | Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time (DriftLens) | Greco et al. | 2024 | `2406.17813_driftlens_unsupervised_concept_drift.pdf` | **Drift Baseline** |
| 10 | Maximum Mean Discrepancy on Exponential Windows for Online Change Detection (MMDEW) | Kalinke et al. (KIT) | 2022 | `2205.12706_mmd_exponential_windows_online_change.pdf` | **Streaming MMD** |
| 11 | Online Drift Detection with Maximum Concept Discrepancy (MCD-DD) | Wan et al. | 2024 | `2407.05375_online_drift_detection_maximum_concept_discrepancy.pdf` | Drift Baseline |
| 12 | Partial Wasserstein and MMD Distances: Bridging Outlier and Drift Detection | (Comparison) | 2021 | `2106.12893_partial_wasserstein_mmd_drift_detection.pdf` | Methods Survey |
| 13 | Testing Homological Equivalence Using Betti Numbers | (Statistical Testing) | 2022 | `2211.13959_testing_homological_equivalence_betti_numbers.pdf` | **TDA Statistics** |
| 14 | Ripser.py: A Lean Persistent Homology Library for Python | Tralie, Saul et al. | 2019 | `1908.02751_ripser_py_persistent_homology.pdf` | TDA Tool |
| 15 | giotto-tda: A Topological Data Analysis Toolkit for Machine Learning | Tauzin et al. | 2020 | `2004.02551_giotto_tda_toolkit.pdf` | TDA Tool |
| 16 | giotto-ph: A Python Library for High-Performance Computation of Persistent Homology | (giotto-ph) | 2021 | `2107.05412_giotto_ph_high_performance.pdf` | TDA Tool |
| 17 | Concept Drift Adaptation in Text Stream Mining: Comprehensive Review | (Survey) | 2023 | `2312.02901_concept_drift_text_stream_review.pdf` | Survey |
| 18 | Beyond Statistical Similarity: Semantic Drift | (Semantic Drift) | 2023 | `2309.16427_beyond_statistical_similarity_semantic_drift.pdf` | Drift Analysis |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 3**

| Name | Source | Size | Classes/Domains | Location | Primary Use |
|---|---|---|---|---|---|
| **AG News** | HuggingFace `ag_news` | 120K train / 7.6K test | 4 (World, Sports, Business, Sci/Tech) | `datasets/ag_news/data/` | Primary drift benchmark |
| **20 Newsgroups** | HuggingFace `SetFit/20_newsgroups` | 11.3K train / 7.5K test | 20 news categories | `datasets/20newsgroups/data/` | Fine-grained drift scenarios |
| **DBpedia14** | HuggingFace `dbpedia_14` | 560K train / 70K test | 14 entity categories | `datasets/dbpedia14/data/` | Large-scale multi-domain |

Data files are excluded from git (see `datasets/.gitignore`). See `datasets/README.md` for download instructions.

---

## Code Repositories

**Total repositories cloned: 8**

| Name | URL | Purpose | Location |
|---|---|---|---|
| **frouros** | github.com/IFCA-Advanced-Computing/frouros | Specialized drift detection library (MMD, KS, etc.) — baseline implementations | `code/frouros/` |
| **alibi-detect** | github.com/SeldonIO/alibi-detect | Comprehensive outlier, adversarial, and drift detection (MMD, LSDD, context-aware) | `code/alibi-detect/` |
| **ripser-py** | github.com/scikit-tda/ripser.py | Efficient persistent homology computation (Vietoris-Rips) | `code/ripser-py/` |
| **giotto-tda** | github.com/giotto-ai/giotto-tda | Full TDA/ML pipeline: filtrations, persistence diagrams, feature extraction, sklearn-compatible | `code/giotto-tda/` |
| **drift-lens** | github.com/grecosalvatore/drift-lens | DriftLens: per-label Fréchet Distance drift detection on neural embeddings | `code/drift-lens/` |
| **mmdew-change-detector** | github.com/FlopsKa/mmdew-change-detector | MMDEW: online streaming MMD change detector with polylog runtime | `code/mmdew-change-detector/` |
| **ZigZagLLMs** | github.com/RitAreaSciencePark/ZigZagLLMs | Zigzag persistence for tracking topological features across LLM layers | `code/ZigZagLLMs/` |
| **AwesomeTDA4NLP** | github.com/AdaUchendu/AwesomeTDA4NLP | Curated collection of TDA methods applied to NLP — survey companion | `code/AwesomeTDA4NLP/` |

See `code/README.md` for detailed descriptions and usage notes.

---

## Resource Gathering Notes

### Search Strategy
- Started with the three arXiv IDs specified in the research topic (which turned out to be unrelated papers — placeholder references)
- Conducted targeted web searches combining: "persistent homology drift detection LLM embeddings", "topological data analysis concept drift text", "maximum mean discrepancy streaming drift detection", "PHD persistent homology dimension text", "Betti numbers two-sample test"
- Used arXiv API to search for papers on relevant subtopics
- Followed citation trails from found papers (Basterrech 2024 → TDA literature; Short-PHD → Tulchinskii et al.)

### Selection Criteria
- **TDA papers**: Must apply persistent homology, Betti numbers, or persistence entropy to data streams or text/embedding analysis
- **Drift detection papers**: Must address embedding-space or text drift without requiring labels (unsupervised)
- **Baseline papers**: Key competing methods that the proposed TDA approach must outperform (MMD, Fréchet distance, covariance shift)
- **Tool papers**: Libraries needed for implementation (ripser, giotto-tda, frouros, alibi-detect)

### Challenges Encountered
- The three specified arXiv URLs (2205.13147, 2105.03468, 2007.01882) are unrelated to the topic — treated as placeholder references
- TDA + LLM drift detection is very new (2023–2026); fewer papers than mature methods like MMD
- Many relevant papers apply TDA to LLM internals (layer analysis) rather than to input stream monitoring — adapted their methods conceptually

### Gaps Identified
- No paper directly combines TDA with continuous monitoring of LLM input embedding streams (this is the research gap our study fills)
- Most TDA drift detection works use synthetic/image data; text embedding streams are underexplored
- Streaming TDA (online persistent homology updates) is largely unexplored

---

## Recommendations for Experiment Design

### 1. Primary Dataset
**AG News** — 4 topics, balanced (30K per class), standard benchmark. Create drift scenarios by:
- Abrupt switch: Window1 = 100% World news → Window2 = 100% Sports
- Gradual drift: Linearly increase Sports from 0% to 100% over 1000 samples
- Geometric-without-centroid-shift: Mix classes with same centroid but different geometry

### 2. Baseline Methods
| Method | Library | Configuration |
|---|---|---|
| Centroid shift | numpy | L2 norm of mean difference |
| Covariance shift | numpy/scipy | Frobenius norm of covariance difference |
| MMD (RBF) | frouros or alibi-detect | `MMD()` from frouros; RBF kernel |
| MMDEW | mmdew-change-detector | Streaming online version |
| DriftLens (Fréchet) | drift-lens | `FrechetInceptionDistance` on embeddings |

### 3. Proposed TDA Methods
| Method | Library | Configuration |
|---|---|---|
| Persistent entropy (H0) | giotto-tda `PersistenceEntropy` | Window of 100–500 embeddings |
| PHD (MST-based PH0) | ripser + scipy `minimum_spanning_tree` | Estimate via log-log regression |
| Betti curve (H0+H1) | giotto-tda `BettiCurve` | H0 and H1, 100 filtration steps |
| Persistence landscape | giotto-tda `PersistenceLandscape` | H0, 5 landscapes |

### 4. Evaluation Protocol
- Sliding window: 250 samples per window
- Statistical test: Mann-Whitney U (non-parametric) between reference and test window TDA features
- Metrics: AUC (drift detection), detection delay (samples until correct alarm), FPR at TPR=0.8
- 10 independent runs per scenario for variance estimation

### 5. Expected Key Finding
TDA features (especially persistent entropy + PHD) should outperform centroid shift and covariance shift on drift scenarios where only the *geometric structure* changes without significant mean/variance shift. MMD may be competitive but should be less sensitive to purely topological changes.

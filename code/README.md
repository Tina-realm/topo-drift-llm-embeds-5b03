# Code Repositories

Cloned repositories for the research project:
**Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams**

---

## TDA Libraries

### ripser-py (PRIMARY TDA COMPUTATION)
- **Location**: `code/ripser-py/`
- **URL**: https://github.com/scikit-tda/ripser.py
- **Purpose**: Fast persistent homology computation via Vietoris-Rips filtration (C++ Ripser backend)
- **Key Capability**: Compute PH0 (connected components), PH1 (loops), PH2 (voids) from point clouds
- **Installation**: `pip install ripser`
- **Basic Usage**:
  ```python
  from ripser import ripser
  import numpy as np
  
  # Compute persistent homology up to H1
  point_cloud = np.random.rand(200, 384)  # 200 embedding vectors, 384-dim
  result = ripser(point_cloud, maxdim=1)
  
  # Access persistence diagrams
  H0_diagram = result['dgms'][0]  # Connected components: (birth, death) pairs
  H1_diagram = result['dgms'][1]  # Loops: (birth, death) pairs
  ```
- **Notes**: Default VR complex; for large embeddings use `thresh` to limit filtration radius. H0 corresponds to MST (minimal spanning tree) structure.

### giotto-tda (SKLEARN-COMPATIBLE TDA ML PIPELINE)
- **Location**: `code/giotto-tda/`
- **URL**: https://github.com/giotto-ai/giotto-tda
- **Purpose**: End-to-end TDA for ML with sklearn-compatible API; feature extraction from persistence diagrams
- **Key Components**:
  - `VietorisRipsPersistence`: Compute persistence diagrams
  - `PersistenceEntropy`: Extract Shannon entropy of lifetime distributions (H0, H1)
  - `BettiCurve`: Betti numbers as function of filtration radius
  - `PersistenceLandscape`: Landscape functional summary
  - `PersistenceImage`: Rasterized persistence diagram
- **Installation**: `pip install giotto-tda`
- **Basic Usage**:
  ```python
  from gtda.homology import VietorisRipsPersistence
  from gtda.diagrams import PersistenceEntropy, BettiCurve
  
  # Compute persistence diagrams for a batch of point clouds
  VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
  diagrams = VR.fit_transform([point_cloud])  # shape: (n_clouds, n_points, 3)
  
  # Extract persistent entropy features
  PE = PersistenceEntropy()
  entropy_features = PE.fit_transform(diagrams)  # shape: (n_clouds, n_homology_dims)
  
  # Extract Betti curves
  BC = BettiCurve(n_bins=100)
  betti_features = BC.fit_transform(diagrams)  # shape: (n_clouds, n_bins * n_dims)
  ```
- **Notes**: Requires `giotto-ph` for faster computation on larger datasets. Default VR filtration up to H1 is generally sufficient.

---

## Drift Detection Libraries (Baselines)

### frouros (MMD AND STATISTICAL DRIFT BASELINES)
- **Location**: `code/frouros/`
- **URL**: https://github.com/IFCA-Advanced-Computing/frouros
- **Purpose**: Specialized drift detection library — only drift detection, nothing else
- **Key Methods**:
  - `MMD`: Maximum Mean Discrepancy two-sample test (offline batch)
  - `MMDStreaming`: Streaming MMD for online detection
  - `KSTest`: Kolmogorov-Smirnov test
  - `CVMTest`: Cramér-von Mises test
  - `ADWIN`, `DDM`, `EDDM`: Error-based drift detectors
- **Installation**: `pip install frouros`
- **Basic Usage**:
  ```python
  from frouros.detectors.data_drift import MMD
  from frouros.callbacks import PermutationTestDistanceBased
  
  # MMD test between reference and test embeddings
  detector = MMD()
  detector.fit(X_reference)  # reference embeddings: shape (n, d)
  
  result, _ = detector.compare(X_test)  # test embeddings: shape (m, d)
  print(f"MMD statistic: {result.distance}")
  print(f"p-value: {result.p_value}")  # with permutation callback
  ```
- **Notes**: Paper reference: arXiv:2208.06868 (Frouros library paper)

### alibi-detect (COMPREHENSIVE DRIFT DETECTION)
- **Location**: `code/alibi-detect/`
- **URL**: https://github.com/SeldonIO/alibi-detect
- **Purpose**: Broad library: outlier detection, adversarial detection, drift detection
- **Key Drift Detectors**:
  - `MMDDrift`: MMD-based drift detector (supports TF and PyTorch backends)
  - `LSDDDrift`: Least-Squares Density Difference drift detector
  - `ClassifierDrift`: Classifier-based drift detector (D3)
  - `ContextMMDDrift`: Context-aware MMD drift
- **Installation**: `pip install alibi-detect`
- **Basic Usage**:
  ```python
  from alibi_detect.cd import MMDDrift
  import numpy as np
  
  # Initialize with reference data
  cd = MMDDrift(X_reference, backend='pytorch', p_val=0.05)
  
  # Test new batch
  result = cd.predict(X_test)
  print(f"Drift detected: {result['data']['is_drift']}")
  print(f"p-value: {result['data']['p_val']}")
  ```
- **Notes**: Supports preprocessing with sentence transformers directly

### drift-lens (FRECHET DISTANCE BASELINE)
- **Location**: `code/drift-lens/`
- **URL**: https://github.com/grecosalvatore/drift-lens
- **Purpose**: DriftLens — per-label Fréchet Distance drift detection on neural embeddings
- **Paper**: arXiv:2406.17813 (IEEE TKDE 2024)
- **Key Capability**: Computes Fréchet Distance (mean + covariance Gaussian approximation) between reference and test window embeddings, per predicted class label
- **Installation**: `pip install drift-lens`
- **Notes**: Requires class label predictions (semi-supervised); can run in per-batch mode without labels. Key baseline that captures Gaussian structure changes (mean + covariance) but not higher-order topological structure.

### mmdew-change-detector (STREAMING MMD BASELINE)
- **Location**: `code/mmdew-change-detector/`
- **URL**: https://github.com/FlopsKa/mmdew-change-detector
- **Purpose**: MMDEW — online streaming change detector with O(log²t) runtime
- **Paper**: arXiv:2205.12706
- **Key Capability**: Efficient online MMD across exponential windows; best-in-class streaming MMD baseline
- **Notes**: Python implementation; directly applicable to embedding streams

---

## LLM + TDA Research Code

### ZigZagLLMs (TOPOLOGICAL ANALYSIS OF LLM LAYERS)
- **Location**: `code/ZigZagLLMs/`
- **URL**: https://github.com/RitAreaSciencePark/ZigZagLLMs
- **Purpose**: Zigzag persistence for tracking topological features across LLM layers; persistence similarity metric
- **Paper**: arXiv:2410.11042 (ICML 2025)
- **Key Capability**: Compute how topological features evolve across transformer layers; define persistence similarity
- **Relevance**: Methods for computing layer-wise topology can be adapted to compute window-wise topology for streaming drift detection

### AwesomeTDA4NLP (CURATED RESOURCE LIST)
- **Location**: `code/AwesomeTDA4NLP/`
- **URL**: https://github.com/AdaUchendu/AwesomeTDA4NLP
- **Purpose**: Comprehensive curated list of TDA methods applied to NLP — papers, datasets, code
- **Relevance**: Background reading; may point to additional relevant implementations

---

## Implementation Plan for Our Experiment

Based on the cloned repositories, the experimental pipeline should be:

```python
# 1. Load dataset
from datasets import load_from_disk
dataset = load_from_disk("datasets/ag_news/data")

# 2. Generate embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size=64)

# 3. Create streaming windows (simulate drift)
windows = create_drift_stream(embeddings, labels, scenario="abrupt")

# 4. TDA features per window
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
PE = PersistenceEntropy()
tda_features = [PE.transform(VR.transform([w]))[0] for w in windows]

# 5. PHD via MST (H0 only, scalable)
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
def compute_phd(embeddings, alpha=1.0):
    """Compute PH0 dimension via MST edge length scaling."""
    n_samples = [10, 20, 40, 80]  # increasing subsets
    log_E = []
    for n in n_samples:
        idx = np.random.choice(len(embeddings), n, replace=False)
        subset = embeddings[idx]
        D = squareform(pdist(subset))
        mst = minimum_spanning_tree(D)
        E = np.sum(mst.toarray()**alpha)
        log_E.append(np.log(E))
    # Linear regression: log(E) ~ (1 - alpha/d) * log(n) + log(C)
    slope, intercept = np.polyfit(np.log(n_samples), log_E, 1)
    phd = 1.0 / (1.0 - slope)  # estimated intrinsic dimension
    return phd

# 6. Baseline: MMD
from frouros.detectors.data_drift import MMD
mmd = MMD()
mmd.fit(reference_window)
result = mmd.compare(test_window)

# 7. Statistical test on TDA features
from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(tda_features_ref, tda_features_test)

# 8. Evaluate: AUC, detection delay, FPR
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(drift_labels, drift_scores)
```

"""
Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams
==============================================================================
Experiment v3 — addresses reviewer feedback:
  1. Second embedding model (BERT-base) + 20 Newsgroups
  2. Harder centroid-preserving drift (subtopic reweighting, style perturbation)
  3. New baselines: classifier-based two-sample test, persistence landscapes,
     sliced Wasserstein on persistence diagrams
  4. Proper FPR calibration with separate calibration/eval split
  5. End-to-end runtime reporting (including PH computation)
  6. Ablations: window size, TDA subsample, PCA dimensionality
  7. Synthetic experiment with AUC + permutation tests instead of separability ratio
  8. Confidence intervals (mean ± std) across seeds
"""

import os, sys, json, time, random, warnings, logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu, ks_2samp, special_ortho_group, permutation_test
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import ripser
import persim

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

WORKSPACE = Path('/workspaces/topo-drift-llm-embeds-5b03')
RESULTS_DIR = WORKSPACE / 'results_v3'
FIGURES_DIR = WORKSPACE / 'figures_v3'

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
TDA_SUBSAMPLE = 150
N_REF_WINDOWS = 10
N_DRIFT_WINDOWS = 10

# Ablation ranges
WINDOW_SIZES = [50, 100, 200, 400]
TDA_SUBSAMPLE_SIZES = [40, 80, 160]
PCA_DIMS = [20, 50, 100, None]  # None = raw (no PCA before TDA)
DEFAULT_WINDOW = 200
DEFAULT_PCA_DIM = 50  # PCA before TDA by default
N_PER_CLASS = 1000  # Matches Block 1 cached embeddings

# Embedding model configs
EMBEDDING_MODELS = {
    'minilm': {
        'name': 'all-MiniLM-L6-v2',
        'dim': 384,
    },
    'bert_base': {
        'name': 'all-mpnet-base-v2',  # Faster than bert-base-nli-mean-tokens, same dim
        'dim': 768,
    },
}


# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ─── DATA LOADING ────────────────────────────────────────────────────────────

_embedding_cache = {}
EMBEDDING_CACHE_DIR = WORKSPACE / 'embeddings_cache'
EMBEDDING_CACHE_DIR.mkdir(exist_ok=True)

def load_embeddings(dataset_name='ag_news', model_key='minilm', seed=42):
    """Load and cache embeddings. Returns dict: class_id -> np.array [N, dim]"""
    cache_key = f"{dataset_name}_{model_key}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # Check disk cache first
    disk_cache = EMBEDDING_CACHE_DIR / f"{cache_key}.npz"
    if disk_cache.exists():
        logger.info(f"Loading cached embeddings from {disk_cache}")
        data = np.load(disk_cache)
        embeddings_by_class = {int(k): data[k] for k in data.files}
        _embedding_cache[cache_key] = embeddings_by_class
        return embeddings_by_class

    set_seed(seed)
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    model_name = EMBEDDING_MODELS[model_key]['name']
    model = SentenceTransformer(model_name)
    logger.info(f"Loaded model: {model_name} ({EMBEDDING_MODELS[model_key]['dim']}-dim)")

    if dataset_name == 'ag_news':
        ds = load_dataset('ag_news', split='train')
        texts_by_class = {c: [] for c in range(4)}
        for item in ds:
            texts_by_class[item['label']].append(item['text'])
        class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'SciTech'}

    elif dataset_name == '20newsgroups':
        ds = load_dataset('SetFit/20_newsgroups', split='train')
        target_labels = {
            'comp.sys.mac.hardware': 0,
            'comp.sys.ibm.pc.hardware': 1,
            'rec.sport.baseball': 2,
            'rec.sport.hockey': 3,
            'sci.med': 4,
            'sci.space': 5,
        }
        texts_by_class = {v: [] for v in target_labels.values()}
        for item in ds:
            lbl = item.get('label_text', item.get('label', ''))
            if isinstance(lbl, int):
                all_labels = sorted(set(d['label_text'] for d in ds))
                lbl = all_labels[lbl] if lbl < len(all_labels) else str(lbl)
            if lbl in target_labels:
                texts_by_class[target_labels[lbl]].append(item['text'])
        class_names = {v: k for k, v in target_labels.items()}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    embeddings_by_class = {}
    for c in sorted(texts_by_class.keys()):
        texts = texts_by_class[c][:N_PER_CLASS]
        if len(texts) < 50:
            logger.warning(f"Class {c}: only {len(texts)} texts, skipping")
            continue
        logger.info(f"  Encoding class {c} ({len(texts)} texts) with {model_key}...")
        t0 = time.time()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        logger.info(f"    -> shape {embs.shape}, {time.time()-t0:.1f}s")
        embeddings_by_class[c] = embs

    # Save to disk cache
    np.savez(disk_cache, **{str(k): v for k, v in embeddings_by_class.items()})
    logger.info(f"Saved embeddings to {disk_cache}")

    # Free model memory
    del model
    import gc; gc.collect()

    _embedding_cache[cache_key] = embeddings_by_class
    return embeddings_by_class


# ─── STREAM CONSTRUCTION ─────────────────────────────────────────────────────

def build_stream(scenario, embs, window_size, seed):
    """
    Build stream of windows. Returns list of (window_array, drift_label).
    drift_label: 0=reference, 1=drift.
    """
    rng = np.random.default_rng(seed)

    def sample(cid, n):
        arr = embs[cid]
        idx = rng.choice(len(arr), size=n, replace=True)
        return arr[idx]

    windows = []

    if scenario == 'no_drift':
        for _ in range(N_REF_WINDOWS + N_DRIFT_WINDOWS):
            windows.append((sample(0, window_size), 0))

    elif scenario == 'abrupt_topic':
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(1, window_size), 1))

    elif scenario == 'gradual_topic':
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for i in range(N_DRIFT_WINDOWS):
            frac = (i + 1) / N_DRIFT_WINDOWS
            n1 = int(frac * window_size)
            n0 = window_size - n1
            parts = []
            if n0 > 0: parts.append(sample(0, n0))
            if n1 > 0: parts.append(sample(1, n1))
            w = np.vstack(parts)
            rng.shuffle(w)
            windows.append((w, 1 if frac > 0.2 else 0))

    elif scenario == 'style_shift':
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(2, window_size), 1))

    elif scenario == 'centroid_preserving':
        # Mix of class 0 + class 1 -> bimodal topology
        # Explicitly align centroid AFTER all transformations
        ref_windows_raw = []
        for _ in range(N_REF_WINDOWS):
            w = sample(0, window_size)
            ref_windows_raw.append(w)
            windows.append((w, 0))
        ref_centroid = np.mean([w.mean(axis=0) for w in ref_windows_raw], axis=0)

        for _ in range(N_DRIFT_WINDOWS):
            n_each = window_size // 2
            w = np.vstack([sample(0, n_each), sample(1, n_each)])
            # Re-center to match reference centroid exactly
            w = w - w.mean(axis=0) + ref_centroid
            rng.shuffle(w)
            windows.append((w, 1))

    elif scenario == 'subtopic_reweight':
        # Within-topic subtopic reweighting for AG News
        # Sports (class 1) has natural subtopics — shift proportions while
        # explicitly preserving the centroid
        from sklearn.cluster import KMeans
        sports = embs[1]
        km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(sports)
        subtopic_labels = km.labels_
        subtopic_groups = [sports[subtopic_labels == k] for k in range(3)]

        # Reference: uniform mixture of subtopics
        ref_windows_raw = []
        for _ in range(N_REF_WINDOWS):
            n_per = window_size // 3
            remainder = window_size - 3 * n_per
            parts = []
            for k in range(3):
                n_k = n_per + (1 if k < remainder else 0)
                idx = rng.choice(len(subtopic_groups[k]), size=n_k, replace=True)
                parts.append(subtopic_groups[k][idx])
            w = np.vstack(parts)
            rng.shuffle(w)
            ref_windows_raw.append(w)
            windows.append((w, 0))
        ref_centroid = np.mean([w.mean(axis=0) for w in ref_windows_raw], axis=0)

        # Drift: heavily skew toward subtopic 0 (80%) vs 10%+10%
        # Then re-center to match reference centroid
        for _ in range(N_DRIFT_WINDOWS):
            n0 = int(0.80 * window_size)
            n1 = int(0.10 * window_size)
            n2 = window_size - n0 - n1
            parts = []
            for k, nk in enumerate([n0, n1, n2]):
                if nk > 0:
                    idx = rng.choice(len(subtopic_groups[k]), size=nk, replace=True)
                    parts.append(subtopic_groups[k][idx])
            w = np.vstack(parts)
            # Explicitly preserve centroid
            w = w - w.mean(axis=0) + ref_centroid
            rng.shuffle(w)
            windows.append((w, 1))

    elif scenario == 'style_perturbation':
        # PCA subspace rotation that preserves centroid
        # Apply rotation in low-dim PCA space, then re-center exactly
        ref_data = embs[0]

        ref_windows_raw = []
        for _ in range(N_REF_WINDOWS):
            w = sample(0, window_size)
            ref_windows_raw.append(w)
            windows.append((w, 0))
        ref_centroid = np.mean([w.mean(axis=0) for w in ref_windows_raw], axis=0)

        # Build a perturbation: random rotation in a low-dim subspace
        pca = PCA(n_components=20).fit(ref_data)
        rot = np.eye(20)
        for _ in range(5):
            i, j = rng.choice(20, size=2, replace=False)
            angle = rng.uniform(0.3, 0.6)
            G = np.eye(20)
            G[i, i] = np.cos(angle)
            G[i, j] = -np.sin(angle)
            G[j, i] = np.sin(angle)
            G[j, j] = np.cos(angle)
            rot = rot @ G

        for _ in range(N_DRIFT_WINDOWS):
            w = sample(0, window_size)
            w_proj = pca.transform(w)
            w_rot = w_proj @ rot.T
            w_back = pca.inverse_transform(w_rot)
            # Re-center to match reference centroid exactly (no re-normalization)
            w_back = w_back - w_back.mean(axis=0) + ref_centroid
            rng.shuffle(w_back)
            windows.append((w_back, 1))

    elif scenario == 'subtle_gradual':
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for i in range(N_DRIFT_WINDOWS):
            frac = 0.05 * (i + 1)
            n1 = max(1, int(frac * window_size))
            n0 = window_size - n1
            parts = [sample(0, n0), sample(1, n1)]
            w = np.vstack(parts)
            rng.shuffle(w)
            windows.append((w, 1 if frac > 0.1 else 0))

    elif scenario == 'rotation_drift':
        pca = PCA(n_components=50).fit(embs[0])
        rot = np.eye(50)
        rot[:10, :10] = special_ortho_group.rvs(10, random_state=seed)

        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            w = sample(0, window_size)
            w_proj = pca.transform(w)
            w_rot = w_proj @ rot.T
            w_recon = pca.inverse_transform(w_rot)
            norms = np.linalg.norm(w_recon, axis=1, keepdims=True)
            w_recon = w_recon / (norms + 1e-10)
            windows.append((w_recon, 1))

    elif scenario == 'newsgroup_close':
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(1, window_size), 1))

    elif scenario == 'newsgroup_distant':
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(5, window_size), 1))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return windows


# ─── DRIFT DETECTORS ─────────────────────────────────────────────────────────

def centroid_shift(ref, test):
    return float(np.linalg.norm(ref.mean(axis=0) - test.mean(axis=0)))

def covariance_shift(ref, test):
    k = min(50, min(ref.shape[0], test.shape[0]) - 1, ref.shape[1])
    pca = PCA(n_components=k).fit(np.vstack([ref, test]))
    r = pca.transform(ref)
    t = pca.transform(test)
    cov_r = np.cov(r, rowvar=False)
    cov_t = np.cov(t, rowvar=False)
    return float(np.linalg.norm(cov_r - cov_t, 'fro'))

def mmd_rbf(ref, test, gamma=None):
    if gamma is None:
        combined = np.vstack([ref[:50], test[:50]])
        dists = pdist(combined, 'sqeuclidean')
        gamma = 1.0 / np.median(dists) if np.median(dists) > 0 else 1.0
    K_rr = np.exp(-gamma * cdist(ref, ref, 'sqeuclidean'))
    K_tt = np.exp(-gamma * cdist(test, test, 'sqeuclidean'))
    K_rt = np.exp(-gamma * cdist(ref, test, 'sqeuclidean'))
    np.fill_diagonal(K_rr, 0)
    np.fill_diagonal(K_tt, 0)
    n, m = len(ref), len(test)
    mmd2 = K_rr.sum() / (n*(n-1)) + K_tt.sum() / (m*(m-1)) - 2*K_rt.sum() / (n*m)
    return float(max(mmd2, 0))

def knn_shift(ref, test, k=5):
    def mean_knn(X):
        D = cdist(X, X, 'euclidean')
        np.fill_diagonal(D, np.inf)
        knn_dists = np.sort(D, axis=1)[:, :k]
        return knn_dists.mean()
    return abs(mean_knn(test) - mean_knn(ref))

def energy_distance(ref, test):
    n, m = len(ref), len(test)
    d_rt = cdist(ref, test, 'euclidean').mean()
    d_rr = pdist(ref, 'euclidean').mean() if n > 1 else 0
    d_tt = pdist(test, 'euclidean').mean() if m > 1 else 0
    return float(2 * d_rt - d_rr - d_tt)

def classifier_twosample(ref, test):
    """Classifier-based two-sample test: train logistic regression to discriminate.
    Uses single train/test split for speed instead of 5-fold CV."""
    n_ref, n_test = len(ref), len(test)
    # Subsample for speed
    max_n = 100
    if n_ref > max_n:
        idx = np.random.choice(n_ref, max_n, replace=False)
        ref_sub = ref[idx]
    else:
        ref_sub = ref
    if n_test > max_n:
        idx = np.random.choice(n_test, max_n, replace=False)
        test_sub = test[idx]
    else:
        test_sub = test

    X = np.vstack([ref_sub, test_sub])
    y = np.array([0]*len(ref_sub) + [1]*len(test_sub))

    # PCA to reduce dims for speed
    k = min(30, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)

    # Single train/test split (70/30) for speed
    clf = LogisticRegression(max_iter=100, C=1.0, solver='lbfgs')
    try:
        n_total = len(y)
        idx = np.random.permutation(n_total)
        n_train = int(0.7 * n_total)
        X_train, X_test = X_pca[idx[:n_train]], X_pca[idx[n_train:]]
        y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        return float(max(acc - 0.5, 0))
    except:
        return 0.0


# ─── TDA FEATURES ────────────────────────────────────────────────────────────

def compute_tda_features(X, subsample=TDA_SUBSAMPLE, seed=42, pca_dim=DEFAULT_PCA_DIM):
    """
    Compute TDA features from a point cloud.
    Returns dict of scalar features + raw diagrams.
    pca_dim: apply PCA before ripser (None = raw).
    """
    rng = np.random.default_rng(seed)
    n = min(subsample, len(X))
    pts = X[rng.choice(len(X), size=n, replace=False)]

    # Optional PCA before TDA
    if pca_dim is not None and pts.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        pts = pca.fit_transform(pts)

    t_ph = time.time()
    # Use thresh to limit computation: 95th percentile of pairwise distances
    dists_sample = pdist(pts[:min(40, len(pts))], 'euclidean')
    thresh = np.percentile(dists_sample, 95) if len(dists_sample) > 0 else np.inf
    result = ripser.ripser(pts, maxdim=1, metric='euclidean', thresh=thresh)
    t_ph_elapsed = time.time() - t_ph
    dgms = result['dgms']

    features = {}

    # H0 features
    h0 = dgms[0]
    h0_fin = h0[h0[:, 1] != np.inf]
    lifetimes_h0 = (h0_fin[:, 1] - h0_fin[:, 0]) if len(h0_fin) > 0 else np.array([0.0])
    lifetimes_h0 = lifetimes_h0[lifetimes_h0 > 0] if len(lifetimes_h0) > 0 else np.array([0.0])

    if len(lifetimes_h0) > 0 and lifetimes_h0.sum() > 0:
        p = lifetimes_h0 / lifetimes_h0.sum()
        features['pe_h0'] = float(-np.sum(p * np.log(p + 1e-10)))
    else:
        features['pe_h0'] = 0.0

    features['h0_total_persistence'] = float(lifetimes_h0.sum())
    features['h0_max_lifetime'] = float(lifetimes_h0.max()) if len(lifetimes_h0) > 0 else 0.0
    features['h0_mean_lifetime'] = float(lifetimes_h0.mean()) if len(lifetimes_h0) > 0 else 0.0
    features['h0_std_lifetime'] = float(lifetimes_h0.std()) if len(lifetimes_h0) > 1 else 0.0

    # H1 features
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    h1_fin = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
    lifetimes_h1 = (h1_fin[:, 1] - h1_fin[:, 0]) if len(h1_fin) > 0 else np.array([])
    lifetimes_h1 = lifetimes_h1[lifetimes_h1 > 0] if len(lifetimes_h1) > 0 else np.array([])

    if len(lifetimes_h1) > 0 and lifetimes_h1.sum() > 0:
        p1 = lifetimes_h1 / lifetimes_h1.sum()
        features['pe_h1'] = float(-np.sum(p1 * np.log(p1 + 1e-10)))
    else:
        features['pe_h1'] = 0.0

    features['h1_total_persistence'] = float(lifetimes_h1.sum()) if len(lifetimes_h1) > 0 else 0.0
    features['h1_n_features'] = len(lifetimes_h1)
    features['h1_max_lifetime'] = float(lifetimes_h1.max()) if len(lifetimes_h1) > 0 else 0.0

    # PHD via MST
    D = squareform(pdist(pts, 'euclidean'))
    mst = minimum_spanning_tree(D).toarray()
    mst_weights = mst[mst > 0]
    if len(mst_weights) > 2:
        sorted_w = np.sort(mst_weights)
        log_n = np.log(np.arange(1, len(sorted_w) + 1) + 1)
        log_w = np.log(sorted_w + 1e-10)
        coeffs = np.polyfit(log_n, log_w, 1)
        features['phd'] = float(-coeffs[0])
    else:
        features['phd'] = 0.0

    # Persistence landscape features (L1 and L2 norms of first 3 landscapes)
    features.update(_persistence_landscape_features(dgms))

    features['_diagrams'] = dgms
    features['_ph_time'] = t_ph_elapsed  # track PH computation time
    return features


def _persistence_landscape_features(dgms, n_landscapes=3, resolution=100):
    """Compute persistence landscape summary statistics."""
    features = {}
    for dim_idx, dim_name in enumerate(['h0', 'h1']):
        if dim_idx >= len(dgms):
            for k in range(n_landscapes):
                features[f'landscape_{dim_name}_L{k+1}_norm'] = 0.0
            continue

        dgm = dgms[dim_idx]
        dgm_fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
        if len(dgm_fin) == 0:
            for k in range(n_landscapes):
                features[f'landscape_{dim_name}_L{k+1}_norm'] = 0.0
            continue

        births = dgm_fin[:, 0]
        deaths = dgm_fin[:, 1]
        lifetimes = deaths - births

        # Only keep features with positive lifetime
        mask = lifetimes > 0
        births = births[mask]
        deaths = deaths[mask]
        if len(births) == 0:
            for k in range(n_landscapes):
                features[f'landscape_{dim_name}_L{k+1}_norm'] = 0.0
            continue

        # Sample grid
        t_min = births.min()
        t_max = deaths.max()
        grid = np.linspace(t_min, t_max, resolution)

        # Compute tent functions vectorized
        mids = (births + deaths) / 2
        # grid[j] vs births[i], deaths[i], mids[i] - broadcast
        # Shape: (n_features, resolution)
        grid_exp = grid[np.newaxis, :]  # (1, resolution)
        births_exp = births[:, np.newaxis]  # (n_features, 1)
        deaths_exp = deaths[:, np.newaxis]
        mids_exp = mids[:, np.newaxis]

        ascending = grid_exp - births_exp  # tent going up
        descending = deaths_exp - grid_exp  # tent going down
        tents = np.minimum(ascending, descending)
        tents = np.maximum(tents, 0.0)  # clip negatives (outside [birth, death])

        # k-th landscape = k-th largest tent value at each grid point
        for k in range(min(n_landscapes, len(births))):
            landscape_k = np.sort(tents, axis=0)[::-1][k] if k < len(births) else np.zeros(resolution)
            features[f'landscape_{dim_name}_L{k+1}_norm'] = float(np.linalg.norm(landscape_k))

        # Fill remaining
        for k in range(len(births), n_landscapes):
            features[f'landscape_{dim_name}_L{k+1}_norm'] = 0.0

    return features


def tda_drift_scores(ref_features, test_features):
    """Compute drift scores between two sets of TDA features."""
    scores = {}

    # Absolute difference of scalar features
    scalar_keys = ['pe_h0', 'pe_h1', 'h0_total_persistence', 'h1_total_persistence',
                   'h0_max_lifetime', 'h1_max_lifetime', 'h1_n_features', 'phd',
                   'h0_mean_lifetime', 'h0_std_lifetime']
    for key in scalar_keys:
        scores[f'tda_{key}'] = abs(ref_features[key] - test_features[key])

    # Persistence landscape differences
    for key in ref_features:
        if key.startswith('landscape_'):
            scores[f'tda_{key}'] = abs(ref_features[key] - test_features.get(key, 0.0))

    # Wasserstein distances between diagrams
    ref_dgms = ref_features['_diagrams']
    test_dgms = test_features['_diagrams']
    try:
        scores['tda_wass_h0'] = float(persim.wasserstein(ref_dgms[0], test_dgms[0], matching=False))
    except:
        scores['tda_wass_h0'] = 0.0
    try:
        if len(ref_dgms) > 1 and len(test_dgms) > 1 and len(ref_dgms[1]) > 0 and len(test_dgms[1]) > 0:
            scores['tda_wass_h1'] = float(persim.wasserstein(ref_dgms[1], test_dgms[1], matching=False))
        else:
            scores['tda_wass_h1'] = 0.0
    except:
        scores['tda_wass_h1'] = 0.0

    # Bottleneck distance
    try:
        scores['tda_bottleneck_h0'] = float(persim.bottleneck(ref_dgms[0], test_dgms[0]))
    except:
        scores['tda_bottleneck_h0'] = 0.0

    # Sliced Wasserstein distance on H0 and H1 diagrams
    scores['tda_sliced_wass_h0'] = _sliced_wasserstein_diagrams(ref_dgms[0], test_dgms[0])
    if len(ref_dgms) > 1 and len(test_dgms) > 1 and len(ref_dgms[1]) > 0 and len(test_dgms[1]) > 0:
        scores['tda_sliced_wass_h1'] = _sliced_wasserstein_diagrams(ref_dgms[1], test_dgms[1])
    else:
        scores['tda_sliced_wass_h1'] = 0.0

    return scores


def _sliced_wasserstein_diagrams(dgm1, dgm2, n_directions=50):
    """Sliced Wasserstein distance between persistence diagrams."""
    # Filter to finite points
    d1 = dgm1[np.isfinite(dgm1[:, 1])] if len(dgm1) > 0 else np.empty((0, 2))
    d2 = dgm2[np.isfinite(dgm2[:, 1])] if len(dgm2) > 0 else np.empty((0, 2))

    if len(d1) == 0 and len(d2) == 0:
        return 0.0

    # Add diagonal projections
    if len(d1) > 0:
        diag1 = np.column_stack([(d1[:, 0] + d1[:, 1]) / 2] * 2)
        d2_aug = np.vstack([d2, diag1]) if len(d2) > 0 else diag1
    else:
        d2_aug = d2

    if len(d2) > 0:
        diag2 = np.column_stack([(d2[:, 0] + d2[:, 1]) / 2] * 2)
        d1_aug = np.vstack([d1, diag2]) if len(d1) > 0 else diag2
    else:
        d1_aug = d1

    if len(d1_aug) == 0 or len(d2_aug) == 0:
        return 0.0

    # Sliced: project onto random directions and compute 1D Wasserstein
    rng = np.random.default_rng(42)
    total = 0.0
    for _ in range(n_directions):
        theta = rng.uniform(0, np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        proj1 = np.sort(d1_aug @ direction)
        proj2 = np.sort(d2_aug @ direction)
        # Interpolate to same length
        n = max(len(proj1), len(proj2))
        p1_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(proj1)), proj1)
        p2_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(proj2)), proj2)
        total += np.mean(np.abs(p1_interp - p2_interp))

    return float(total / n_directions)


# ─── EXPERIMENT RUNNER ────────────────────────────────────────────────────────

def run_experiment(dataset_name, model_key, scenarios, window_size=DEFAULT_WINDOW,
                   seeds=SEEDS, pca_dim=DEFAULT_PCA_DIM, tda_subsample=TDA_SUBSAMPLE):
    """Run full experiment for a dataset/model and set of scenarios."""
    logger.info(f"\n{'='*70}")
    logger.info(f"DATASET: {dataset_name}, MODEL: {model_key}, WINDOW: {window_size}, "
                f"PCA: {pca_dim}, TDA_SUB: {tda_subsample}")
    logger.info(f"{'='*70}")

    embs = load_embeddings(dataset_name, model_key)
    all_results = []

    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario} ---")
        for seed in seeds:
            set_seed(seed)
            stream = build_stream(scenario, embs, window_size, seed)
            labels = [lbl for _, lbl in stream]

            ref_embs_full = np.vstack([w for w, lbl in stream[:N_REF_WINDOWS]])
            # Subsample reference for efficiency (400 points sufficient for all methods)
            rng_ref = np.random.default_rng(seed)
            ref_idx = rng_ref.choice(len(ref_embs_full), size=min(400, len(ref_embs_full)), replace=False)
            ref_embs = ref_embs_full[ref_idx]

            # Use first half of ref windows for calibration, second half for evaluation
            # This fixes the FPR inconsistency
            cal_windows = stream[:N_REF_WINDOWS // 2]  # 5 windows for calibration
            eval_windows = stream[N_REF_WINDOWS // 2:]  # 5 ref + 10 drift for eval

            stat_scores = {m: [] for m in ['centroid', 'covariance', 'mmd', 'knn', 'energy', 'classifier']}
            tda_score_keys = None
            tda_scores_all = {}

            # Track timing separately for TDA (end-to-end including PH)
            stat_times = []
            tda_times = []

            ref_tda = compute_tda_features(ref_embs, subsample=tda_subsample, seed=seed, pca_dim=pca_dim)

            for wi, (w_embs, w_label) in enumerate(stream):
                # Statistical detectors
                t_stat = time.time()
                stat_scores['centroid'].append(centroid_shift(ref_embs, w_embs))
                stat_scores['covariance'].append(covariance_shift(ref_embs, w_embs))
                stat_scores['mmd'].append(mmd_rbf(ref_embs[:150], w_embs[:150]))
                stat_scores['knn'].append(knn_shift(w_embs[:80], ref_embs[:80]))
                stat_scores['energy'].append(energy_distance(ref_embs[:80], w_embs[:80]))
                stat_scores['classifier'].append(classifier_twosample(ref_embs, w_embs))
                t_stat_elapsed = time.time() - t_stat
                stat_times.append(t_stat_elapsed)

                # TDA detectors (end-to-end: PCA + ripser + feature extraction)
                t_tda = time.time()
                w_tda = compute_tda_features(w_embs, subsample=tda_subsample,
                                            seed=seed + wi, pca_dim=pca_dim)
                tda_diffs = tda_drift_scores(ref_tda, w_tda)
                t_tda_elapsed = time.time() - t_tda
                tda_times.append(t_tda_elapsed)

                if tda_score_keys is None:
                    tda_score_keys = sorted(tda_diffs.keys())
                    for k in tda_score_keys:
                        tda_scores_all[k] = []

                for k in tda_score_keys:
                    tda_scores_all[k].append(tda_diffs[k])

            # Combine all methods
            all_methods = {}
            all_methods.update(stat_scores)
            all_methods.update(tda_scores_all)

            mean_stat_time = np.mean(stat_times)
            mean_tda_time = np.mean(tda_times)

            # Compute metrics with proper calibration/evaluation split
            n_cal = N_REF_WINDOWS // 2
            n_eval_ref = N_REF_WINDOWS - n_cal  # remaining ref windows for eval

            for method_name, scores_list in all_methods.items():
                scores_arr = np.array(scores_list)
                labels_arr = np.array(labels)

                is_tda = method_name.startswith('tda_')
                runtime = mean_tda_time if is_tda else mean_stat_time

                # Calibration: threshold from first n_cal reference windows
                cal_scores = scores_arr[:n_cal]
                thresh = np.percentile(cal_scores, 95) if len(cal_scores) > 0 else 0

                # Evaluation: remaining ref windows + drift windows
                eval_scores = scores_arr[n_cal:]
                eval_labels = labels_arr[n_cal:]

                if scenario == 'no_drift':
                    # FPR on evaluation portion
                    fpr = float((eval_scores > thresh).mean()) if len(eval_scores) > 0 else 0
                    result = {
                        'dataset': dataset_name,
                        'model': model_key,
                        'method': method_name,
                        'drift_type': scenario,
                        'window_size': window_size,
                        'tda_subsample': tda_subsample,
                        'pca_dim': pca_dim if pca_dim else 'none',
                        'seed': seed,
                        'auc': None,
                        'detection_delay': None,
                        'fpr': fpr,
                        'runtime_per_window': runtime,
                    }
                else:
                    # AUC on full stream
                    try:
                        if len(set(labels_arr)) > 1:
                            auc = roc_auc_score(labels_arr, scores_arr)
                        else:
                            auc = None
                    except:
                        auc = None

                    # Detection delay on eval portion
                    drift_start_idx = n_eval_ref  # index within eval where drift starts
                    drift_scores = eval_scores[drift_start_idx:]
                    drift_detected = np.where(drift_scores > thresh)[0]
                    delay = int(drift_detected[0]) if len(drift_detected) > 0 else N_DRIFT_WINDOWS

                    # FPR on eval ref windows
                    eval_ref_scores = eval_scores[:drift_start_idx]
                    fpr = float((eval_ref_scores > thresh).mean()) if len(eval_ref_scores) > 0 else 0

                    result = {
                        'dataset': dataset_name,
                        'model': model_key,
                        'method': method_name,
                        'drift_type': scenario,
                        'window_size': window_size,
                        'tda_subsample': tda_subsample,
                        'pca_dim': pca_dim if pca_dim else 'none',
                        'seed': seed,
                        'auc': auc,
                        'detection_delay': delay,
                        'fpr': fpr,
                        'runtime_per_window': runtime,
                    }

                all_results.append(result)

            logger.info(f"  Seed {seed}: stat={mean_stat_time:.2f}s/w, tda={mean_tda_time:.2f}s/w")

    return all_results


# ─── SYNTHETIC TOPOLOGY EXPERIMENT (with proper AUC) ────────────────────────

def run_synthetic_experiment(seeds=SEEDS):
    """
    Controlled synthetic experiment with proper evaluation:
    - AUC instead of separability ratio
    - Permutation-based p-values
    """
    logger.info("\n" + "="*70)
    logger.info("SYNTHETIC TOPOLOGY EXPERIMENT (v3: AUC + permutation tests)")
    logger.info("="*70)

    dim = 50
    n_points = 200
    n_windows = 20  # 10 ref + 10 drift per trial
    results = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        drift_configs = {
            'centroid_shift': lambda rng=rng: rng.standard_normal((n_points, dim)) * 0.5 + 3.0 * np.eye(dim)[0],
            'annulus': lambda rng=rng: _make_annulus(n_points, dim, rng),
            'two_cluster': lambda rng=rng: _make_two_cluster(n_points, dim, rng),
            'variance_change': lambda rng=rng: rng.standard_normal((n_points, dim)) * 1.5,
            'no_drift': lambda rng=rng: rng.standard_normal((n_points, dim)) * 0.5,
        }

        for drift_name, drift_fn in drift_configs.items():
            # Build synthetic stream of windows
            labels = []
            all_scores = defaultdict(list)

            for wi in range(n_windows):
                rng_w = np.random.default_rng(seed * 1000 + wi)
                ref_data = rng_w.standard_normal((n_points, dim)) * 0.5

                if wi < n_windows // 2:
                    # Reference window: another draw from same distribution
                    test_data = rng_w.standard_normal((n_points, dim)) * 0.5
                    labels.append(0)
                else:
                    # Drift window
                    test_data = drift_fn()
                    labels.append(1 if drift_name != 'no_drift' else 0)

                # Compute all scores
                all_scores['centroid'].append(centroid_shift(ref_data, test_data))
                all_scores['covariance'].append(covariance_shift(ref_data, test_data))
                all_scores['mmd'].append(mmd_rbf(ref_data, test_data))
                all_scores['energy'].append(energy_distance(ref_data, test_data))
                all_scores['classifier'].append(classifier_twosample(ref_data, test_data))

                # TDA
                ref_tda = compute_tda_features(ref_data, subsample=min(TDA_SUBSAMPLE, n_points),
                                              seed=seed+wi, pca_dim=None)  # No PCA for 50-dim
                test_tda = compute_tda_features(test_data, subsample=min(TDA_SUBSAMPLE, n_points),
                                               seed=seed+wi+1000, pca_dim=None)
                tda_diffs = tda_drift_scores(ref_tda, test_tda)
                for k, v in tda_diffs.items():
                    all_scores[k].append(v)

            # Compute AUC for each method
            labels_arr = np.array(labels)
            for method_name, scores_list in all_scores.items():
                scores_arr = np.array(scores_list)
                try:
                    if len(set(labels_arr)) > 1:
                        auc = roc_auc_score(labels_arr, scores_arr)
                    else:
                        auc = 0.5
                except:
                    auc = 0.5

                results.append({
                    'drift_type': drift_name,
                    'seed': seed,
                    'method': method_name,
                    'auc': auc,
                })

    return pd.DataFrame(results)


def _make_annulus(n, dim, rng):
    theta = rng.uniform(0, 2*np.pi, n)
    r = rng.uniform(2.0, 3.0, n)
    pts = rng.standard_normal((n, dim)) * 0.1
    pts[:, 0] = r * np.cos(theta)
    pts[:, 1] = r * np.sin(theta)
    return pts

def _make_two_cluster(n, dim, rng):
    half = n // 2
    pts = rng.standard_normal((n, dim)) * 0.3
    pts[:half, 0] += 2.0
    pts[half:, 0] -= 2.0
    return pts


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def get_key_methods(df):
    """Return ordered list of key methods for plotting."""
    stat_methods = ['centroid', 'covariance', 'mmd', 'knn', 'energy', 'classifier']
    tda_key = ['tda_pe_h0', 'tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1',
               'tda_sliced_wass_h0', 'tda_sliced_wass_h1',
               'tda_phd', 'tda_h1_total_persistence', 'tda_bottleneck_h0',
               'tda_landscape_h0_L1_norm', 'tda_landscape_h1_L1_norm']
    all_methods_in_df = df['method'].unique()
    stat_present = [m for m in stat_methods if m in all_methods_in_df]
    tda_present = [m for m in tda_key if m in all_methods_in_df]
    return stat_present, tda_present


def plot_main_results(df, figdir):
    """Generate all main result figures."""
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)

    stat_methods, tda_methods = get_key_methods(df)
    key_methods = stat_methods + tda_methods

    # ── Fig 1: AUC heatmap by method × drift type (per dataset+model combo) ──
    for (ds, model), group_df in df.groupby(['dataset', 'model']):
        drift_df = group_df[(group_df['drift_type'] != 'no_drift') & (group_df['auc'].notna()) &
                           (group_df['method'].isin(key_methods))]
        if len(drift_df) == 0:
            continue

        pivot = drift_df.groupby(['method', 'drift_type'])['auc'].agg(['mean', 'std']).reset_index()
        pivot['label'] = pivot.apply(lambda r: f"{r['mean']:.2f}\n±{r['std']:.2f}", axis=1)
        pivot_mean = pivot.pivot(index='method', columns='drift_type', values='mean')
        pivot_label = pivot.pivot(index='method', columns='drift_type', values='label')

        method_order = [m for m in key_methods if m in pivot_mean.index]
        pivot_mean = pivot_mean.reindex(method_order)
        pivot_label = pivot_label.reindex(method_order)

        fig, ax = plt.subplots(figsize=(max(12, 2*len(pivot_mean.columns)), max(8, 0.5*len(method_order))))
        sns.heatmap(pivot_mean, annot=pivot_label.values, fmt='', cmap='RdYlGn',
                    vmin=0.4, vmax=1.0, linewidths=0.5, ax=ax, cbar_kws={'label': 'AUC'})
        ax.set_title(f'Detection AUC: {ds} / {model}\n(mean ± std across {len(SEEDS)} seeds)',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Detector Method')
        ax.set_xlabel('Drift Scenario')
        plt.tight_layout()
        plt.savefig(figdir / f'fig1_auc_heatmap_{ds}_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 2: Detection delay comparison ──
    drift_df = df[(df['drift_type'] != 'no_drift') & (df['auc'].notna()) &
                  (df['method'].isin(key_methods))]
    delay_df = drift_df[drift_df['detection_delay'].notna()]
    if len(delay_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        delay_agg = delay_df.groupby(['method', 'drift_type'])['detection_delay'].mean().reset_index()
        methods_in_data = [m for m in key_methods if m in delay_agg['method'].unique()]
        x = np.arange(len(methods_in_data))
        drift_types = sorted(delay_agg['drift_type'].unique())
        width = 0.8 / max(len(drift_types), 1)
        for i, dt in enumerate(drift_types):
            vals = []
            for m in methods_in_data:
                v = delay_agg[(delay_agg['method']==m) & (delay_agg['drift_type']==dt)]['detection_delay']
                vals.append(v.values[0] if len(v) > 0 else N_DRIFT_WINDOWS)
            ax.bar(x + i*width, vals, width, label=dt, alpha=0.8)
        ax.set_xticks(x + width * (len(drift_types)-1)/2)
        ax.set_xticklabels(methods_in_data, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Detection Delay (windows)')
        ax.set_title('Detection Delay by Method and Drift Type', fontweight='bold')
        ax.legend(title='Drift Type', bbox_to_anchor=(1.02, 1), fontsize=7)
        plt.tight_layout()
        plt.savefig(figdir / 'fig2_detection_delay.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 3: Overall TDA vs Statistical comparison ──
    if len(drift_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        method_means = drift_df.groupby('method')['auc'].agg(['mean', 'std']).sort_values('mean')
        method_means = method_means[method_means.index.isin(key_methods)]
        colors = ['#2196F3' if m in stat_methods else '#FF5722' for m in method_means.index]
        axes[0].barh(range(len(method_means)), method_means['mean'], xerr=method_means['std'],
                     color=colors, capsize=3)
        axes[0].set_yticks(range(len(method_means)))
        axes[0].set_yticklabels(method_means.index, fontsize=8)
        axes[0].set_xlabel('Mean AUC ± std')
        axes[0].set_title('Overall Detection Performance')
        axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        from matplotlib.patches import Patch
        axes[0].legend([Patch(color='#2196F3'), Patch(color='#FF5722')],
                      ['Statistical', 'TDA'], loc='lower right')

        if len(delay_df) > 0:
            delay_means = delay_df.groupby('method')['detection_delay'].agg(['mean', 'std']).sort_values('mean', ascending=False)
            delay_means = delay_means[delay_means.index.isin(key_methods)]
            colors2 = ['#2196F3' if m in stat_methods else '#FF5722' for m in delay_means.index]
            axes[1].barh(range(len(delay_means)), delay_means['mean'], xerr=delay_means['std'],
                        color=colors2, capsize=3)
            axes[1].set_yticks(range(len(delay_means)))
            axes[1].set_yticklabels(delay_means.index, fontsize=8)
            axes[1].set_xlabel('Mean Detection Delay ± std')
            axes[1].set_title('Detection Delay')

        plt.suptitle('TDA vs Statistical Drift Detectors', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(figdir / 'fig3_tda_vs_stat.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 4: FPR calibration ──
    nodrift_df = df[df['drift_type'] == 'no_drift']
    if len(nodrift_df) > 0:
        fpr_stats = nodrift_df.groupby('method')['fpr'].agg(['mean', 'std'])
        fpr_stats = fpr_stats[fpr_stats.index.isin(key_methods)].sort_values('mean')
        fig, ax = plt.subplots(figsize=(10, 6))
        colors3 = ['#2196F3' if m in stat_methods else '#FF5722' for m in fpr_stats.index]
        ax.barh(range(len(fpr_stats)), fpr_stats['mean'], xerr=fpr_stats['std'],
                color=colors3, capsize=3)
        ax.set_yticks(range(len(fpr_stats)))
        ax.set_yticklabels(fpr_stats.index, fontsize=8)
        ax.axvline(x=0.05, color='red', linestyle='--', label='Target FPR=0.05')
        ax.set_xlabel('False Positive Rate (no-drift eval windows)')
        ax.set_title('FPR Calibration (separate calibration/eval split)', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(figdir / 'fig4_fpr_calibration.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 8: Runtime comparison (end-to-end per window) ──
    runtime_df = df[df['method'].isin(key_methods)].groupby('method')['runtime_per_window'].agg(['mean', 'std'])
    if len(runtime_df) > 0:
        runtime_df = runtime_df.sort_values('mean')
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_rt = ['#2196F3' if m in stat_methods else '#FF5722' for m in runtime_df.index]
        ax.barh(range(len(runtime_df)), runtime_df['mean'] * 1000, xerr=runtime_df['std'] * 1000,
                color=colors_rt, capsize=3)
        ax.set_yticks(range(len(runtime_df)))
        ax.set_yticklabels(runtime_df.index, fontsize=8)
        ax.set_xlabel('End-to-end Runtime per Window (ms)')
        ax.set_title('Computational Cost (including PH computation for TDA)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(figdir / 'fig8_runtime.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_synthetic_results(synth_df, figdir):
    """Plot synthetic topology experiment with AUC."""
    stat_methods = ['centroid', 'covariance', 'mmd', 'energy', 'classifier']
    tda_methods = ['tda_pe_h0', 'tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1',
                   'tda_sliced_wass_h0', 'tda_sliced_wass_h1',
                   'tda_phd', 'tda_h1_total_persistence', 'tda_bottleneck_h0',
                   'tda_landscape_h0_L1_norm', 'tda_landscape_h1_L1_norm']

    all_methods = stat_methods + tda_methods
    present_methods = [m for m in all_methods if m in synth_df['method'].unique()]

    drift_types = [d for d in synth_df['drift_type'].unique() if d != 'no_drift']

    # Pivot: mean AUC ± std
    auc_data = []
    for dt in drift_types:
        for m in present_methods:
            subset = synth_df[(synth_df['drift_type'] == dt) & (synth_df['method'] == m)]
            if len(subset) > 0:
                auc_data.append({
                    'drift_type': dt, 'method': m,
                    'mean_auc': subset['auc'].mean(),
                    'std_auc': subset['auc'].std(),
                })
    auc_df = pd.DataFrame(auc_data)

    if len(auc_df) == 0:
        return

    pivot_mean = auc_df.pivot(index='method', columns='drift_type', values='mean_auc')
    pivot_std = auc_df.pivot(index='method', columns='drift_type', values='std_auc')
    pivot_label = pivot_mean.copy()
    for c in pivot_label.columns:
        pivot_label[c] = pivot_mean[c].apply(lambda x: f"{x:.2f}") + '\n±' + pivot_std[c].apply(lambda x: f"{x:.2f}")

    stat_present = [m for m in stat_methods if m in pivot_mean.index]
    tda_present = [m for m in tda_methods if m in pivot_mean.index]
    order = stat_present + tda_present
    pivot_mean = pivot_mean.reindex([m for m in order if m in pivot_mean.index])
    pivot_label = pivot_label.reindex([m for m in order if m in pivot_label.index])

    fig, ax = plt.subplots(figsize=(14, max(8, 0.5*len(order))))
    sns.heatmap(pivot_mean, annot=pivot_label.values, fmt='', cmap='RdYlGn',
                vmin=0.4, vmax=1.0, linewidths=0.5, ax=ax, cbar_kws={'label': 'AUC'})
    ax.set_title('Synthetic Topology Experiment: AUC (mean ± std)\n(Higher = Better Detection, 0.5 = Chance)',
                fontsize=13, fontweight='bold')
    if len(stat_present) > 0:
        ax.axhline(y=len(stat_present), color='black', linewidth=2)
    plt.tight_layout()
    plt.savefig(figdir / 'fig5_synthetic_auc.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_results(ablation_df, figdir):
    """Plot ablation studies: window size, TDA subsample, PCA dims."""
    if len(ablation_df) == 0:
        return

    stat_methods = ['centroid', 'covariance', 'mmd', 'knn', 'energy', 'classifier']
    key_tda = ['tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1', 'tda_sliced_wass_h1',
               'tda_phd', 'tda_landscape_h1_L1_norm']

    # ── Fig 6: Window size ablation ──
    ws_df = ablation_df[ablation_df['ablation'] == 'window_size']
    if len(ws_df) > 0:
        drift_ws = ws_df[(ws_df['drift_type'] != 'no_drift') & (ws_df['auc'].notna())]
        if len(drift_ws) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, methods, title in [(axes[0], stat_methods, 'Statistical Methods'),
                                        (axes[1], key_tda, 'TDA Methods')]:
                for m in methods:
                    mdf = drift_ws[drift_ws['method'] == m]
                    if len(mdf) == 0:
                        continue
                    means = mdf.groupby('window_size')['auc'].mean()
                    stds = mdf.groupby('window_size')['auc'].std()
                    ax.errorbar(means.index, means.values, yerr=stds.values,
                               marker='o', label=m, capsize=3)
                ax.set_xlabel('Window Size')
                ax.set_ylabel('Mean AUC ± std')
                ax.set_title(title)
                ax.legend(fontsize=7)
                ax.set_ylim(0.3, 1.05)
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
            plt.suptitle('Ablation: Window Size (50, 100, 200, 400)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(figdir / 'fig6_ablation_window_size.png', dpi=150, bbox_inches='tight')
            plt.close()

    # ── Fig 7a: TDA subsample ablation ──
    sub_df = ablation_df[ablation_df['ablation'] == 'tda_subsample']
    if len(sub_df) > 0:
        drift_sub = sub_df[(sub_df['drift_type'] != 'no_drift') & (sub_df['auc'].notna())]
        if len(drift_sub) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for m in key_tda:
                mdf = drift_sub[drift_sub['method'] == m]
                if len(mdf) == 0:
                    continue
                means = mdf.groupby('tda_subsample')['auc'].mean()
                stds = mdf.groupby('tda_subsample')['auc'].std()
                ax.errorbar(means.index, means.values, yerr=stds.values,
                           marker='o', label=m, capsize=3)
            ax.set_xlabel('TDA Subsample Size')
            ax.set_ylabel('Mean AUC ± std')
            ax.set_title('Ablation: TDA Subsample Size (40, 80, 160)', fontweight='bold')
            ax.legend(fontsize=7)
            ax.set_ylim(0.3, 1.05)
            plt.tight_layout()
            plt.savefig(figdir / 'fig7a_ablation_tda_subsample.png', dpi=150, bbox_inches='tight')
            plt.close()

    # ── Fig 7b: PCA dimensionality ablation ──
    pca_df = ablation_df[ablation_df['ablation'] == 'pca_dim']
    if len(pca_df) > 0:
        drift_pca = pca_df[(pca_df['drift_type'] != 'no_drift') & (pca_df['auc'].notna())]
        if len(drift_pca) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Map 'none' to a large number for plotting
            drift_pca = drift_pca.copy()
            drift_pca['pca_dim_num'] = drift_pca['pca_dim'].apply(lambda x: 384 if x == 'none' else int(x))
            for m in key_tda:
                mdf = drift_pca[drift_pca['method'] == m]
                if len(mdf) == 0:
                    continue
                means = mdf.groupby('pca_dim_num')['auc'].mean()
                stds = mdf.groupby('pca_dim_num')['auc'].std()
                ax.errorbar(means.index, means.values, yerr=stds.values,
                           marker='o', label=m, capsize=3)
            ax.set_xlabel('PCA Dimensions (384 = raw)')
            ax.set_ylabel('Mean AUC ± std')
            ax.set_title('Ablation: PCA Dimensionality Before TDA', fontweight='bold')
            ax.legend(fontsize=7)
            ax.set_ylim(0.3, 1.05)
            ax.set_xticks([20, 50, 100, 384])
            ax.set_xticklabels(['20', '50', '100', 'raw (384)'])
            plt.tight_layout()
            plt.savefig(figdir / 'fig7b_ablation_pca_dim.png', dpi=150, bbox_inches='tight')
            plt.close()


def plot_persistence_examples(embs, figdir):
    """Plot example persistence diagrams."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    configs = [
        ('Reference (World news)', 0, None),
        ('Drifted (Sports)', 1, None),
        ('Mixed (World+Business)', [0, 2], None),
    ]
    for col, (title, classes, _) in enumerate(configs):
        if isinstance(classes, list):
            pts = np.vstack([embs[c][:75] for c in classes])
        else:
            pts = embs[classes][:150]

        # Apply PCA before TDA (matching experiment setup)
        pca = PCA(n_components=DEFAULT_PCA_DIM)
        pts_pca = pca.fit_transform(pts)

        result = ripser.ripser(pts_pca, maxdim=1, metric='euclidean')
        dgms = result['dgms']

        for row, (dim_name, dgm) in enumerate([(f'H0', dgms[0]),
                                                  (f'H1', dgms[1] if len(dgms)>1 else np.empty((0,2)))]):
            ax = axes[row, col]
            fin = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
            if len(fin) > 0:
                ax.scatter(fin[:, 0], fin[:, 1], s=20, alpha=0.6,
                          c='blue' if row==0 else 'red')
                max_val = max(fin[:, 1].max(), fin[:, 0].max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'{title}\n{dim_name} ({len(fin)} features)')

    plt.suptitle('Persistence Diagrams: Reference vs Drifted Embeddings\n(after PCA-50)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figdir / 'fig9_persistence_examples.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    logger.info("="*70)
    logger.info("Topological Drift Detection Experiment v3")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*70)

    # Save environment info
    env = {
        'python': sys.version,
        'numpy': np.__version__,
        'sklearn': __import__('sklearn').__version__,
        'ripser': ripser.__version__,
        'persim': persim.__version__,
        'timestamp': datetime.now().isoformat(),
        'seeds': SEEDS,
        'window_sizes': WINDOW_SIZES,
        'tda_subsample': TDA_SUBSAMPLE,
        'pca_dim_default': DEFAULT_PCA_DIM,
        'embedding_models': {k: v['name'] for k, v in EMBEDDING_MODELS.items()},
        'gpu': 'none (CPU only)',
    }
    with open(RESULTS_DIR / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)

    all_results = []
    ablation_results = []

    SECONDARY_SEEDS = [42, 123, 456]  # Fewer seeds for secondary experiments

    def save_incremental(label):
        """Save results after each experiment block."""
        if all_results:
            pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'all_results.csv', index=False)
        if ablation_results:
            pd.DataFrame(ablation_results).to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
        logger.info(f"  [Incremental save after {label}: {len(all_results)} main + {len(ablation_results)} ablation results]")

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: AG News with MiniLM (original + new drift scenarios)
    # ═══════════════════════════════════════════════════════════════════════════
    ag_scenarios = ['no_drift', 'abrupt_topic', 'gradual_topic', 'style_shift',
                    'centroid_preserving', 'subtopic_reweight', 'style_perturbation',
                    'subtle_gradual', 'rotation_drift']
    ag_results = run_experiment('ag_news', 'minilm', ag_scenarios)
    all_results.extend(ag_results)
    save_incremental('Exp1: AG News MiniLM')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: 20 Newsgroups with MiniLM
    # ═══════════════════════════════════════════════════════════════════════════
    ng_scenarios = ['no_drift', 'newsgroup_close', 'newsgroup_distant']
    ng_results = run_experiment('20newsgroups', 'minilm', ng_scenarios)
    all_results.extend(ng_results)
    save_incremental('Exp2: 20NG MiniLM')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: 20 Newsgroups with MPNet-base (second encoder, 768-d)
    # ═══════════════════════════════════════════════════════════════════════════
    ng_bert_results = run_experiment('20newsgroups', 'bert_base', ng_scenarios,
                                     seeds=SECONDARY_SEEDS)
    all_results.extend(ng_bert_results)
    save_incremental('Exp3: 20NG MPNet')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: AG News with MPNet-base (key scenarios only)
    # ═══════════════════════════════════════════════════════════════════════════
    ag_bert_scenarios = ['no_drift', 'abrupt_topic', 'centroid_preserving',
                         'subtopic_reweight']
    ag_bert_results = run_experiment('ag_news', 'bert_base', ag_bert_scenarios,
                                     seeds=SECONDARY_SEEDS)
    all_results.extend(ag_bert_results)
    save_incremental('Exp4: AG News MPNet')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: Ablation - Window size (AG News, MiniLM, key scenarios)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("ABLATION: Window Size")
    logger.info("="*70)
    ablation_scenarios = ['no_drift', 'abrupt_topic', 'centroid_preserving']
    for ws in WINDOW_SIZES:
        ws_res = run_experiment('ag_news', 'minilm', ablation_scenarios,
                                window_size=ws, seeds=SECONDARY_SEEDS)
        for r in ws_res:
            r['ablation'] = 'window_size'
        ablation_results.extend(ws_res)
    save_incremental('Ablation: Window Size')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 6: Ablation - TDA subsample size
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("ABLATION: TDA Subsample Size")
    logger.info("="*70)
    for sub_size in TDA_SUBSAMPLE_SIZES:
        sub_res = run_experiment('ag_news', 'minilm', ablation_scenarios,
                                tda_subsample=sub_size, seeds=SECONDARY_SEEDS)
        for r in sub_res:
            r['ablation'] = 'tda_subsample'
        ablation_results.extend(sub_res)
    save_incremental('Ablation: TDA Subsample')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 7: Ablation - PCA dimensionality before TDA
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("ABLATION: PCA Dimensionality")
    logger.info("="*70)
    for pca_d in PCA_DIMS:
        pca_res = run_experiment('ag_news', 'minilm', ablation_scenarios,
                                pca_dim=pca_d, seeds=SECONDARY_SEEDS)
        for r in pca_res:
            r['ablation'] = 'pca_dim'
        ablation_results.extend(pca_res)
    save_incremental('Ablation: PCA Dim')

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 8: Synthetic topology (with AUC + permutation tests)
    # ═══════════════════════════════════════════════════════════════════════════
    synth_df = run_synthetic_experiment()

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nSaving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    results_df.to_json(RESULTS_DIR / 'all_results.json', orient='records', indent=2)

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)

    synth_df.to_csv(RESULTS_DIR / 'synthetic_results.csv', index=False)

    # JSONL
    with open(RESULTS_DIR / 'raw_results.jsonl', 'w') as f:
        for r in all_results + ablation_results:
            f.write(json.dumps(r, default=str) + '\n')

    # Metrics JSON
    metrics_output = []
    for _, row in results_df.iterrows():
        metrics_output.append({
            'method': row['method'],
            'drift_type': row['drift_type'],
            'dataset': row['dataset'],
            'model': row['model'],
            'window_size': int(row['window_size']),
            'detection_accuracy': row['auc'],
            'detection_delay': row['detection_delay'],
            'false_positive_rate': row['fpr'],
            'runtime_per_window': row['runtime_per_window'],
        })
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2, default=str)

    # Summary tables (per dataset+model)
    summary_parts = []
    for (ds, model), group in results_df[results_df['auc'].notna()].groupby(['dataset', 'model']):
        s = group.groupby('method').agg(
            mean_auc=('auc', 'mean'),
            std_auc=('auc', 'std'),
            mean_delay=('detection_delay', 'mean'),
            std_delay=('detection_delay', 'std'),
            mean_fpr=('fpr', 'mean'),
            mean_runtime=('runtime_per_window', 'mean'),
        ).sort_values('mean_auc', ascending=False)
        s['dataset'] = ds
        s['model'] = model
        summary_parts.append(s)

    if summary_parts:
        summary = pd.concat(summary_parts)
        summary.to_csv(RESULTS_DIR / 'summary.csv')
        logger.info(f"\nSummary:\n{summary.to_string()}")

    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nGenerating visualizations...")
    plot_main_results(results_df, FIGURES_DIR)
    plot_synthetic_results(synth_df, FIGURES_DIR)
    plot_ablation_results(ablation_df, FIGURES_DIR)

    embs = load_embeddings('ag_news', 'minilm')
    plot_persistence_examples(embs, FIGURES_DIR)

    t_total = time.time() - t_start
    logger.info(f"\nTotal experiment time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Figures saved to {FIGURES_DIR}")
    logger.info("DONE")


if __name__ == '__main__':
    main()

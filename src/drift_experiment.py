"""
Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams
============================================================================

Main experiment script comparing TDA-based drift detectors vs classical baselines
on AG News text embedding streams with controlled drift scenarios.

Author: Research Pipeline
Date: 2026-04-05
"""

import os
import sys
import json
import time
import random
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix as sp_dist_matrix
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import ripser
import persim

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE = Path('/workspaces/topo-drift-llm-embeds-5b03')
RESULTS_DIR = WORKSPACE / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
DATASETS_DIR = WORKSPACE / 'datasets'

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456]
WINDOW_SIZE = 200       # samples per window
SUBSAMPLE = 80          # max points for TDA (ripser is O(n^3))
N_REF_WINDOWS = 10      # no-drift windows for threshold calibration
N_DRIFT_WINDOWS = 10    # windows after drift injection
N_FILTRATION = 50       # filtration steps for Betti curves

# Drift scenarios to test
DRIFT_SCENARIOS = [
    'no_drift',       # Control: calibration
    'abrupt_topic',   # Sudden topic switch (World → Sports)
    'gradual_topic',  # Linear mix from World → Sports over 10 windows
    'geometric',      # Mix classes with same centroid but different topology
    'style_shift',    # Within-domain style variation (World → Business)
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ag_news_embeddings(seed: int = 42) -> dict:
    """
    Load AG News dataset and generate sentence embeddings.

    AG News classes:
        0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech

    Returns dict mapping class_id -> numpy array of embeddings [N, 384]
    """
    set_seed(seed)

    # Load pre-downloaded dataset
    logger.info("Loading AG News dataset...")
    from datasets import load_from_disk

    ag_path = DATASETS_DIR / 'ag_news' / 'data'
    ds = load_from_disk(str(ag_path))
    train = ds['train']

    logger.info(f"  Loaded {len(train)} training samples")
    logger.info(f"  Classes: {set(train['label'])}")

    # Group texts by class
    texts_by_class = {c: [] for c in range(4)}
    for item in train:
        texts_by_class[item['label']].append(item['text'])

    for c, texts in texts_by_class.items():
        logger.info(f"  Class {c}: {len(texts)} samples")

    # Generate embeddings
    logger.info("Loading sentence-transformer model...")
    from sentence_transformers import SentenceTransformer

    # Try to find cached model; fall back to download
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("  Model loaded")

    # Sample 2000 per class for efficiency
    N_PER_CLASS = 2000
    embeddings_by_class = {}

    for c in range(4):
        texts = texts_by_class[c][:N_PER_CLASS]
        logger.info(f"  Encoding class {c} ({len(texts)} texts)...")
        t0 = time.time()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        logger.info(f"    -> shape {embs.shape}, {time.time()-t0:.1f}s")
        embeddings_by_class[c] = embs

    logger.info("Embeddings generated.")
    return embeddings_by_class


# ─────────────────────────────────────────────────────────────────────────────
# 3. DRIFT SCENARIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_stream(scenario: str, embeddings_by_class: dict,
                 window_size: int, seed: int) -> list:
    """
    Build a stream of windows for a given drift scenario.

    Returns list of (window_embeddings, is_drift_label) tuples.
    is_drift_label = 0 for reference (no-drift), 1 for drift windows.

    Window structure:
        Windows 0-9: reference (no drift)
        Windows 10+: drift injected
    """
    rng = np.random.default_rng(seed)

    def sample(class_id, n):
        """Sample n embeddings from a class."""
        arr = embeddings_by_class[class_id]
        idx = rng.choice(len(arr), size=n, replace=True)
        return arr[idx]

    windows = []

    if scenario == 'no_drift':
        # All windows from class 0 (World)
        for i in range(N_REF_WINDOWS + N_DRIFT_WINDOWS):
            w = sample(0, window_size)
            windows.append((w, 0))  # never drift

    elif scenario == 'abrupt_topic':
        # Reference: class 0 (World). After window 10: class 1 (Sports)
        for i in range(N_REF_WINDOWS):
            w = sample(0, window_size)
            windows.append((w, 0))
        for i in range(N_DRIFT_WINDOWS):
            w = sample(1, window_size)
            windows.append((w, 1))

    elif scenario == 'gradual_topic':
        # Linearly increase fraction of class 1 from 0 to 1 over drift windows
        for i in range(N_REF_WINDOWS):
            w = sample(0, window_size)
            windows.append((w, 0))
        for i in range(N_DRIFT_WINDOWS):
            frac_drift = (i + 1) / N_DRIFT_WINDOWS
            n1 = int(frac_drift * window_size)
            n0 = window_size - n1
            parts = []
            if n0 > 0:
                parts.append(sample(0, n0))
            if n1 > 0:
                parts.append(sample(1, n1))
            w = np.vstack(parts)
            rng.shuffle(w)
            label = 1 if frac_drift > 0.2 else 0  # label drift when >20% contamination
            windows.append((w, label))

    elif scenario == 'geometric':
        # Both reference and drift have mix of 2 classes but with DIFFERENT class combos
        # Reference: mix of classes 0 (World) + 2 (Business) — similar semantic space
        # Drift: mix of classes 1 (Sports) + 3 (Sci/Tech) — different topics but same overall spread
        # Key: the *centroid* will shift somewhat, but the topological structure differs
        # To make it harder for low-order stats, we combine 2 classes in each window
        for i in range(N_REF_WINDOWS):
            n_each = window_size // 2
            w = np.vstack([sample(0, n_each), sample(2, n_each)])
            rng.shuffle(w)
            windows.append((w, 0))
        for i in range(N_DRIFT_WINDOWS):
            n_each = window_size // 2
            w = np.vstack([sample(1, n_each), sample(3, n_each)])
            rng.shuffle(w)
            windows.append((w, 1))

    elif scenario == 'style_shift':
        # World news → Business news (different domain but overlapping topics)
        for i in range(N_REF_WINDOWS):
            w = sample(0, window_size)
            windows.append((w, 0))
        for i in range(N_DRIFT_WINDOWS):
            w = sample(2, window_size)  # class 2 = Business
            windows.append((w, 1))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 4. TDA DRIFT DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def subsample_points(X: np.ndarray, n: int, rng=None) -> np.ndarray:
    """Subsample n points from X for tractable TDA computation."""
    if len(X) <= n:
        return X
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def compute_persistence_h0(X: np.ndarray, subsample: int = SUBSAMPLE) -> dict:
    """
    Compute H0 persistent homology summary statistics.

    H0 = connected components. Uses ripser for Vietoris-Rips filtration.

    Returns dict with:
        - entropy: persistent entropy (Shannon entropy of lifetimes)
        - n_components: number of H0 components at half-max radius
        - max_lifetime: maximum lifetime (proxy for diameter)
        - betti_auc: area under Betti-0 curve
        - diagrams: raw persistence diagrams
    """
    rng = np.random.default_rng(int(abs(np.sum(X)) % (2**31)))
    pts = subsample_points(X, subsample, rng)

    # Use euclidean distance matrix; ripser on point cloud
    result = ripser.ripser(pts, maxdim=1, metric='euclidean')
    diagrams = result['dgms']

    h0 = diagrams[0]  # shape [n, 2]; last point is infinite (born=0)
    h0_finite = h0[h0[:, 1] != np.inf]

    if len(h0_finite) == 0:
        return {
            'entropy': 0.0,
            'n_components': 1,
            'max_lifetime': 0.0,
            'betti_auc': 0.0,
            'diagrams': diagrams
        }

    lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return {
            'entropy': 0.0,
            'n_components': 1,
            'max_lifetime': 0.0,
            'betti_auc': 0.0,
            'diagrams': diagrams
        }

    # Persistent entropy (H0)
    total = lifetimes.sum()
    if total > 0:
        probs = lifetimes / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0.0

    # Betti-0 curve area (approximation: sum of lifetimes)
    betti_auc = float(lifetimes.sum())

    return {
        'entropy': float(entropy),
        'n_components': len(h0_finite),
        'max_lifetime': float(lifetimes.max()),
        'betti_auc': betti_auc,
        'diagrams': diagrams
    }


def compute_phd(X: np.ndarray, subsample: int = SUBSAMPLE, n_pts: list = None) -> float:
    """
    Persistent Homology Dimension (PHD) estimation.

    PHD estimates the fractal-like dimension of the point cloud via
    log-log regression of the expected number of connected components
    vs number of sampled points (from Tulchinskii et al., 2023).

    Higher PHD = more "spread out"/fractal embedding structure.
    Lower PHD = more compact/clustered.
    """
    if n_pts is None:
        n_pts = [20, 30, 40, 50, 60, 70]

    rng = np.random.default_rng(int(np.abs(X).sum() % (2**31)))

    log_ns = []
    log_e0s = []

    for n in n_pts:
        if n > len(X):
            continue
        # Multiple subsamples to get stable estimate
        e0_vals = []
        for trial in range(3):
            pts = subsample_points(X, n, rng)
            # H0 via MST: number of edges in MST = n-1; lifetimes = edge weights
            if len(pts) < 2:
                continue
            dm = cdist(pts, pts, metric='euclidean')
            # Minimal spanning tree
            mst = minimum_spanning_tree(dm).toarray()
            mst_weights = mst[mst > 0]
            # Expected edge length as proxy for E^0_alpha
            if len(mst_weights) > 0:
                e0_vals.append(np.mean(mst_weights))

        if len(e0_vals) > 0:
            log_ns.append(np.log(n))
            log_e0s.append(np.log(np.mean(e0_vals) + 1e-10))

    if len(log_ns) < 2:
        return 0.0

    # PHD = slope of log(E0) vs log(n) regression
    # Note: PHD is negative slope conventionally (more points = smaller avg distance)
    coeffs = np.polyfit(log_ns, log_e0s, 1)
    phd = float(coeffs[0])  # slope; negative for clustered, less negative for spread
    return phd


def compute_h1_features(X: np.ndarray, subsample: int = SUBSAMPLE) -> dict:
    """
    Compute H1 (loops/cycles) features from Vietoris-Rips.

    Returns:
        - h1_entropy: persistent entropy of H1 lifetimes
        - h1_count: number of H1 features
        - h1_max_lifetime: longest-living loop
    """
    rng = np.random.default_rng(int(np.abs(X).sum() % (2**31)))
    pts = subsample_points(X, subsample, rng)

    result = ripser.ripser(pts, maxdim=1, metric='euclidean')
    diagrams = result['dgms']

    if len(diagrams) < 2:
        return {'h1_entropy': 0.0, 'h1_count': 0, 'h1_max_lifetime': 0.0}

    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf]

    if len(h1_finite) == 0:
        return {'h1_entropy': 0.0, 'h1_count': 0, 'h1_max_lifetime': 0.0}

    lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return {'h1_entropy': 0.0, 'h1_count': 0, 'h1_max_lifetime': 0.0}

    total = lifetimes.sum()
    if total > 0:
        probs = lifetimes / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        entropy = 0.0

    return {
        'h1_entropy': entropy,
        'h1_count': len(lifetimes),
        'h1_max_lifetime': float(lifetimes.max())
    }


def _h1_from_diagrams(diagrams: list) -> dict:
    """Extract H1 features from pre-computed ripser diagrams (avoids redundant call)."""
    if len(diagrams) < 2:
        return {'h1_entropy': 0.0, 'h1_count': 0, 'h1_max_lifetime': 0.0}

    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf]

    if len(h1_finite) == 0:
        return {'h1_entropy': 0.0, 'h1_count': 0, 'h1_max_lifetime': 0.0}

    lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return {'h1_entropy': 0.0, 'h1_count': 0, 'h1_max_lifetime': 0.0}

    total = lifetimes.sum()
    entropy = float(-np.sum((lifetimes / total) * np.log(lifetimes / total + 1e-10))) if total > 0 else 0.0

    return {
        'h1_entropy': entropy,
        'h1_count': len(lifetimes),
        'h1_max_lifetime': float(lifetimes.max())
    }


def wasserstein_diagram_distance(dgm1: list, dgm2: list, dim: int = 0) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.
    Uses persim for efficient computation.
    """
    d1 = dgm1[dim]
    d2 = dgm2[dim]

    # Filter infinities
    d1 = d1[d1[:, 1] != np.inf] if len(d1) > 0 else d1
    d2 = d2[d2[:, 1] != np.inf] if len(d2) > 0 else d2

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        d1 = np.array([[0, 0]])
    if len(d2) == 0:
        d2 = np.array([[0, 0]])

    try:
        dist = persim.wasserstein(d1, d2)
        return float(dist)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLASSICAL DRIFT DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def centroid_shift(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """L2 norm of difference between window centroids."""
    return float(np.linalg.norm(X_ref.mean(axis=0) - X_test.mean(axis=0)))


def covariance_shift(X_ref: np.ndarray, X_test: np.ndarray) -> float:
    """Frobenius norm of covariance matrix difference."""
    # Use PCA-reduced covariance for efficiency (top-50 dims)
    from sklearn.decomposition import PCA
    n_components = min(50, X_ref.shape[1], X_ref.shape[0]-1, X_test.shape[0]-1)
    if n_components < 2:
        return 0.0
    try:
        pca = PCA(n_components=n_components)
        # Fit on ref, transform both
        ref_proj = pca.fit_transform(X_ref)
        test_proj = pca.transform(X_test)
        cov_ref = np.cov(ref_proj.T)
        cov_test = np.cov(test_proj.T)
        return float(np.linalg.norm(cov_ref - cov_test, 'fro'))
    except Exception:
        return 0.0


def mmd_rbf(X_ref: np.ndarray, X_test: np.ndarray, subsample: int = 200) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.

    Uses median heuristic for bandwidth.
    Subsamples for O(n^2) tractability.
    """
    rng = np.random.default_rng(42)

    n = min(subsample, len(X_ref), len(X_test))
    idx_ref = rng.choice(len(X_ref), size=n, replace=False)
    idx_test = rng.choice(len(X_test), size=n, replace=False)

    X = X_ref[idx_ref]
    Y = X_test[idx_test]

    # Median heuristic for bandwidth
    XY = np.vstack([X, Y])
    dists = cdist(XY, XY, 'sqeuclidean')
    sigma2 = float(np.median(dists[dists > 0]))
    if sigma2 == 0:
        sigma2 = 1.0

    K_XX = np.exp(-cdist(X, X, 'sqeuclidean') / sigma2)
    K_YY = np.exp(-cdist(Y, Y, 'sqeuclidean') / sigma2)
    K_XY = np.exp(-cdist(X, Y, 'sqeuclidean') / sigma2)

    mmd2 = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
    return float(max(0.0, mmd2))


def knn_distance_shift(X_ref: np.ndarray, X_test: np.ndarray, k: int = 5) -> float:
    """
    Mean k-NN distance shift.

    Computes mean kNN distance for each window and returns the difference.
    """
    def mean_knn_dist(X, k_):
        dm = cdist(X, X, 'euclidean')
        np.fill_diagonal(dm, np.inf)
        knn_dists = np.sort(dm, axis=1)[:, :k_]
        return float(knn_dists.mean())

    n = min(200, len(X_ref), len(X_test))
    rng = np.random.default_rng(42)
    Xr = X_ref[rng.choice(len(X_ref), size=n, replace=False)]
    Xt = X_test[rng.choice(len(X_test), size=n, replace=False)]

    return abs(mean_knn_dist(Xr, k) - mean_knn_dist(Xt, k))


# ─────────────────────────────────────────────────────────────────────────────
# 6. WINDOW-LEVEL SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_scores(window_ref: np.ndarray, window_test: np.ndarray) -> dict:
    """
    Compute all drift scores between two windows.

    Returns dict with scores from all detectors.
    """
    scores = {}
    timings = {}

    # --- Classical methods ---
    t0 = time.time()
    scores['centroid'] = centroid_shift(window_ref, window_test)
    timings['centroid'] = time.time() - t0

    t0 = time.time()
    scores['covariance'] = covariance_shift(window_ref, window_test)
    timings['covariance'] = time.time() - t0

    t0 = time.time()
    scores['mmd'] = mmd_rbf(window_ref, window_test)
    timings['mmd'] = time.time() - t0

    t0 = time.time()
    scores['knn'] = knn_distance_shift(window_ref, window_test)
    timings['knn'] = time.time() - t0

    # --- TDA methods ---
    t0 = time.time()
    tda_ref = compute_persistence_h0(window_ref)
    tda_test = compute_persistence_h0(window_test)
    timings['tda_h0'] = time.time() - t0

    scores['tda_entropy_h0'] = abs(tda_ref['entropy'] - tda_test['entropy'])
    scores['tda_betti_auc'] = abs(tda_ref['betti_auc'] - tda_test['betti_auc'])
    scores['tda_max_lifetime'] = abs(tda_ref['max_lifetime'] - tda_test['max_lifetime'])

    t0 = time.time()
    scores['tda_wasserstein_h0'] = wasserstein_diagram_distance(
        tda_ref['diagrams'], tda_test['diagrams'], dim=0)
    timings['tda_wasserstein'] = time.time() - t0

    t0 = time.time()
    scores['tda_phd'] = abs(compute_phd(window_ref) - compute_phd(window_test))
    timings['tda_phd'] = time.time() - t0

    # Extract H1 features from already-computed diagrams (no extra ripser call)
    t0 = time.time()
    h1_ref = _h1_from_diagrams(tda_ref['diagrams'])
    h1_test = _h1_from_diagrams(tda_test['diagrams'])
    timings['tda_h1'] = time.time() - t0
    scores['tda_entropy_h1'] = abs(h1_ref['h1_entropy'] - h1_test['h1_entropy'])
    scores['tda_h1_count'] = abs(h1_ref['h1_count'] - h1_test['h1_count'])

    return scores, timings


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario: str, embeddings_by_class: dict, seed: int) -> dict:
    """
    Run one drift scenario experiment.

    Returns dict with per-window scores, labels, and timings.
    """
    logger.info(f"  Running scenario={scenario}, seed={seed}")
    set_seed(seed)

    windows = build_stream(scenario, embeddings_by_class, WINDOW_SIZE, seed)
    logger.info(f"    {len(windows)} windows built")

    # Compute reference window (aggregate of first N_REF_WINDOWS windows)
    ref_windows = [w for w, label in windows[:N_REF_WINDOWS]]
    ref_pool = np.vstack(ref_windows)

    # Compute scores for each consecutive pair (reference vs current window)
    all_scores = []
    all_labels = []
    all_timings = []

    for i, (window, label) in enumerate(windows):
        # Use reference pool for comparison
        t_start = time.time()
        scores, timings = compute_all_scores(ref_pool[:WINDOW_SIZE], window)
        total_time = time.time() - t_start

        scores['window_idx'] = i
        scores['is_drift'] = label
        scores['scenario'] = scenario
        scores['seed'] = seed
        scores['total_time'] = total_time

        all_scores.append(scores)
        all_labels.append(label)
        all_timings.append(timings)

        if (i + 1) % 5 == 0:
            logger.info(f"    Window {i+1}/{len(windows)} done")

    return {
        'scores': all_scores,
        'labels': all_labels,
        'timings': all_timings,
        'scenario': scenario,
        'seed': seed
    }


def compute_detection_metrics(scores_list: list, labels: list, method: str) -> dict:
    """
    Compute detection performance metrics for one method.

    Args:
        scores_list: list of score dicts (one per window)
        labels: ground truth drift labels (0=no-drift, 1=drift)
        method: name of the method

    Returns dict with accuracy, detection_delay, FPR, etc.
    """
    scores = np.array([s[method] for s in scores_list])
    labels = np.array(labels)

    # AUC-ROC (higher score = more likely drift)
    if len(np.unique(labels)) < 2:
        auc = float('nan')
    else:
        auc = float(roc_auc_score(labels, scores))

    # Calibrate threshold on reference windows (95th percentile)
    ref_scores = scores[labels == 0]
    if len(ref_scores) == 0:
        threshold = 0.0
    else:
        threshold = float(np.percentile(ref_scores, 95))

    drift_preds = (scores > threshold).astype(int)

    # FPR
    ref_mask = labels == 0
    if ref_mask.sum() > 0:
        fpr = float(drift_preds[ref_mask].mean())
    else:
        fpr = float('nan')

    # Detection delay: first window correctly flagged after drift injection
    drift_mask = labels == 1
    drift_windows_pred = drift_preds[drift_mask]
    if drift_mask.sum() > 0 and drift_preds[drift_mask].sum() > 0:
        first_detected = int(np.argmax(drift_windows_pred))
        detection_delay = first_detected  # windows after drift starts
    else:
        detection_delay = N_DRIFT_WINDOWS  # never detected

    # TPR at calibrated threshold
    if drift_mask.sum() > 0:
        tpr = float(drift_preds[drift_mask].mean())
    else:
        tpr = float('nan')

    return {
        'method': method,
        'auc': auc,
        'tpr': tpr,
        'fpr': fpr,
        'detection_delay': detection_delay,
        'threshold': threshold,
        'detection_accuracy': (drift_preds == labels).mean() if len(labels) > 0 else float('nan')
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

METHODS_DISPLAY = {
    'centroid': 'Centroid Shift',
    'covariance': 'Covariance Shift',
    'mmd': 'MMD (RBF)',
    'knn': 'kNN Distance',
    'tda_entropy_h0': 'TDA: PE (H0)',
    'tda_wasserstein_h0': 'TDA: Wasserstein (H0)',
    'tda_phd': 'TDA: PHD',
    'tda_entropy_h1': 'TDA: PE (H1)',
    'tda_betti_auc': 'TDA: Betti AUC',
}

TDA_METHODS = ['tda_entropy_h0', 'tda_wasserstein_h0', 'tda_phd', 'tda_entropy_h1', 'tda_betti_auc']
STAT_METHODS = ['centroid', 'covariance', 'mmd', 'knn']


def plot_detection_by_drift_type(metrics_df: pd.DataFrame, save_path: Path):
    """Figure 1: Detection AUC by drift type for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    scenarios = [s for s in DRIFT_SCENARIOS if s != 'no_drift']

    # AUC heatmap
    pivot = metrics_df[metrics_df['scenario'] != 'no_drift'].pivot_table(
        values='auc', index='method', columns='scenario', aggfunc='mean')

    ax = axes[0]
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', vmin=0.4, vmax=1.0,
                annot=True, fmt='.2f', linewidths=0.5)
    ax.set_title('Detection AUC by Method & Drift Type\n(mean over 3 seeds)', fontsize=12)
    ax.set_xlabel('Drift Scenario')
    ax.set_ylabel('Method')
    ax.set_yticklabels([METHODS_DISPLAY.get(m, m) for m in pivot.index], rotation=0)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in pivot.columns], rotation=15)

    # Bar chart comparing best TDA vs best statistical
    ax2 = axes[1]
    tda_auc = metrics_df[
        (metrics_df['method'].isin(TDA_METHODS)) &
        (metrics_df['scenario'] != 'no_drift')
    ].groupby('scenario')['auc'].max().reset_index()
    tda_auc['type'] = 'Best TDA'

    stat_auc = metrics_df[
        (metrics_df['method'].isin(STAT_METHODS)) &
        (metrics_df['scenario'] != 'no_drift')
    ].groupby('scenario')['auc'].max().reset_index()
    stat_auc['type'] = 'Best Statistical'

    combined = pd.concat([tda_auc, stat_auc])
    sns.barplot(data=combined, x='scenario', y='auc', hue='type', ax=ax2,
                palette=['#2196F3', '#FF5722'])
    ax2.set_title('Best TDA vs Best Statistical Method\nAUC by Drift Type', fontsize=12)
    ax2.set_xlabel('Drift Scenario')
    ax2.set_ylabel('Detection AUC')
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in combined['scenario'].unique()], rotation=15)
    ax2.legend(title='Method Type')
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_delay_vs_fpr(metrics_df: pd.DataFrame, save_path: Path):
    """Figure 2: Detection delay vs FPR trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter drift scenarios only
    drift_data = metrics_df[metrics_df['scenario'] != 'no_drift']

    # Detection delay per method
    ax1 = axes[0]
    delay_data = drift_data.groupby('method')['detection_delay'].mean().reset_index()
    delay_data['type'] = delay_data['method'].apply(
        lambda m: 'TDA' if m in TDA_METHODS else 'Statistical')
    delay_data = delay_data.sort_values('detection_delay')

    colors = ['#2196F3' if t == 'TDA' else '#FF5722' for t in delay_data['type']]
    bars = ax1.barh(range(len(delay_data)), delay_data['detection_delay'], color=colors)
    ax1.set_yticks(range(len(delay_data)))
    ax1.set_yticklabels([METHODS_DISPLAY.get(m, m) for m in delay_data['method']])
    ax1.set_xlabel('Mean Detection Delay (windows)')
    ax1.set_title('Detection Delay by Method\n(lower is better)', fontsize=11)
    ax1.axvline(5, color='gray', linestyle='--', alpha=0.5, label='Mid-point')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='TDA'),
                       Patch(facecolor='#FF5722', label='Statistical')]
    ax1.legend(handles=legend_elements)

    # FPR per method
    ax2 = axes[1]
    fpr_data = metrics_df[metrics_df['scenario'] == 'no_drift'].groupby('method')['fpr'].mean().reset_index()
    fpr_data['type'] = fpr_data['method'].apply(
        lambda m: 'TDA' if m in TDA_METHODS else 'Statistical')
    fpr_data = fpr_data.sort_values('fpr')

    colors = ['#2196F3' if t == 'TDA' else '#FF5722' for t in fpr_data['type']]
    ax2.barh(range(len(fpr_data)), fpr_data['fpr'], color=colors)
    ax2.set_yticks(range(len(fpr_data)))
    ax2.set_yticklabels([METHODS_DISPLAY.get(m, m) for m in fpr_data['method']])
    ax2.set_xlabel('False Positive Rate (on no-drift windows)')
    ax2.set_title('False Positive Rate by Method\n(lower is better)', fontsize=11)
    ax2.axvline(0.05, color='gray', linestyle='--', alpha=0.5, label='Target FPR')
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_tda_vs_statistical(metrics_df: pd.DataFrame, save_path: Path):
    """Figure 3: TDA vs statistical method comparison scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter plot: per-scenario AUC for TDA vs statistical
    scenarios = [s for s in DRIFT_SCENARIOS if s != 'no_drift']

    tda_best = {}
    stat_best = {}

    for sc in scenarios:
        sc_data = metrics_df[metrics_df['scenario'] == sc]
        tda_vals = sc_data[sc_data['method'].isin(TDA_METHODS)]['auc'].dropna()
        stat_vals = sc_data[sc_data['method'].isin(STAT_METHODS)]['auc'].dropna()
        if len(tda_vals) > 0 and len(stat_vals) > 0:
            tda_best[sc] = tda_vals.max()
            stat_best[sc] = stat_vals.max()

    ax1 = axes[0]
    sc_names = list(tda_best.keys())
    tda_auc_vals = [tda_best[s] for s in sc_names]
    stat_auc_vals = [stat_best[s] for s in sc_names]

    ax1.scatter(stat_auc_vals, tda_auc_vals, s=150, zorder=5,
                c=['#e74c3c' if t > s else '#2ecc71' for t, s in zip(tda_auc_vals, stat_auc_vals)])
    ax1.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.4, label='Equal performance')

    for i, sc in enumerate(sc_names):
        ax1.annotate(sc.replace('_', '\n'),
                     (stat_auc_vals[i], tda_auc_vals[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_xlabel('Best Statistical Method AUC')
    ax1.set_ylabel('Best TDA Method AUC')
    ax1.set_title('TDA vs Statistical: Detection AUC\nRed = TDA wins, Green = Statistical wins', fontsize=10)
    ax1.set_xlim(0.4, 1.05)
    ax1.set_ylim(0.4, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Overall AUC comparison bar chart
    ax2 = axes[1]
    avg_auc = metrics_df[metrics_df['scenario'] != 'no_drift'].groupby('method')['auc'].mean().reset_index()
    avg_auc = avg_auc.sort_values('auc', ascending=True)
    avg_auc['type'] = avg_auc['method'].apply(
        lambda m: 'TDA' if m in TDA_METHODS else 'Statistical')

    colors = ['#2196F3' if t == 'TDA' else '#FF5722' for t in avg_auc['type']]
    bars = ax2.barh(range(len(avg_auc)), avg_auc['auc'], color=colors)
    ax2.set_yticks(range(len(avg_auc)))
    ax2.set_yticklabels([METHODS_DISPLAY.get(m, m) for m in avg_auc['method']])
    ax2.set_xlabel('Mean Detection AUC (across all drift scenarios)')
    ax2.set_title('Overall Detection Performance\n(mean AUC over scenarios & seeds)', fontsize=10)
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlim(0.4, 1.0)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='TDA'),
                       Patch(facecolor='#FF5722', label='Statistical')]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_sensitivity_vs_window_size(embeddings_by_class: dict, save_path: Path):
    """Figure 4: Detection AUC vs window size for key methods."""
    logger.info("  Computing window size sensitivity...")

    window_sizes = [50, 100, 150, 200, 300]
    methods_to_test = ['centroid', 'mmd', 'tda_entropy_h0', 'tda_wasserstein_h0']

    results = {m: [] for m in methods_to_test}

    rng = np.random.default_rng(42)

    for ws in window_sizes:
        logger.info(f"    Window size: {ws}")
        # Use abrupt topic drift scenario
        ref_pool = embeddings_by_class[0][:1000]
        drift_pool = embeddings_by_class[1][:1000]

        aucs = {m: [] for m in methods_to_test}

        for trial in range(5):
            # Reference windows
            ref_idx = rng.choice(len(ref_pool), size=ws, replace=True)
            ref_w = ref_pool[ref_idx]

            # Drift window
            drift_idx = rng.choice(len(drift_pool), size=ws, replace=True)
            drift_w = drift_pool[drift_idx]

            # No-drift window
            nodrift_idx = rng.choice(len(ref_pool), size=ws, replace=True)
            nodrift_w = ref_pool[nodrift_idx]

            for method in methods_to_test:
                score_drift, _ = compute_all_scores(ref_w, drift_w)
                score_nodrift, _ = compute_all_scores(ref_w, nodrift_w)

                s_d = score_drift[method]
                s_n = score_nodrift[method]

                # Simple separability: ratio of scores
                if s_n > 0:
                    aucs[method].append(s_d / (s_n + 1e-10))
                else:
                    aucs[method].append(1.0 if s_d > 0 else 0.0)

        for m in methods_to_test:
            results[m].append(np.mean(aucs[m]))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {'centroid': '#FF5722', 'mmd': '#FF9800',
              'tda_entropy_h0': '#2196F3', 'tda_wasserstein_h0': '#9C27B0'}

    for m in methods_to_test:
        ax.plot(window_sizes, results[m], marker='o', label=METHODS_DISPLAY.get(m, m),
                color=colors.get(m, 'black'), linewidth=2)

    ax.set_xlabel('Window Size (samples)')
    ax.set_ylabel('Mean Drift Score Ratio (drift/no-drift)')
    ax.set_title('Sensitivity vs Window Size\n(Abrupt Topic Drift; higher = more sensitive)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_persistence_examples(embeddings_by_class: dict, save_path: Path):
    """Figure 5: Example persistence diagrams before and after drift."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    rng = np.random.default_rng(42)

    # Before drift: class 0 (World)
    pts_ref = embeddings_by_class[0][rng.choice(len(embeddings_by_class[0]), size=SUBSAMPLE, replace=False)]
    # After drift: class 1 (Sports)
    pts_drift = embeddings_by_class[1][rng.choice(len(embeddings_by_class[1]), size=SUBSAMPLE, replace=False)]

    result_ref = ripser.ripser(pts_ref, maxdim=1, metric='euclidean')
    result_drift = ripser.ripser(pts_drift, maxdim=1, metric='euclidean')

    # Row 1: Reference (World news)
    ax = axes[0, 0]
    persim.plot_diagrams(result_ref['dgms'], ax=ax, title='Persistence Diagram\n(Reference: World news)')

    # H0 lifetime histogram - reference
    ax = axes[0, 1]
    h0_ref = result_ref['dgms'][0]
    h0_ref_finite = h0_ref[h0_ref[:, 1] != np.inf]
    if len(h0_ref_finite) > 0:
        lifetimes_ref = h0_ref_finite[:, 1] - h0_ref_finite[:, 0]
        ax.hist(lifetimes_ref, bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.set_xlabel('H0 Lifetime')
    ax.set_ylabel('Count')
    ax.set_title(f'H0 Lifetimes (Reference)\nEntropy={compute_persistence_h0(pts_ref)["entropy"]:.3f}')
    ax.grid(True, alpha=0.3)

    # Betti-0 curve - reference
    ax = axes[0, 2]
    if len(h0_ref_finite) > 0:
        max_birth_death = max(h0_ref_finite[:, 1].max(), 0.1)
        filtration_vals = np.linspace(0, max_birth_death * 1.1, 100)
        betti0_ref = [np.sum((h0_ref_finite[:, 0] <= t) & (h0_ref_finite[:, 1] > t))
                      for t in filtration_vals]
        ax.plot(filtration_vals, betti0_ref, color='#2196F3', linewidth=2)
    ax.set_xlabel('Filtration Parameter')
    ax.set_ylabel('Betti-0 Number')
    ax.set_title('Betti-0 Curve (Reference)')
    ax.grid(True, alpha=0.3)

    # Row 2: After drift (Sports)
    ax = axes[1, 0]
    persim.plot_diagrams(result_drift['dgms'], ax=ax, title='Persistence Diagram\n(After Drift: Sports news)')

    # H0 lifetime histogram - drift
    ax = axes[1, 1]
    h0_drift = result_drift['dgms'][0]
    h0_drift_finite = h0_drift[h0_drift[:, 1] != np.inf]
    if len(h0_drift_finite) > 0:
        lifetimes_drift = h0_drift_finite[:, 1] - h0_drift_finite[:, 0]
        ax.hist(lifetimes_drift, bins=20, color='#FF5722', alpha=0.7, edgecolor='white')
    ax.set_xlabel('H0 Lifetime')
    ax.set_ylabel('Count')
    ax.set_title(f'H0 Lifetimes (After Drift)\nEntropy={compute_persistence_h0(pts_drift)["entropy"]:.3f}')
    ax.grid(True, alpha=0.3)

    # Betti-0 curve - drift
    ax = axes[1, 2]
    if len(h0_drift_finite) > 0:
        max_bd = max(h0_drift_finite[:, 1].max(), 0.1)
        filt_vals = np.linspace(0, max_bd * 1.1, 100)
        betti0_drift = [np.sum((h0_drift_finite[:, 0] <= t) & (h0_drift_finite[:, 1] > t))
                        for t in filt_vals]
        ax.plot(filt_vals, betti0_drift, color='#FF5722', linewidth=2)
    ax.set_xlabel('Filtration Parameter')
    ax.set_ylabel('Betti-0 Number')
    ax.set_title('Betti-0 Curve (After Drift)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Topological Features Before and After Topic Drift\n(World News → Sports News)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_score_traces(all_results: list, save_path: Path):
    """Figure 6: Score traces over time for different scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    scenarios_to_plot = ['abrupt_topic', 'gradual_topic', 'geometric', 'style_shift']
    titles = {
        'abrupt_topic': 'Abrupt Topic Drift (World → Sports)',
        'gradual_topic': 'Gradual Topic Drift',
        'geometric': 'Geometric Drift (Multi-class mix)',
        'style_shift': 'Style/Domain Shift (World → Business)'
    }

    methods_to_plot = {
        'centroid': ('Centroid', '#FF5722', '-'),
        'mmd': ('MMD', '#FF9800', '--'),
        'tda_entropy_h0': ('TDA PE H0', '#2196F3', '-'),
        'tda_wasserstein_h0': ('TDA Wasserstein', '#9C27B0', '--'),
    }

    for ax, sc in zip(axes.flat, scenarios_to_plot):
        # Get first seed's results for this scenario
        sc_results = [r for r in all_results if r['scenario'] == sc and r['seed'] == 42]
        if not sc_results:
            continue

        scores_list = sc_results[0]['scores']
        labels = sc_results[0]['labels']

        n_windows = len(scores_list)
        x = range(n_windows)

        # Plot drift boundary
        if 1 in labels:
            first_drift_idx = labels.index(1)
            ax.axvline(x=first_drift_idx - 0.5, color='red', linestyle=':',
                      linewidth=2, alpha=0.7, label='Drift start')
            ax.axvspan(first_drift_idx, n_windows, alpha=0.05, color='red')

        # Plot each method (normalized)
        for method, (name, color, ls) in methods_to_plot.items():
            vals = np.array([s[method] for s in scores_list])
            if vals.std() > 0:
                vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
            else:
                vals_norm = vals
            ax.plot(x, vals_norm, label=name, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)

        ax.set_title(titles.get(sc, sc), fontsize=10)
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Normalized Score')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Drift Score Traces Over Time\n(Scores normalized to [0,1]; dashed line = drift injection)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("Topological Drift Detection Experiment")
    logger.info("=" * 70)

    t_start_all = time.time()

    # --- Environment info ---
    import sklearn
    import sentence_transformers
    env_info = {
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__,
        'sklearn_version': sklearn.__version__,
        'ripser_version': ripser.__version__,
        'persim_version': persim.__version__,
        'sentence_transformers_version': sentence_transformers.__version__,
        'timestamp': datetime.now().isoformat(),
        'device': 'cpu',
        'seeds': SEEDS,
        'window_size': WINDOW_SIZE,
        'subsample_for_tda': SUBSAMPLE,
    }
    logger.info(f"Environment: {env_info}")

    with open(RESULTS_DIR / 'environment.json', 'w') as f:
        json.dump(env_info, f, indent=2)

    # --- Load data ---
    logger.info("\nPhase 1: Loading embeddings...")
    embeddings_by_class = load_ag_news_embeddings(seed=42)

    # --- Run experiments ---
    logger.info("\nPhase 2: Running experiments...")
    all_results = []

    for scenario in DRIFT_SCENARIOS:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Scenario: {scenario}")
        for seed in SEEDS:
            result = run_scenario(scenario, embeddings_by_class, seed)
            all_results.append(result)

    # --- Compute metrics ---
    logger.info("\nPhase 3: Computing metrics...")

    all_methods = list(METHODS_DISPLAY.keys())
    metrics_rows = []

    for result in all_results:
        scenario = result['scenario']
        seed = result['seed']
        scores_list = result['scores']
        labels = result['labels']

        for method in all_methods:
            m = compute_detection_metrics(scores_list, labels, method)
            m['scenario'] = scenario
            m['seed'] = seed
            m['window_size'] = WINDOW_SIZE

            # Add mean runtime for this method
            if result['timings']:
                method_key = method.split('_')[0] if '_' in method else method
                # Get matching timing key
                if method in ['tda_entropy_h0', 'tda_betti_auc', 'tda_max_lifetime']:
                    rt_key = 'tda_h0'
                elif method == 'tda_wasserstein_h0':
                    rt_key = 'tda_wasserstein'
                elif method == 'tda_phd':
                    rt_key = 'tda_phd'
                elif method in ['tda_entropy_h1', 'tda_h1_count']:
                    rt_key = 'tda_h1'
                else:
                    rt_key = method

                rts = [t.get(rt_key, np.nan) for t in result['timings']]
                m['runtime'] = float(np.nanmean(rts))
            else:
                m['runtime'] = float('nan')

            metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)

    # Save metrics
    metrics_df.to_csv(RESULTS_DIR / 'metrics.csv', index=False)

    # Save metrics as JSON
    metrics_json = []
    for _, row in metrics_df.iterrows():
        metrics_json.append({
            'method': row['method'],
            'drift_type': row['scenario'],
            'window_size': row['window_size'],
            'detection_accuracy': float(row['detection_accuracy']) if not pd.isna(row['detection_accuracy']) else None,
            'detection_delay': float(row['detection_delay']) if not pd.isna(row['detection_delay']) else None,
            'false_positive_rate': float(row['fpr']) if not pd.isna(row['fpr']) else None,
            'auc': float(row['auc']) if not pd.isna(row['auc']) else None,
            'tpr': float(row['tpr']) if not pd.isna(row['tpr']) else None,
            'seed': int(row['seed']),
            'runtime': float(row['runtime']) if not pd.isna(row['runtime']) else None,
        })

    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    logger.info(f"  Saved metrics ({len(metrics_json)} rows)")

    # Save raw results as JSONL
    logger.info("  Saving raw results (JSONL)...")
    with open(RESULTS_DIR / 'raw_results.jsonl', 'w') as f:
        for result in all_results:
            for score_dict in result['scores']:
                # Remove non-serializable items
                row = {k: v for k, v in score_dict.items()
                       if isinstance(v, (int, float, str, bool, type(None)))}
                f.write(json.dumps(row) + '\n')

    # --- Visualizations ---
    logger.info("\nPhase 4: Generating visualizations...")

    plot_detection_by_drift_type(metrics_df, FIGURES_DIR / 'fig1_detection_by_drift_type.png')
    plot_delay_vs_fpr(metrics_df, FIGURES_DIR / 'fig2_delay_vs_fpr.png')
    plot_tda_vs_statistical(metrics_df, FIGURES_DIR / 'fig3_tda_vs_statistical.png')
    plot_sensitivity_vs_window_size(embeddings_by_class, FIGURES_DIR / 'fig4_sensitivity_vs_window_size.png')
    plot_persistence_examples(embeddings_by_class, FIGURES_DIR / 'fig5_persistence_examples.png')
    plot_score_traces(all_results, FIGURES_DIR / 'fig6_score_traces.png')

    # --- Print summary ---
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)

    # Mean AUC per method (excluding no_drift scenario)
    summary = metrics_df[metrics_df['scenario'] != 'no_drift'].groupby('method').agg(
        mean_auc=('auc', 'mean'),
        mean_delay=('detection_delay', 'mean'),
        mean_fpr=('fpr', 'mean'),
        mean_runtime=('runtime', 'mean')
    ).sort_values('mean_auc', ascending=False)

    logger.info("\nMean AUC across drift scenarios:")
    for method, row in summary.iterrows():
        flag = " [TDA]" if method in TDA_METHODS else " [STAT]"
        logger.info(f"  {METHODS_DISPLAY.get(method, method):30s}{flag}: "
                   f"AUC={row['mean_auc']:.3f}, "
                   f"delay={row['mean_delay']:.1f}, "
                   f"FPR={row['mean_fpr']:.3f}, "
                   f"rt={row['mean_runtime']:.3f}s")

    # Per-scenario best TDA vs best stat
    logger.info("\nBest TDA vs Best Statistical per scenario:")
    for sc in [s for s in DRIFT_SCENARIOS if s != 'no_drift']:
        sc_data = metrics_df[metrics_df['scenario'] == sc]
        best_tda = sc_data[sc_data['method'].isin(TDA_METHODS)]['auc'].max()
        best_stat = sc_data[sc_data['method'].isin(STAT_METHODS)]['auc'].max()
        winner = "TDA WINS" if best_tda > best_stat else "STAT WINS"
        logger.info(f"  {sc:20s}: TDA={best_tda:.3f}, STAT={best_stat:.3f} → {winner}")

    total_time = time.time() - t_start_all
    logger.info(f"\nTotal experiment time: {total_time:.1f}s")

    # Save summary
    summary.to_csv(RESULTS_DIR / 'summary.csv')

    logger.info("\n✓ Experiment complete. Results saved to results/ and figures/")

    return metrics_df, all_results


if __name__ == '__main__':
    metrics_df, all_results = main()

"""
Topological Drift Detection for Continuous Monitoring of LLM Embedding Streams
==============================================================================
Improved experiment v2 with:
- Centroid-preserving drift on real embeddings
- 20 Newsgroups with closely-related categories
- AG News with multiple drift types
- Larger TDA subsample (150 pts)
- Multiple window sizes (100, 200, 400)
- 5 seeds for robust statistics
- Persistence landscapes + normalized features
"""

import os, sys, json, time, random, warnings, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import ripser
import persim

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

WORKSPACE = Path('/workspaces/topo-drift-llm-embeds-5b03')
RESULTS_DIR = WORKSPACE / 'results_v2'
FIGURES_DIR = WORKSPACE / 'figures_v2'

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
TDA_SUBSAMPLE = 150
N_REF_WINDOWS = 10
N_DRIFT_WINDOWS = 10
WINDOW_SIZES = [100, 200, 400]
DEFAULT_WINDOW = 200
N_PER_CLASS = 3000  # samples per class to encode

# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ─── DATA LOADING ────────────────────────────────────────────────────────────

_embedding_cache = {}

def load_embeddings(dataset_name='ag_news', seed=42):
    """Load and cache embeddings. Returns dict: class_id -> np.array [N, dim]"""
    cache_key = dataset_name
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    set_seed(seed)
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info(f"Loaded sentence-transformer model (384-dim)")

    if dataset_name == 'ag_news':
        ds = load_dataset('ag_news', split='train')
        texts_by_class = {c: [] for c in range(4)}
        for item in ds:
            texts_by_class[item['label']].append(item['text'])
        class_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'SciTech'}

    elif dataset_name == '20newsgroups':
        ds = load_dataset('SetFit/20_newsgroups', split='train')
        # Pick 6 categories: 3 pairs of closely-related topics
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
                # Need to map int label to text
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
        logger.info(f"  Encoding class {c} ({len(texts)} texts)...")
        t0 = time.time()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
        logger.info(f"    -> shape {embs.shape}, {time.time()-t0:.1f}s")
        embeddings_by_class[c] = embs

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
        # World -> Sports (very distinct)
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
        # World -> Business (closer domains)
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(2, window_size), 1))

    elif scenario == 'centroid_preserving':
        # KEY EXPERIMENT: Both reference and drift windows have same centroid
        # Reference: class 0 centered
        # Drift: mix of class 0 + class 1 centered to same mean
        # The mix creates a bimodal structure (topology change) without centroid shift
        ref_centroid = embs[0].mean(axis=0)
        drift_centroid_raw = np.vstack([embs[0][:N_PER_CLASS//2], embs[1][:N_PER_CLASS//2]]).mean(axis=0)
        shift = ref_centroid - drift_centroid_raw

        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            n_each = window_size // 2
            w = np.vstack([sample(0, n_each), sample(1, n_each)])
            w = w + shift  # shift so centroid matches reference
            # Re-normalize to unit sphere
            norms = np.linalg.norm(w, axis=1, keepdims=True)
            w = w / (norms + 1e-10)
            rng.shuffle(w)
            windows.append((w, 1))

    elif scenario == 'subtle_gradual':
        # Very gradual: only 5% contamination per step (50 steps worth but we use 10 windows)
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for i in range(N_DRIFT_WINDOWS):
            frac = 0.05 * (i + 1)  # 5% to 50%
            n1 = max(1, int(frac * window_size))
            n0 = window_size - n1
            parts = [sample(0, n0), sample(1, n1)]
            w = np.vstack(parts)
            rng.shuffle(w)
            windows.append((w, 1 if frac > 0.1 else 0))

    elif scenario == 'rotation_drift':
        # Apply random rotation to embeddings in PCA subspace (geometric change, centroid preserved)
        pca = PCA(n_components=50).fit(embs[0])
        ref_proj = pca.transform(embs[0])
        # Random rotation in first 10 PCA dimensions
        from scipy.stats import special_ortho_group
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
        # Close categories: comp.sys.mac.hardware -> comp.sys.ibm.pc.hardware
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(1, window_size), 1))

    elif scenario == 'newsgroup_distant':
        # Distant categories: comp.sys.mac.hardware -> sci.space
        for _ in range(N_REF_WINDOWS):
            windows.append((sample(0, window_size), 0))
        for _ in range(N_DRIFT_WINDOWS):
            windows.append((sample(5, window_size), 1))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return windows


# ─── DRIFT DETECTORS ─────────────────────────────────────────────────────────

def centroid_shift(ref, test):
    """L2 distance between mean embeddings."""
    return float(np.linalg.norm(ref.mean(axis=0) - test.mean(axis=0)))

def covariance_shift(ref, test):
    """Frobenius norm of PCA-reduced covariance difference."""
    k = min(50, min(ref.shape[0], test.shape[0]) - 1, ref.shape[1])
    pca = PCA(n_components=k).fit(np.vstack([ref, test]))
    r = pca.transform(ref)
    t = pca.transform(test)
    cov_r = np.cov(r, rowvar=False)
    cov_t = np.cov(t, rowvar=False)
    return float(np.linalg.norm(cov_r - cov_t, 'fro'))

def mmd_rbf(ref, test, gamma=None):
    """Maximum Mean Discrepancy with RBF kernel (median heuristic)."""
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
    """Change in mean k-NN distance."""
    def mean_knn(X):
        D = cdist(X, X, 'euclidean')
        np.fill_diagonal(D, np.inf)
        knn_dists = np.sort(D, axis=1)[:, :k]
        return knn_dists.mean()
    return abs(mean_knn(test) - mean_knn(ref))

def energy_distance(ref, test):
    """Energy distance between two samples."""
    n, m = len(ref), len(test)
    d_rt = cdist(ref, test, 'euclidean').mean()
    d_rr = pdist(ref, 'euclidean').mean() if n > 1 else 0
    d_tt = pdist(test, 'euclidean').mean() if m > 1 else 0
    return float(2 * d_rt - d_rr - d_tt)


# ─── TDA FEATURES ────────────────────────────────────────────────────────────

def compute_tda_features(X, subsample=TDA_SUBSAMPLE, seed=42):
    """
    Compute comprehensive TDA features from a point cloud.
    Returns dict of scalar features.
    """
    rng = np.random.default_rng(seed)
    pts = X[rng.choice(len(X), size=min(subsample, len(X)), replace=False)]

    result = ripser.ripser(pts, maxdim=1, metric='euclidean')
    dgms = result['dgms']

    features = {}

    # H0 features
    h0 = dgms[0]
    h0_fin = h0[h0[:, 1] != np.inf]
    if len(h0_fin) > 0:
        lifetimes_h0 = h0_fin[:, 1] - h0_fin[:, 0]
        lifetimes_h0 = lifetimes_h0[lifetimes_h0 > 0]
    else:
        lifetimes_h0 = np.array([0.0])

    # H0 persistent entropy
    if len(lifetimes_h0) > 0 and lifetimes_h0.sum() > 0:
        p = lifetimes_h0 / lifetimes_h0.sum()
        features['pe_h0'] = float(-np.sum(p * np.log(p + 1e-10)))
    else:
        features['pe_h0'] = 0.0

    features['h0_total_persistence'] = float(lifetimes_h0.sum())
    features['h0_max_lifetime'] = float(lifetimes_h0.max()) if len(lifetimes_h0) > 0 else 0.0
    features['h0_mean_lifetime'] = float(lifetimes_h0.mean()) if len(lifetimes_h0) > 0 else 0.0
    features['h0_std_lifetime'] = float(lifetimes_h0.std()) if len(lifetimes_h0) > 1 else 0.0
    features['h0_n_features'] = len(lifetimes_h0)

    # H1 features (loops/holes)
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    h1_fin = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
    if len(h1_fin) > 0:
        lifetimes_h1 = h1_fin[:, 1] - h1_fin[:, 0]
        lifetimes_h1 = lifetimes_h1[lifetimes_h1 > 0]
    else:
        lifetimes_h1 = np.array([])

    if len(lifetimes_h1) > 0 and lifetimes_h1.sum() > 0:
        p1 = lifetimes_h1 / lifetimes_h1.sum()
        features['pe_h1'] = float(-np.sum(p1 * np.log(p1 + 1e-10)))
    else:
        features['pe_h1'] = 0.0

    features['h1_total_persistence'] = float(lifetimes_h1.sum()) if len(lifetimes_h1) > 0 else 0.0
    features['h1_n_features'] = len(lifetimes_h1)
    features['h1_max_lifetime'] = float(lifetimes_h1.max()) if len(lifetimes_h1) > 0 else 0.0

    # PHD (persistent homology dimension) via MST
    D = squareform(pdist(pts, 'euclidean'))
    mst = minimum_spanning_tree(D).toarray()
    mst_weights = mst[mst > 0]
    if len(mst_weights) > 2:
        sorted_w = np.sort(mst_weights)
        n_pts = np.arange(1, len(sorted_w) + 1)
        # Log-log regression: log(weight) vs log(index)
        log_n = np.log(n_pts + 1)
        log_w = np.log(sorted_w + 1e-10)
        # Linear fit
        coeffs = np.polyfit(log_n, log_w, 1)
        features['phd'] = float(-coeffs[0])  # negative slope = dimension estimate
    else:
        features['phd'] = 0.0

    features['_diagrams'] = dgms
    return features


def tda_drift_scores(ref_features, test_features):
    """Compute drift scores between two sets of TDA features."""
    scores = {}

    # Absolute difference of scalar features
    for key in ['pe_h0', 'pe_h1', 'h0_total_persistence', 'h1_total_persistence',
                'h0_max_lifetime', 'h1_max_lifetime', 'h1_n_features', 'phd',
                'h0_mean_lifetime', 'h0_std_lifetime']:
        scores[f'tda_{key}'] = abs(ref_features[key] - test_features[key])

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

    return scores


# ─── EXPERIMENT RUNNER ────────────────────────────────────────────────────────

def run_experiment(dataset_name, scenarios, window_size=DEFAULT_WINDOW, seeds=SEEDS):
    """Run full experiment for a dataset and set of scenarios."""
    logger.info(f"\n{'='*70}")
    logger.info(f"DATASET: {dataset_name}, WINDOW: {window_size}")
    logger.info(f"{'='*70}")

    embs = load_embeddings(dataset_name)
    all_results = []

    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario} ---")
        for seed in seeds:
            set_seed(seed)
            t0_scenario = time.time()

            stream = build_stream(scenario, embs, window_size, seed)
            labels = [lbl for _, lbl in stream]

            # Compute reference pool: aggregate embeddings from ref windows
            ref_embs = np.vstack([w for w, lbl in stream[:N_REF_WINDOWS]])

            # Compute drift scores for each window
            stat_scores = {m: [] for m in ['centroid', 'covariance', 'mmd', 'knn', 'energy']}
            tda_score_keys = None
            tda_scores_all = {}

            # Pre-compute reference TDA features (use stable seed)
            ref_tda = compute_tda_features(ref_embs, subsample=TDA_SUBSAMPLE, seed=seed)

            for wi, (w_embs, w_label) in enumerate(stream):
                # Statistical detectors
                t_stat = time.time()
                stat_scores['centroid'].append(centroid_shift(ref_embs, w_embs))
                stat_scores['covariance'].append(covariance_shift(ref_embs, w_embs))
                stat_scores['mmd'].append(mmd_rbf(ref_embs[:200], w_embs[:200]))  # subsample for speed
                stat_scores['knn'].append(knn_shift(w_embs[:100], ref_embs[:100]))
                stat_scores['energy'].append(energy_distance(ref_embs[:100], w_embs[:100]))
                t_stat_elapsed = time.time() - t_stat

                # TDA detectors
                t_tda = time.time()
                w_tda = compute_tda_features(w_embs, subsample=TDA_SUBSAMPLE, seed=seed + wi)
                tda_diffs = tda_drift_scores(ref_tda, w_tda)
                t_tda_elapsed = time.time() - t_tda

                if tda_score_keys is None:
                    tda_score_keys = sorted(tda_diffs.keys())
                    for k in tda_score_keys:
                        tda_scores_all[k] = []

                for k in tda_score_keys:
                    tda_scores_all[k].append(tda_diffs[k])

            t_total = time.time() - t0_scenario

            # Combine all methods
            all_methods = {}
            all_methods.update(stat_scores)
            all_methods.update(tda_scores_all)

            # Compute AUC and detection metrics for each method
            for method_name, scores_list in all_methods.items():
                scores_arr = np.array(scores_list)
                labels_arr = np.array(labels)

                # Only compute AUC on drift scenarios (skip no_drift for AUC)
                if scenario == 'no_drift':
                    # For no_drift: measure false positive rate
                    # Use 95th percentile of reference windows as threshold
                    ref_scores = scores_arr[:N_REF_WINDOWS]
                    thresh = np.percentile(ref_scores, 95) if len(ref_scores) > 0 else 0
                    test_scores = scores_arr[N_REF_WINDOWS:]
                    fpr = float((test_scores > thresh).mean()) if len(test_scores) > 0 else 0
                    result = {
                        'dataset': dataset_name,
                        'method': method_name,
                        'drift_type': scenario,
                        'window_size': window_size,
                        'seed': seed,
                        'auc': None,
                        'detection_delay': None,
                        'fpr': fpr,
                        'runtime': t_total / len(stream),
                    }
                else:
                    # Compute AUC
                    try:
                        if len(set(labels_arr)) > 1:
                            auc = roc_auc_score(labels_arr, scores_arr)
                        else:
                            auc = None
                    except:
                        auc = None

                    # Detection delay: threshold at 95th percentile of reference
                    ref_scores = scores_arr[:N_REF_WINDOWS]
                    thresh = np.percentile(ref_scores, 95)
                    drift_scores = scores_arr[N_REF_WINDOWS:]
                    drift_detected = np.where(drift_scores > thresh)[0]
                    delay = int(drift_detected[0]) if len(drift_detected) > 0 else N_DRIFT_WINDOWS

                    # FPR on reference windows
                    fpr = float((ref_scores > thresh).mean())

                    result = {
                        'dataset': dataset_name,
                        'method': method_name,
                        'drift_type': scenario,
                        'window_size': window_size,
                        'seed': seed,
                        'auc': auc,
                        'detection_delay': delay,
                        'fpr': fpr,
                        'runtime': t_total / len(stream),
                    }

                all_results.append(result)

            logger.info(f"  Seed {seed}: {t_total:.1f}s total")

    return all_results


# ─── SYNTHETIC TOPOLOGY EXPERIMENT ───────────────────────────────────────────

def run_synthetic_experiment(seeds=SEEDS):
    """
    Controlled synthetic experiment where we know ground truth topology.
    Tests: blob, annulus (H1 loop), two-cluster (H0 split), swiss roll.
    """
    logger.info("\n" + "="*70)
    logger.info("SYNTHETIC TOPOLOGY EXPERIMENT")
    logger.info("="*70)

    dim = 50
    n_points = 200
    results = []

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Reference: isotropic Gaussian blob
        ref = rng.standard_normal((n_points, dim)) * 0.5

        # Drift types
        drift_configs = {
            'centroid_shift': lambda: ref + 3.0 * np.eye(dim)[0],  # shift mean
            'annulus': lambda: _make_annulus(n_points, dim, rng),
            'two_cluster': lambda: _make_two_cluster(n_points, dim, rng),
            'variance_change': lambda: rng.standard_normal((n_points, dim)) * 1.5,
            'no_drift': lambda: rng.standard_normal((n_points, dim)) * 0.5,
        }

        ref_tda = compute_tda_features(ref, subsample=TDA_SUBSAMPLE, seed=seed)

        for drift_name, drift_fn in drift_configs.items():
            drift_data = drift_fn()

            # Statistical scores
            c_score = centroid_shift(ref, drift_data)
            cov_score = covariance_shift(ref, drift_data)
            mmd_score = mmd_rbf(ref, drift_data)
            en_score = energy_distance(ref, drift_data)

            # TDA scores
            drift_tda = compute_tda_features(drift_data, subsample=TDA_SUBSAMPLE, seed=seed+1)
            tda_diffs = tda_drift_scores(ref_tda, drift_tda)

            row = {
                'drift_type': drift_name,
                'seed': seed,
                'centroid': c_score,
                'covariance': cov_score,
                'mmd': mmd_score,
                'energy': en_score,
            }
            row.update(tda_diffs)
            results.append(row)

    return pd.DataFrame(results)


def _make_annulus(n, dim, rng):
    """Create annular point cloud (has H1 hole) in first 2 dims, noise in rest."""
    theta = rng.uniform(0, 2*np.pi, n)
    r = rng.uniform(2.0, 3.0, n)  # annulus with inner radius 2, outer 3
    pts = rng.standard_normal((n, dim)) * 0.1  # small noise in all dims
    pts[:, 0] = r * np.cos(theta)
    pts[:, 1] = r * np.sin(theta)
    return pts

def _make_two_cluster(n, dim, rng):
    """Two symmetric clusters (same centroid = 0)."""
    half = n // 2
    pts = rng.standard_normal((n, dim)) * 0.3
    pts[:half, 0] += 2.0   # cluster 1 shifted right
    pts[half:, 0] -= 2.0   # cluster 2 shifted left (centroid ~ 0)
    return pts


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def plot_main_results(df, figdir):
    """Generate all main result figures."""
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)

    # Classify methods
    stat_methods = ['centroid', 'covariance', 'mmd', 'knn', 'energy']
    tda_methods = [m for m in df['method'].unique() if m.startswith('tda_')]

    # Select key TDA methods for cleaner plots
    key_tda = ['tda_pe_h0', 'tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1',
               'tda_phd', 'tda_h1_total_persistence', 'tda_bottleneck_h0']
    key_tda = [m for m in key_tda if m in df['method'].unique()]
    key_methods = stat_methods + key_tda

    # ── Fig 1: AUC heatmap by method × drift type ──
    drift_df = df[(df['drift_type'] != 'no_drift') & (df['auc'].notna()) &
                  (df['method'].isin(key_methods))]
    if len(drift_df) > 0:
        pivot = drift_df.groupby(['method', 'drift_type'])['auc'].mean().reset_index()
        pivot_wide = pivot.pivot(index='method', columns='drift_type', values='auc')

        # Sort methods: stat first, then TDA
        method_order = [m for m in stat_methods if m in pivot_wide.index] + \
                       [m for m in key_tda if m in pivot_wide.index]
        pivot_wide = pivot_wide.reindex(method_order)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_wide, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.4, vmax=1.0,
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'AUC'})
        ax.set_title('Detection AUC by Method and Drift Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Detector Method')
        ax.set_xlabel('Drift Scenario')
        plt.tight_layout()
        plt.savefig(figdir / 'fig1_auc_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 2: Detection delay comparison ──
    delay_df = drift_df[drift_df['detection_delay'].notna()]
    if len(delay_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        delay_agg = delay_df.groupby(['method', 'drift_type'])['detection_delay'].mean().reset_index()
        # Bar chart
        methods_in_data = [m for m in key_methods if m in delay_agg['method'].unique()]
        x = np.arange(len(methods_in_data))
        width = 0.8 / len(delay_agg['drift_type'].unique())
        for i, dt in enumerate(sorted(delay_agg['drift_type'].unique())):
            vals = []
            for m in methods_in_data:
                v = delay_agg[(delay_agg['method']==m) & (delay_agg['drift_type']==dt)]['detection_delay']
                vals.append(v.values[0] if len(v) > 0 else N_DRIFT_WINDOWS)
            ax.bar(x + i*width, vals, width, label=dt, alpha=0.8)
        ax.set_xticks(x + width * (len(delay_agg['drift_type'].unique())-1)/2)
        ax.set_xticklabels(methods_in_data, rotation=45, ha='right')
        ax.set_ylabel('Detection Delay (windows)')
        ax.set_title('Detection Delay by Method and Drift Type', fontweight='bold')
        ax.legend(title='Drift Type', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(figdir / 'fig2_detection_delay.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 3: TDA vs Statistical scatter (AUC) ──
    if len(drift_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: mean AUC bar comparison
        method_means = drift_df.groupby('method')['auc'].mean().sort_values(ascending=True)
        colors = ['#2196F3' if m in stat_methods else '#FF5722' for m in method_means.index]
        method_means.plot.barh(ax=axes[0], color=colors)
        axes[0].set_xlabel('Mean AUC across drift scenarios')
        axes[0].set_title('Overall Detection Performance')
        axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        # Legend
        from matplotlib.patches import Patch
        axes[0].legend([Patch(color='#2196F3'), Patch(color='#FF5722')],
                      ['Statistical', 'TDA'], loc='lower right')

        # Right: mean delay bar
        if len(delay_df) > 0:
            delay_means = delay_df.groupby('method')['detection_delay'].mean().sort_values(ascending=False)
            colors2 = ['#2196F3' if m in stat_methods else '#FF5722' for m in delay_means.index]
            delay_means.plot.barh(ax=axes[1], color=colors2)
            axes[1].set_xlabel('Mean Detection Delay (windows)')
            axes[1].set_title('Detection Delay')

        plt.suptitle('TDA vs Statistical Drift Detectors', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(figdir / 'fig3_tda_vs_stat.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ── Fig 4: FPR calibration ──
    nodrift_df = df[df['drift_type'] == 'no_drift']
    if len(nodrift_df) > 0:
        fpr_means = nodrift_df.groupby('method')['fpr'].mean()
        fpr_means = fpr_means[fpr_means.index.isin(key_methods)].sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors3 = ['#2196F3' if m in stat_methods else '#FF5722' for m in fpr_means.index]
        fpr_means.plot.barh(ax=ax, color=colors3)
        ax.axvline(x=0.05, color='red', linestyle='--', label='Target FPR=0.05')
        ax.set_xlabel('False Positive Rate (no-drift windows)')
        ax.set_title('FPR Calibration on No-Drift Data', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(figdir / 'fig4_fpr_calibration.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_synthetic_results(synth_df, figdir):
    """Plot synthetic topology experiment results."""
    # Normalize each score by no_drift baseline for separability ratios
    methods = ['centroid', 'covariance', 'mmd', 'energy',
               'tda_pe_h0', 'tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1',
               'tda_phd', 'tda_h1_total_persistence', 'tda_bottleneck_h0']
    methods = [m for m in methods if m in synth_df.columns]

    # Get baseline scores (no_drift)
    baseline = synth_df[synth_df['drift_type'] == 'no_drift'][methods].mean()

    # Separability ratios
    drift_types = [d for d in synth_df['drift_type'].unique() if d != 'no_drift']

    sep_data = []
    for dt in drift_types:
        dt_means = synth_df[synth_df['drift_type'] == dt][methods].mean()
        for m in methods:
            ratio = dt_means[m] / (baseline[m] + 1e-10)
            sep_data.append({'drift_type': dt, 'method': m, 'separability': ratio})

    sep_df = pd.DataFrame(sep_data)

    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_sep = sep_df.pivot(index='method', columns='drift_type', values='separability')
    stat_idx = [m for m in methods if not m.startswith('tda_')]
    tda_idx = [m for m in methods if m.startswith('tda_')]
    order = stat_idx + tda_idx
    pivot_sep = pivot_sep.reindex([m for m in order if m in pivot_sep.index])

    sns.heatmap(pivot_sep, annot=True, fmt='.1f', cmap='YlOrRd', vmin=0, vmax=20,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Separability Ratio (vs no-drift)'})
    ax.set_title('Synthetic Topology Experiment: Separability Ratios\n(Higher = Better Detection)',
                fontsize=13, fontweight='bold')
    ax.axhline(y=len(stat_idx), color='black', linewidth=2)
    ax.text(-0.5, len(stat_idx)/2, 'Statistical', ha='right', va='center', fontweight='bold', fontsize=10)
    ax.text(-0.5, len(stat_idx) + len(tda_idx)/2, 'TDA', ha='right', va='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(figdir / 'fig5_synthetic_separability.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_window_size_sensitivity(ws_results, figdir):
    """Plot AUC vs window size."""
    df = pd.DataFrame(ws_results)
    drift_df = df[(df['drift_type'] != 'no_drift') & (df['auc'].notna())]
    if len(drift_df) == 0:
        return

    stat_methods = ['centroid', 'covariance', 'mmd', 'knn', 'energy']
    key_tda = ['tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1', 'tda_phd']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, methods, title in [(axes[0], stat_methods, 'Statistical Methods'),
                                (axes[1], key_tda, 'TDA Methods')]:
        for m in methods:
            mdf = drift_df[drift_df['method'] == m]
            if len(mdf) == 0:
                continue
            means = mdf.groupby('window_size')['auc'].mean()
            stds = mdf.groupby('window_size')['auc'].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                       marker='o', label=m, capsize=3)
        ax.set_xlabel('Window Size')
        ax.set_ylabel('Mean AUC')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    plt.suptitle('Detection AUC vs Window Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figdir / 'fig6_window_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_persistence_examples(embs, figdir):
    """Plot example persistence diagrams for reference vs drifted data."""
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

        result = ripser.ripser(pts, maxdim=1, metric='euclidean')
        dgms = result['dgms']

        # H0 diagram
        ax0 = axes[0, col]
        h0 = dgms[0]
        h0_fin = h0[h0[:, 1] != np.inf]
        if len(h0_fin) > 0:
            ax0.scatter(h0_fin[:, 0], h0_fin[:, 1], s=20, alpha=0.6, c='blue')
        max_val = max(h0_fin[:, 1].max(), h0_fin[:, 0].max()) if len(h0_fin) > 0 else 1
        ax0.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax0.set_xlabel('Birth')
        ax0.set_ylabel('Death')
        ax0.set_title(f'{title}\nH0 ({len(h0_fin)} features)')

        # H1 diagram
        ax1 = axes[1, col]
        h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        h1_fin = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
        if len(h1_fin) > 0:
            ax1.scatter(h1_fin[:, 0], h1_fin[:, 1], s=20, alpha=0.6, c='red')
            max_val1 = max(h1_fin[:, 1].max(), h1_fin[:, 0].max())
            ax1.plot([0, max_val1], [0, max_val1], 'k--', alpha=0.3)
        ax1.set_xlabel('Birth')
        ax1.set_ylabel('Death')
        ax1.set_title(f'H1 ({len(h1_fin)} features)')

    plt.suptitle('Persistence Diagrams: Reference vs Drifted Embeddings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figdir / 'fig7_persistence_examples.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    logger.info("="*70)
    logger.info("Topological Drift Detection Experiment v2")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*70)

    # Save environment info
    import sys
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
        'gpu': 'none (CPU only)',
    }
    with open(RESULTS_DIR / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)

    # ── Experiment 1: AG News with multiple drift types ──
    ag_scenarios = ['no_drift', 'abrupt_topic', 'gradual_topic', 'style_shift',
                    'centroid_preserving', 'subtle_gradual', 'rotation_drift']
    ag_results = run_experiment('ag_news', ag_scenarios, window_size=DEFAULT_WINDOW)

    # ── Experiment 2: 20 Newsgroups (close vs distant categories) ──
    ng_scenarios = ['no_drift', 'newsgroup_close', 'newsgroup_distant']
    ng_results = run_experiment('20newsgroups', ng_scenarios, window_size=DEFAULT_WINDOW)

    all_results = ag_results + ng_results

    # ── Experiment 3: Window size sensitivity (AG News, key scenarios) ──
    ws_results = []
    for ws in WINDOW_SIZES:
        if ws == DEFAULT_WINDOW:
            # Already computed
            ws_results.extend([r for r in ag_results if r['drift_type'] in
                             ['abrupt_topic', 'centroid_preserving', 'no_drift']])
        else:
            ws_res = run_experiment('ag_news', ['abrupt_topic', 'centroid_preserving', 'no_drift'],
                                   window_size=ws)
            ws_results.extend(ws_res)
            all_results.extend(ws_res)

    # ── Experiment 4: Synthetic topology ──
    synth_df = run_synthetic_experiment()

    # ── Save results ──
    logger.info("\nSaving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    results_df.to_json(RESULTS_DIR / 'all_results.json', orient='records', indent=2)

    # JSONL for raw results
    with open(RESULTS_DIR / 'raw_results.jsonl', 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + '\n')

    # Metrics JSON (required output format)
    metrics_output = []
    for _, row in results_df.iterrows():
        metrics_output.append({
            'method': row['method'],
            'drift_type': row['drift_type'],
            'window_size': int(row['window_size']),
            'detection_accuracy': row['auc'],
            'detection_delay': row['detection_delay'],
            'false_positive_rate': row['fpr'],
            'runtime': row['runtime'],
        })
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2, default=str)

    # Summary table
    summary = results_df[results_df['auc'].notna()].groupby('method').agg(
        mean_auc=('auc', 'mean'),
        std_auc=('auc', 'std'),
        mean_delay=('detection_delay', 'mean'),
        mean_fpr=('fpr', 'mean'),
        mean_runtime=('runtime', 'mean'),
    ).sort_values('mean_auc', ascending=False)
    summary.to_csv(RESULTS_DIR / 'summary.csv')
    logger.info(f"\nSummary:\n{summary.to_string()}")

    # Synthetic results
    synth_df.to_csv(RESULTS_DIR / 'synthetic_results.csv', index=False)

    # ── Visualizations ──
    logger.info("\nGenerating visualizations...")
    plot_main_results(results_df, FIGURES_DIR)
    plot_synthetic_results(synth_df, FIGURES_DIR)
    plot_window_size_sensitivity(ws_results, FIGURES_DIR)

    # Persistence diagram examples
    embs = load_embeddings('ag_news')
    plot_persistence_examples(embs, FIGURES_DIR)

    t_total = time.time() - t_start
    logger.info(f"\nTotal experiment time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Figures saved to {FIGURES_DIR}")
    logger.info("DONE")


if __name__ == '__main__':
    main()

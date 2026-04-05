"""
Supplementary Synthetic Experiment: Topology-Only Drift
=======================================================

Creates controlled synthetic drift scenarios where the topology of the
point cloud changes WITHOUT changing the centroid or covariance structure.
This demonstrates the theoretical advantage of TDA over statistical methods.

Scenarios:
1. Blob → Blob (no drift): same Gaussian; all methods should have low score
2. Blob → Different Blob (centroid drift): all methods should detect
3. Blob → Annulus (topology drift): centroid stays near 0, but H1 loop appears
4. Blob → Two-cluster (H0 drift): same total points, but splits into 2 clusters
"""

import os
import json
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ripser
import persim
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WORKSPACE = Path('/workspaces/topo-drift-llm-embeds-5b03')
RESULTS_DIR = WORKSPACE / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC POINT CLOUD GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_blob(n, dim=10, center=None, scale=1.0, seed=None):
    """Gaussian blob in dim dimensions."""
    rng = np.random.default_rng(seed)
    if center is None:
        center = np.zeros(dim)
    return rng.normal(loc=center, scale=scale, size=(n, dim))


def make_annulus(n, dim=10, inner_r=1.5, outer_r=2.5, seed=None):
    """
    Annular distribution in first 2 dims, Gaussian noise in rest.
    Centroid ≈ (0, 0, ...) - same as blob centered at 0.
    Topology: H1 has 1 loop (the annular hole).
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, n)
    radii = rng.uniform(inner_r, outer_r, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    # Additional dim-2 Gaussian noise dims
    noise = rng.normal(0, 0.2, size=(n, max(dim-2, 0)))
    pts = np.column_stack([x, y, noise])
    return pts


def make_two_clusters(n, dim=10, separation=3.0, seed=None):
    """
    Two Gaussian clusters, symmetric about the origin.
    Centroid = (0, 0, ...) - same as single blob.
    Topology: H0 has 2 connected components (at intermediate filtration).
    """
    rng = np.random.default_rng(seed)
    n_each = n // 2
    c1 = np.zeros(dim)
    c2 = np.zeros(dim)
    c1[0] = separation / 2
    c2[0] = -separation / 2

    pts1 = rng.normal(loc=c1, scale=0.5, size=(n_each, dim))
    pts2 = rng.normal(loc=c2, scale=0.5, size=(n - n_each, dim))
    pts = np.vstack([pts1, pts2])
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTORS (simplified)
# ─────────────────────────────────────────────────────────────────────────────

def centroid_score(X_ref, X_test):
    return float(np.linalg.norm(X_ref.mean(0) - X_test.mean(0)))


def covariance_score(X_ref, X_test):
    cov_ref = np.cov(X_ref.T)
    cov_test = np.cov(X_test.T)
    return float(np.linalg.norm(cov_ref - cov_test, 'fro'))


def mmd_rbf(X_ref, X_test):
    XY = np.vstack([X_ref, X_test])
    dists = cdist(XY, XY, 'sqeuclidean')
    sigma2 = float(np.median(dists[dists > 0]) + 1e-10)

    K_XX = np.exp(-cdist(X_ref, X_ref, 'sqeuclidean') / sigma2)
    K_YY = np.exp(-cdist(X_test, X_test, 'sqeuclidean') / sigma2)
    K_XY = np.exp(-cdist(X_ref, X_test, 'sqeuclidean') / sigma2)
    return float(max(0, K_XX.mean() + K_YY.mean() - 2*K_XY.mean()))


def tda_h0_entropy(X):
    result = ripser.ripser(X, maxdim=1, metric='euclidean')
    h0 = result['dgms'][0]
    h0_finite = h0[h0[:, 1] != np.inf]
    if len(h0_finite) == 0:
        return 0.0, result['dgms']
    lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0, result['dgms']
    total = lifetimes.sum()
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-10))), result['dgms']


def tda_h1_entropy(diagrams):
    """H1 persistent entropy from pre-computed diagrams."""
    if len(diagrams) < 2:
        return 0.0
    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf]
    if len(h1_finite) == 0:
        return 0.0
    lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def wasserstein_score(dgm1, dgm2, dim=1):
    """Wasserstein distance between H1 diagrams."""
    d1 = dgm1[dim][dgm1[dim][:, 1] != np.inf] if len(dgm1) > dim else np.array([[0, 0]])
    d2 = dgm2[dim][dgm2[dim][:, 1] != np.inf] if len(dgm2) > dim else np.array([[0, 0]])

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        d1 = np.array([[0, 0]])
    if len(d2) == 0:
        d2 = np.array([[0, 0]])

    try:
        return float(persim.wasserstein(d1, d2))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic_comparison():
    """
    Compare drift detectors on 4 synthetic scenarios:
    1. No drift (blob → blob)
    2. Centroid drift (blob → shifted blob)
    3. Topology drift: loop appears (blob → annulus, same centroid)
    4. Topology drift: split (blob → two-clusters, same centroid)
    """
    scenarios = {
        'no_drift': {
            'ref_fn': lambda seed: make_blob(100, dim=10, center=None, seed=seed),
            'test_fn': lambda seed: make_blob(100, dim=10, center=None, seed=seed+1000),
        },
        'centroid_drift': {
            'ref_fn': lambda seed: make_blob(100, dim=10, center=None, seed=seed),
            'test_fn': lambda seed: make_blob(100, dim=10, center=np.array([3.]+[0.]*9), seed=seed+1000),
        },
        'loop_topology': {
            'ref_fn': lambda seed: make_blob(100, dim=10, seed=seed),
            'test_fn': lambda seed: make_annulus(100, dim=10, seed=seed+1000),
        },
        'split_topology': {
            'ref_fn': lambda seed: make_blob(100, dim=10, seed=seed),
            'test_fn': lambda seed: make_two_clusters(100, dim=10, separation=3.0, seed=seed+1000),
        },
    }

    seeds = [42, 123, 456, 789, 1000]
    results = []

    for sc_name, sc in scenarios.items():
        logger.info(f"  Scenario: {sc_name}")
        for seed in seeds:
            X_ref = sc['ref_fn'](seed)
            X_test = sc['test_fn'](seed)

            # Statistical methods
            t0 = time.time()
            c_score = centroid_score(X_ref, X_test)
            t_centroid = time.time() - t0

            t0 = time.time()
            cov_score = covariance_score(X_ref, X_test)
            t_cov = time.time() - t0

            t0 = time.time()
            m_score = mmd_rbf(X_ref, X_test)
            t_mmd = time.time() - t0

            # TDA methods
            t0 = time.time()
            h0_ref, dgms_ref = tda_h0_entropy(X_ref)
            h0_test, dgms_test = tda_h0_entropy(X_test)
            t_tda = time.time() - t0

            h0_score = abs(h0_ref - h0_test)
            h1_ref_score = tda_h1_entropy(dgms_ref)
            h1_test_score = tda_h1_entropy(dgms_test)
            h1_score = abs(h1_ref_score - h1_test_score)

            t0 = time.time()
            w1_score = wasserstein_score(dgms_ref, dgms_test, dim=1)  # H1 Wasserstein
            t_wass = time.time() - t0

            results.append({
                'scenario': sc_name,
                'seed': seed,
                'centroid': c_score,
                'covariance': cov_score,
                'mmd': m_score,
                'tda_h0_entropy': h0_score,
                'tda_h1_entropy': h1_score,
                'tda_h1_wasserstein': w1_score,
                'is_drift': 0 if sc_name == 'no_drift' else 1,
                'rt_centroid': t_centroid,
                'rt_cov': t_cov,
                'rt_mmd': t_mmd,
                'rt_tda': t_tda,
                'rt_wass': t_wass,
            })

    df = pd.DataFrame(results)

    # Print comparison
    logger.info("\nSynthetic experiment results (mean over seeds):")
    methods = ['centroid', 'covariance', 'mmd', 'tda_h0_entropy', 'tda_h1_entropy', 'tda_h1_wasserstein']
    pivot = df.groupby('scenario')[methods].mean()
    logger.info("\n" + pivot.to_string())

    # Compute separability ratio (drift / no_drift score) for each method
    logger.info("\nSeparability ratio (drift_score / no-drift_score) per scenario and method:")
    nd = df[df['scenario'] == 'no_drift'][methods].mean()
    for sc in ['centroid_drift', 'loop_topology', 'split_topology']:
        drift = df[df['scenario'] == sc][methods].mean()
        ratio = drift / (nd + 1e-10)
        logger.info(f"  {sc}: " + ", ".join([f"{m}={ratio[m]:.1f}x" for m in methods]))

    # Determine winner per scenario
    logger.info("\nBest method per scenario:")
    for sc in ['centroid_drift', 'loop_topology', 'split_topology']:
        drift = df[df['scenario'] == sc][methods].mean()
        nd_vals = nd.copy()
        ratios = drift / (nd_vals + 1e-10)
        best = ratios.idxmax()
        logger.info(f"  {sc}: BEST = {best} (ratio = {ratios[best]:.1f}x)")

    # Save results
    df.to_csv(RESULTS_DIR / 'synthetic_results.csv', index=False)
    logger.info(f"Saved synthetic results to {RESULTS_DIR}/synthetic_results.csv")

    return df, pivot


def make_synthetic_visualizations(df: pd.DataFrame, pivot: pd.DataFrame):
    """Create visualization for synthetic experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Point cloud comparison
    ax = axes[0, 0]
    rng = np.random.default_rng(42)
    blob_2d = make_blob(150, dim=2, seed=42)
    annulus_2d = make_annulus(150, dim=2, seed=42)
    two_clust_2d = make_two_clusters(150, dim=2, separation=3.0, seed=42)
    shifted_2d = make_blob(150, dim=2, center=np.array([3., 0.]), seed=42)

    ax.scatter(blob_2d[:, 0], blob_2d[:, 1], alpha=0.5, s=20, label='Reference (Blob)', color='blue')
    ax.scatter(annulus_2d[:, 0], annulus_2d[:, 1], alpha=0.5, s=20, marker='x', label='Loop drift (Annulus)', color='red')
    ax.scatter(two_clust_2d[:, 0], two_clust_2d[:, 1], alpha=0.5, s=20, marker='^', label='Split drift (2-cluster)', color='green')
    ax.scatter(shifted_2d[:, 0], shifted_2d[:, 1], alpha=0.5, s=20, marker='s', label='Centroid drift (shifted)', color='orange')
    ax.set_title('Synthetic Drift Scenarios (2D projection)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Heatmap of scores per scenario and method
    ax = axes[0, 1]
    scenarios = ['no_drift', 'centroid_drift', 'loop_topology', 'split_topology']
    methods = ['centroid', 'covariance', 'mmd', 'tda_h0_entropy', 'tda_h1_entropy', 'tda_h1_wasserstein']
    method_labels = ['Centroid', 'Covariance', 'MMD', 'TDA H0 Entropy', 'TDA H1 Entropy', 'TDA H1 Wass.']
    sc_labels = ['No Drift', 'Centroid Drift', 'Loop (Annulus)', 'Split (2-cluster)']

    mean_scores = df.groupby('scenario')[methods].mean()
    mean_scores = mean_scores.reindex(scenarios)

    # Normalize each method column
    norm_scores = mean_scores.copy()
    for m in methods:
        rng_val = mean_scores[m].max() - mean_scores[m].min()
        if rng_val > 0:
            norm_scores[m] = (mean_scores[m] - mean_scores[m].min()) / rng_val

    sns.heatmap(norm_scores.T, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                xticklabels=sc_labels, yticklabels=method_labels)
    ax.set_title('Normalized Drift Scores\n(0=min, 1=max per method)', fontsize=11)
    ax.set_xticklabels(sc_labels, rotation=15)

    # Plot 3: Separability ratio by method type
    ax = axes[1, 0]
    nd = df[df['scenario'] == 'no_drift'][methods].mean()
    drift_scenarios = ['centroid_drift', 'loop_topology', 'split_topology']
    sc_labels2 = ['Centroid Drift', 'Loop (Annulus)', 'Split (2-cluster)']

    x = np.arange(len(drift_scenarios))
    width = 0.12
    colors = {'centroid': '#FF5722', 'covariance': '#FF9800', 'mmd': '#FFC107',
              'tda_h0_entropy': '#2196F3', 'tda_h1_entropy': '#9C27B0', 'tda_h1_wasserstein': '#3F51B5'}

    for i, (m, ml) in enumerate(zip(methods, method_labels)):
        ratios = []
        for sc in drift_scenarios:
            dv = df[df['scenario'] == sc][m].mean()
            nv = nd[m]
            ratios.append(dv / (nv + 1e-10))
        ax.bar(x + i * width - 2.5 * width, ratios, width, label=ml, color=colors[m], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels2)
    ax.set_ylabel('Drift Score Ratio (drift / no-drift)')
    ax.set_title('Sensitivity: Drift Score Ratio\n(Higher = better detection)', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yscale('log')

    # Plot 4: Bar chart - TDA advantage on topology scenarios
    ax = axes[1, 1]
    tda_methods = ['tda_h0_entropy', 'tda_h1_entropy', 'tda_h1_wasserstein']
    stat_methods = ['centroid', 'covariance', 'mmd']

    loop_data = df[df['scenario'] == 'loop_topology']
    split_data = df[df['scenario'] == 'split_topology']

    loop_tda = loop_data[tda_methods].values.flatten()
    loop_stat = loop_data[stat_methods].values.flatten()
    split_tda = split_data[tda_methods].values.flatten()
    split_stat = split_data[stat_methods].values.flatten()

    # Normalize by no-drift baseline
    nd_tda = df[df['scenario'] == 'no_drift'][tda_methods].values.flatten()
    nd_stat = df[df['scenario'] == 'no_drift'][stat_methods].values.flatten()

    loop_tda_ratio = loop_tda.mean() / (nd_tda.mean() + 1e-10)
    loop_stat_ratio = loop_stat.mean() / (nd_stat.mean() + 1e-10)
    split_tda_ratio = split_tda.mean() / (nd_tda.mean() + 1e-10)
    split_stat_ratio = split_stat.mean() / (nd_stat.mean() + 1e-10)

    categories = ['Loop (Annulus)', 'Split (2-cluster)']
    tda_vals = [loop_tda_ratio, split_tda_ratio]
    stat_vals = [loop_stat_ratio, split_stat_ratio]

    x2 = np.arange(len(categories))
    bars1 = ax.bar(x2 - 0.2, tda_vals, 0.35, label='TDA methods (avg)', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x2 + 0.2, stat_vals, 0.35, label='Statistical methods (avg)', color='#FF5722', alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Mean Separability Ratio (drift/no-drift)')
    ax.set_title('TDA vs Statistical on Topology-Only Drift\n(Higher = more sensitive)', fontsize=11)
    ax.legend()
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No discrimination')

    # Annotate with "TDA wins" / "STAT wins"
    for i, (tv, sv) in enumerate(zip(tda_vals, stat_vals)):
        winner = 'TDA wins' if tv > sv else 'STAT wins'
        color = '#2196F3' if tv > sv else '#FF5722'
        ax.text(i, max(tv, sv) + 0.5, winner, ha='center', fontsize=9,
                color=color, fontweight='bold')

    plt.suptitle('Synthetic Experiment: Topology-Only Drift Detection\n'
                 '(Centroid-preserving drift scenarios)', fontsize=13)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'fig7_synthetic_topology_experiment.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


if __name__ == '__main__':
    logger.info("Running synthetic topology experiment...")
    df, pivot = run_synthetic_comparison()
    make_synthetic_visualizations(df, pivot)
    logger.info("Done!")

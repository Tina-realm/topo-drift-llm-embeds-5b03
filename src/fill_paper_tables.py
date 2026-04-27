"""
Post-processing script: Fill paper tables from experiment v3 results.
Run after experiment_v3.py completes.

Usage:
    python src/fill_paper_tables.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE = Path('/workspaces/topo-drift-llm-embeds-5b03')
RESULTS_DIR = WORKSPACE / 'results_v3'

def fmt(mean, std):
    """Format mean ± std."""
    if pd.isna(mean):
        return '---'
    if pd.isna(std) or std == 0:
        return f'{mean:.3f}'
    return f'{mean:.3f} ± {std:.3f}'

def fmt_delay(mean, std):
    if pd.isna(mean):
        return '---'
    return f'{mean:.1f} ± {std:.1f}' if not pd.isna(std) else f'{mean:.1f}'

def main():
    # Load results
    results_df = pd.read_csv(RESULTS_DIR / 'all_results.csv')
    print(f"Loaded {len(results_df)} results rows")

    # Key method labels for the paper
    method_labels = {
        'centroid': 'Centroid Shift',
        'covariance': 'Covariance Shift',
        'mmd': 'MMD (RBF)',
        'knn': 'kNN Distance',
        'energy': 'Energy Distance',
        'classifier': 'Classifier 2-Sample',
        'tda_wass_h1': 'Wasserstein H1',
        'tda_wass_h0': 'Wasserstein H0',
        'tda_sliced_wass_h1': 'Sliced Wass. H1',
        'tda_sliced_wass_h0': 'Sliced Wass. H0',
        'tda_pe_h0': 'PE-H0',
        'tda_pe_h1': 'PE-H1',
        'tda_phd': 'PHD',
        'tda_bottleneck_h0': 'Bottleneck H0',
        'tda_h1_total_persistence': 'H1 Total Persist.',
        'tda_landscape_h0_L1_norm': 'Landscape H0',
        'tda_landscape_h1_L1_norm': 'Landscape H1',
    }

    stat_methods = ['centroid', 'covariance', 'mmd', 'knn', 'energy', 'classifier']
    tda_key = ['tda_wass_h1', 'tda_sliced_wass_h1', 'tda_landscape_h1_L1_norm',
               'tda_pe_h1', 'tda_phd', 'tda_wass_h0']

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 1: Main results (AG News, MiniLM)
    # ═══════════════════════════════════════════════════════════════════════════
    ag_minilm = results_df[(results_df['dataset'] == 'ag_news') &
                           (results_df['model'] == 'minilm') &
                           (results_df['window_size'] == 200)]

    drift_only = ag_minilm[(ag_minilm['drift_type'] != 'no_drift') & (ag_minilm['auc'].notna())]
    nodrift = ag_minilm[ag_minilm['drift_type'] == 'no_drift']

    print("\n" + "="*80)
    print("TABLE 1: Main Results (AG News, MiniLM)")
    print("="*80)
    print(f"{'Method':<25} {'Type':<12} {'Mean AUC':<15} {'Mean Delay':<15} {'FPR':<10} {'Runtime/win':<12}")
    print("-"*89)

    for m in stat_methods + tda_key:
        mdf = drift_only[drift_only['method'] == m]
        ndf = nodrift[nodrift['method'] == m]
        label = method_labels.get(m, m)
        mtype = 'Statistical' if m in stat_methods else 'TDA'

        if len(mdf) > 0:
            mean_auc = mdf['auc'].mean()
            std_auc = mdf['auc'].std()
            mean_delay = mdf['detection_delay'].mean()
            std_delay = mdf['detection_delay'].std()
            mean_fpr = ndf['fpr'].mean() if len(ndf) > 0 else float('nan')
            mean_rt = mdf['runtime_per_window'].mean()
            print(f"{label:<25} {mtype:<12} {fmt(mean_auc, std_auc):<15} {fmt_delay(mean_delay, std_delay):<15} {mean_fpr:.3f}     {mean_rt*1000:.0f}ms")

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE: Centroid-preserving scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("TABLE: Centroid-Preserving Scenarios (AG News, MiniLM)")
    print("="*80)
    cp_scenarios = ['centroid_preserving', 'subtopic_reweight', 'style_perturbation']
    methods_to_show = ['centroid', 'covariance', 'mmd', 'energy', 'classifier',
                       'tda_wass_h1', 'tda_sliced_wass_h1', 'tda_landscape_h1_L1_norm',
                       'tda_pe_h1']

    header = f"{'Method':<25}" + "".join(f"{s:<20}" for s in cp_scenarios)
    print(header)
    print("-"*85)

    for m in methods_to_show:
        label = method_labels.get(m, m)
        vals = []
        for s in cp_scenarios:
            sdf = ag_minilm[(ag_minilm['drift_type'] == s) & (ag_minilm['method'] == m) & (ag_minilm['auc'].notna())]
            if len(sdf) > 0:
                vals.append(fmt(sdf['auc'].mean(), sdf['auc'].std()))
            else:
                vals.append('---')
        print(f"{label:<25}" + "".join(f"{v:<20}" for v in vals))

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE: Cross-dataset / cross-model
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("TABLE: Cross-Dataset / Cross-Model")
    print("="*80)

    for (ds, model), group in results_df[results_df['auc'].notna()].groupby(['dataset', 'model']):
        print(f"\n--- {ds} / {model} ---")
        summary = group.groupby('method').agg(
            mean_auc=('auc', 'mean'),
            std_auc=('auc', 'std'),
        ).sort_values('mean_auc', ascending=False)
        for m in stat_methods + tda_key:
            if m in summary.index:
                row = summary.loc[m]
                label = method_labels.get(m, m)
                print(f"  {label:<25} {fmt(row['mean_auc'], row['std_auc'])}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SYNTHETIC RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    synth_file = RESULTS_DIR / 'synthetic_results.csv'
    if synth_file.exists():
        synth_df = pd.read_csv(synth_file)
        print("\n" + "="*80)
        print("TABLE: Synthetic Topology Experiment (AUC)")
        print("="*80)

        drift_types = [d for d in synth_df['drift_type'].unique() if d != 'no_drift']
        header = f"{'Method':<25}" + "".join(f"{d:<20}" for d in drift_types)
        print(header)
        print("-"*105)

        for m in ['centroid', 'covariance', 'mmd', 'energy', 'classifier',
                   'tda_pe_h0', 'tda_pe_h1', 'tda_wass_h0', 'tda_wass_h1',
                   'tda_sliced_wass_h1', 'tda_landscape_h1_L1_norm']:
            label = method_labels.get(m, m)
            vals = []
            for dt in drift_types:
                sdf = synth_df[(synth_df['drift_type'] == dt) & (synth_df['method'] == m)]
                if len(sdf) > 0:
                    vals.append(fmt(sdf['auc'].mean(), sdf['auc'].std()))
                else:
                    vals.append('---')
            print(f"{label:<25}" + "".join(f"{v:<20}" for v in vals))

    # ═══════════════════════════════════════════════════════════════════════════
    # ABLATION RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    ablation_file = RESULTS_DIR / 'ablation_results.csv'
    if ablation_file.exists():
        abl_df = pd.read_csv(ablation_file)
        print("\n" + "="*80)
        print("ABLATION SUMMARIES")
        print("="*80)

        for abl_type in ['window_size', 'tda_subsample', 'pca_dim']:
            adf = abl_df[(abl_df['ablation'] == abl_type) & (abl_df['auc'].notna())]
            if len(adf) == 0:
                continue

            param_col = abl_type
            print(f"\n--- {abl_type} ---")
            for m in ['tda_wass_h1', 'tda_sliced_wass_h1', 'tda_pe_h1']:
                mdf = adf[adf['method'] == m]
                if len(mdf) == 0:
                    continue
                label = method_labels.get(m, m)
                group_means = mdf.groupby(param_col)['auc'].agg(['mean', 'std'])
                parts = [f"{idx}: {fmt(row['mean'], row['std'])}" for idx, row in group_means.iterrows()]
                print(f"  {label:<25} {' | '.join(parts)}")

    print("\n\nDone. Use these values to fill in paper_draft_v2/sections/results.tex")


if __name__ == '__main__':
    main()

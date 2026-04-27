"""
Generate filled paper tables from experiment v3 results.
Updates LaTeX files in paper_draft_v2/sections/ with actual numbers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re

WORKSPACE = Path('/workspaces/topo-drift-llm-embeds-5b03')
RESULTS_DIR = WORKSPACE / 'results_v3'
SECTIONS_DIR = WORKSPACE / 'paper_draft_v2' / 'sections'

def fmt(mean, std=None, decimals=3):
    if pd.isna(mean):
        return '---'
    if std is None or pd.isna(std) or std == 0:
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}'

def fmt_ms(val):
    if pd.isna(val):
        return '---'
    ms = val * 1000
    if ms < 1:
        return '$<$1'
    return f'{ms:.0f}'

def bold_best(vals, labels, higher_better=True):
    """Bold the best value in a list."""
    numeric = []
    for v in vals:
        try:
            numeric.append(float(v.split('$')[0].strip()))
        except:
            numeric.append(float('-inf') if higher_better else float('inf'))
    best_idx = np.argmax(numeric) if higher_better else np.argmin(numeric)
    result = list(vals)
    result[best_idx] = '\\textbf{' + result[best_idx] + '}'
    return result

# Method display names
METHOD_LABELS = {
    'centroid': 'Centroid Shift',
    'covariance': 'Covariance Shift',
    'mmd': 'MMD (RBF)',
    'knn': 'kNN Distance',
    'energy': 'Energy Distance',
    'classifier': 'Classifier 2-Sample',
    'tda_wass_h1': 'Wasserstein H1',
    'tda_wass_h0': 'Wasserstein H0',
    'tda_sliced_wass_h1': 'Sliced Wass.\\ H1',
    'tda_sliced_wass_h0': 'Sliced Wass.\\ H0',
    'tda_pe_h0': 'PE-H0',
    'tda_pe_h1': 'PE-H1',
    'tda_phd': 'PHD',
    'tda_bottleneck_h0': 'Bottleneck H0',
    'tda_h1_total_persistence': 'H1 Total Persist.',
    'tda_landscape_h0_L1_norm': 'Landscape H0',
    'tda_landscape_h1_L1_norm': 'Landscape H1',
}

STAT_METHODS = ['centroid', 'covariance', 'mmd', 'knn', 'energy', 'classifier']
TDA_KEY = ['tda_wass_h1', 'tda_sliced_wass_h1', 'tda_landscape_h1_L1_norm',
           'tda_pe_h1', 'tda_phd', 'tda_wass_h0']


def generate_table1(df):
    """Table 1: Main results (AG News, MiniLM)."""
    ag = df[(df['dataset'] == 'ag_news') & (df['model'] == 'minilm') & (df['window_size'] == 200)]
    drift_only = ag[(ag['drift_type'] != 'no_drift') & ag['auc'].notna()]
    nodrift = ag[ag['drift_type'] == 'no_drift']

    rows = []
    for m in STAT_METHODS + TDA_KEY:
        mdf = drift_only[drift_only['method'] == m]
        ndf = nodrift[nodrift['method'] == m]
        label = METHOD_LABELS.get(m, m)
        mtype = 'Statistical' if m in STAT_METHODS else 'TDA'

        if len(mdf) > 0:
            auc_str = fmt(mdf['auc'].mean(), mdf['auc'].std())
            delay_str = fmt(mdf['detection_delay'].mean(), mdf['detection_delay'].std(), 1)
            fpr_str = fmt(ndf['fpr'].mean(), decimals=3) if len(ndf) > 0 else '---'
            rt_str = fmt_ms(mdf['runtime_per_window'].mean()) + 'ms'
            rows.append(f"        {label} & {mtype} & {auc_str} & {delay_str} & {fpr_str} & {rt_str} \\\\")

    # Insert midrule between stat and TDA
    stat_count = len(STAT_METHODS)
    rows.insert(stat_count, '        \\midrule')
    return rows


def generate_table_cp(df):
    """Table 2: Centroid-preserving scenarios."""
    ag = df[(df['dataset'] == 'ag_news') & (df['model'] == 'minilm') & (df['window_size'] == 200)]
    cp_scenarios = ['centroid_preserving', 'subtopic_reweight', 'style_perturbation']
    methods = ['centroid', 'covariance', 'mmd', 'energy', 'classifier',
               'tda_wass_h1', 'tda_sliced_wass_h1', 'tda_landscape_h1_L1_norm',
               'tda_pe_h1']

    rows = []
    for i, m in enumerate(methods):
        label = METHOD_LABELS.get(m, m)
        vals = []
        for s in cp_scenarios:
            sdf = ag[(ag['drift_type'] == s) & (ag['method'] == m) & ag['auc'].notna()]
            if len(sdf) > 0:
                vals.append(fmt(sdf['auc'].mean(), sdf['auc'].std()))
            else:
                vals.append('---')
        row = f"        {label} & {' & '.join(vals)} \\\\"
        rows.append(row)
        if i == 4:  # After classifier (last stat method)
            rows.append('        \\midrule')
    return rows


def generate_table_ng(df):
    """Table 3: Cross-dataset/model results (20 Newsgroups)."""
    methods = ['centroid', 'mmd', 'classifier', 'tda_wass_h1', 'tda_sliced_wass_h1',
               'tda_landscape_h1_L1_norm']

    rows = []
    for i, m in enumerate(methods):
        label = METHOD_LABELS.get(m, m)
        vals = []
        for model_key in ['minilm', 'bert_base']:
            for scenario in ['newsgroup_close', 'newsgroup_distant']:
                sdf = df[(df['dataset'] == '20newsgroups') & (df['model'] == model_key) &
                         (df['drift_type'] == scenario) & (df['method'] == m) & df['auc'].notna()]
                if len(sdf) > 0:
                    vals.append(fmt(sdf['auc'].mean(), sdf['auc'].std()))
                else:
                    vals.append('---')
        row = f"        {label} & {' & '.join(vals)} \\\\"
        rows.append(row)
        if i == 2:  # After classifier
            rows.append('        \\midrule')
    return rows


def generate_table_synthetic(synth_df):
    """Table 4: Synthetic topology experiment."""
    methods = ['centroid', 'energy', 'classifier', 'mmd',
               'tda_pe_h1', 'tda_wass_h1', 'tda_sliced_wass_h1', 'tda_pe_h0']
    drift_types = ['centroid_shift', 'annulus', 'two_cluster', 'variance_change']

    rows = []
    for i, m in enumerate(methods):
        label = METHOD_LABELS.get(m, m)
        vals = []
        for dt in drift_types:
            sdf = synth_df[(synth_df['drift_type'] == dt) & (synth_df['method'] == m)]
            if len(sdf) > 0:
                vals.append(fmt(sdf['auc'].mean(), sdf['auc'].std()))
            else:
                vals.append('---')
        row = f"        {label} & {' & '.join(vals)} \\\\"
        rows.append(row)
        if i == 3:  # After MMD
            rows.append('        \\midrule')
    return rows


def generate_table_runtime(df):
    """Table 5: Runtime per window."""
    ag = df[(df['dataset'] == 'ag_news') & (df['model'] == 'minilm') &
            (df['window_size'] == 200) & df['auc'].notna()]

    # Runtime is per-family (all stat methods combined, all TDA combined)
    # Use median to avoid outlier inflation
    stat_methods_list = [m for m in ag['method'].unique() if not m.startswith('tda_')]
    tda_methods_list = [m for m in ag['method'].unique() if m.startswith('tda_')]

    rows = []
    if stat_methods_list:
        stat_df = ag[ag['method'] == stat_methods_list[0]]
        stat_rt = stat_df['runtime_per_window'].median()
        rows.append(f"        All statistical (6 methods) & {fmt_ms(stat_rt)}ms & Centroid + cov + MMD + kNN + energy + classifier \\\\")

    rows.append('        \\midrule')

    if tda_methods_list:
        tda_df = ag[ag['method'] == tda_methods_list[0]]
        tda_rt = tda_df['runtime_per_window'].median()
        rows.append(f"        All TDA (21 features) & {fmt_ms(tda_rt)}ms & PCA-50 + ripser + all features/distances \\\\")

    return rows


def replace_table_body(tex_content, label, new_rows):
    r"""Replace table body between first \midrule and \bottomrule for a table containing the given label."""
    lines = tex_content.split('\n')

    # Find which line has the label
    label_line = None
    for i, line in enumerate(lines):
        if f'\\label{{{label}}}' in line:
            label_line = i
            break

    if label_line is None:
        print(f"WARNING: Label {label} not found in tex")
        return tex_content

    # Scan backwards from label to find the enclosing \begin{table}
    table_start = None
    for i in range(label_line, -1, -1):
        if '\\begin{table}' in lines[i]:
            table_start = i
            break

    if table_start is None:
        print(f"WARNING: No \\begin{{table}} found before {label}")
        return tex_content

    # Find first \midrule after table_start (this marks end of header)
    first_midrule = None
    for i in range(table_start, label_line):
        if '\\midrule' in lines[i]:
            first_midrule = i
            break

    if first_midrule is None:
        print(f"WARNING: No \\midrule found in table {label}")
        return tex_content

    # Find \bottomrule after first_midrule
    bottomrule = None
    for i in range(first_midrule + 1, label_line + 5):
        if '\\bottomrule' in lines[i]:
            bottomrule = i
            break

    if bottomrule is None:
        print(f"WARNING: No \\bottomrule found in table {label}")
        return tex_content

    # Replace lines between first_midrule and bottomrule (exclusive both)
    result = lines[:first_midrule + 1] + new_rows + lines[bottomrule:]
    return '\n'.join(result)


def update_results_tex(df, synth_df):
    """Update results.tex with actual numbers."""
    results_path = SECTIONS_DIR / 'results.tex'
    tex = results_path.read_text()

    # Table 1: Main results
    t1_rows = generate_table1(df)
    tex = replace_table_body(tex, 'tab:main_results', t1_rows)

    # Table 2: Centroid-preserving
    t2_rows = generate_table_cp(df)
    tex = replace_table_body(tex, 'tab:centroid_preserving', t2_rows)

    # Table 3: 20 Newsgroups
    t3_rows = generate_table_ng(df)
    tex = replace_table_body(tex, 'tab:newsgroups', t3_rows)

    # Table 4: Synthetic
    if synth_df is not None and len(synth_df) > 0:
        t4_rows = generate_table_synthetic(synth_df)
        tex = replace_table_body(tex, 'tab:synthetic', t4_rows)

    # Table 5: Runtime
    t5_rows = generate_table_runtime(df)
    tex = replace_table_body(tex, 'tab:runtime', t5_rows)

    results_path.write_text(tex)
    print(f"Updated {results_path}")


def main():
    print("Loading results...")
    df = pd.read_csv(RESULTS_DIR / 'all_results.csv')
    print(f"  Main results: {len(df)} rows")
    print(f"  Coverage: {df.groupby(['dataset','model']).size().to_dict()}")

    synth_file = RESULTS_DIR / 'synthetic_results.csv'
    synth_df = pd.read_csv(synth_file) if synth_file.exists() else pd.DataFrame()
    print(f"  Synthetic: {len(synth_df)} rows")

    # Print tables to console first
    print("\n" + "="*80)
    print("TABLE 1: Main Results")
    print("="*80)
    for row in generate_table1(df):
        print(row)

    print("\n" + "="*80)
    print("TABLE 2: Centroid-Preserving")
    print("="*80)
    for row in generate_table_cp(df):
        print(row)

    print("\n" + "="*80)
    print("TABLE 3: 20 Newsgroups")
    print("="*80)
    for row in generate_table_ng(df):
        print(row)

    if len(synth_df) > 0:
        print("\n" + "="*80)
        print("TABLE 4: Synthetic")
        print("="*80)
        for row in generate_table_synthetic(synth_df):
            print(row)

    print("\n" + "="*80)
    print("TABLE 5: Runtime")
    print("="*80)
    for row in generate_table_runtime(df):
        print(row)

    # Update LaTeX
    print("\n" + "="*80)
    print("Updating LaTeX files...")
    print("="*80)
    update_results_tex(df, synth_df)

    print("\nDone!")


if __name__ == '__main__':
    main()

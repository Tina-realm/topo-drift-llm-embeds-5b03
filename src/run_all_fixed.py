"""
Re-run experiment with fixed centroid-preserving scenarios + remaining blocks.
Block 1 needs re-run for 3 centroid-preserving scenarios only.
Block 2 (20NG MiniLM) already done correctly (no CP scenarios).
Blocks 3-8 still needed.
"""
import sys, time, json, logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from experiment_v3 import (
    run_experiment, run_synthetic_experiment,
    load_embeddings, plot_main_results, plot_synthetic_results,
    plot_ablation_results, plot_persistence_examples,
    SEEDS, WINDOW_SIZES, TDA_SUBSAMPLE_SIZES, PCA_DIMS,
    DEFAULT_WINDOW, DEFAULT_PCA_DIM, TDA_SUBSAMPLE,
    RESULTS_DIR, FIGURES_DIR, logger
)

def main():
    t_start = time.time()
    logger.info("="*70)
    logger.info("FIXED Experiment v3 — re-running CP scenarios + remaining blocks")
    logger.info("="*70)

    # Load existing results
    existing_df = pd.read_csv(RESULTS_DIR / 'all_results.csv')
    logger.info(f"Loaded {len(existing_df)} existing results")

    # Remove old centroid-preserving results from AG News MiniLM (they're wrong)
    cp_scenarios = ['centroid_preserving', 'subtopic_reweight', 'style_perturbation']
    mask_bad = (
        (existing_df['dataset'] == 'ag_news') &
        (existing_df['model'] == 'minilm') &
        (existing_df['drift_type'].isin(cp_scenarios))
    )
    logger.info(f"Removing {mask_bad.sum()} bad CP results")
    existing_df = existing_df[~mask_bad]

    all_results = existing_df.to_dict('records')
    logger.info(f"Kept {len(all_results)} valid results")

    ablation_results = []
    SECONDARY_SEEDS = [42, 123, 456]

    def save_incremental(label):
        if all_results:
            pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'all_results.csv', index=False)
        if ablation_results:
            pd.DataFrame(ablation_results).to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
        logger.info(f"  [Save after {label}: {len(all_results)} main + {len(ablation_results)} ablation]")

    # Re-run centroid-preserving scenarios for AG News MiniLM
    logger.info("\n" + "="*70)
    logger.info("RE-RUNNING: AG News MiniLM centroid-preserving scenarios (FIXED)")
    logger.info("="*70)
    cp_results = run_experiment('ag_news', 'minilm', cp_scenarios)
    all_results.extend(cp_results)
    save_incremental('Fixed CP scenarios')

    # Block 2: Already done (20NG MiniLM)
    has_ng_minilm = existing_df[(existing_df['dataset'] == '20newsgroups') &
                                 (existing_df['model'] == 'minilm')].shape[0] >= 100
    if has_ng_minilm:
        logger.info("SKIP Block 2: 20NG MiniLM already done")
    else:
        ng_scenarios = ['no_drift', 'newsgroup_close', 'newsgroup_distant']
        ng_results = run_experiment('20newsgroups', 'minilm', ng_scenarios)
        all_results.extend(ng_results)
        save_incremental('Exp2: 20NG MiniLM')

    # Block 3: 20 Newsgroups + MPNet-base
    ng_scenarios = ['no_drift', 'newsgroup_close', 'newsgroup_distant']
    has_ng_bert = existing_df[(existing_df['dataset'] == '20newsgroups') &
                               (existing_df['model'] == 'bert_base')].shape[0] >= 100
    if has_ng_bert:
        logger.info("SKIP Block 3: 20NG MPNet already done")
    else:
        ng_bert_results = run_experiment('20newsgroups', 'bert_base', ng_scenarios,
                                         seeds=SECONDARY_SEEDS)
        all_results.extend(ng_bert_results)
        save_incremental('Exp3: 20NG MPNet')

    # Block 4: AG News + MPNet-base (key scenarios including CP)
    has_ag_bert = existing_df[(existing_df['dataset'] == 'ag_news') &
                               (existing_df['model'] == 'bert_base')].shape[0] >= 100
    if has_ag_bert:
        logger.info("SKIP Block 4: AG News MPNet already done")
    else:
        ag_bert_scenarios = ['no_drift', 'abrupt_topic', 'centroid_preserving', 'subtopic_reweight']
        ag_bert_results = run_experiment('ag_news', 'bert_base', ag_bert_scenarios,
                                         seeds=SECONDARY_SEEDS)
        all_results.extend(ag_bert_results)
        save_incremental('Exp4: AG News MPNet')

    # Blocks 5-7: Ablations
    ablation_scenarios = ['no_drift', 'abrupt_topic', 'centroid_preserving']

    logger.info("\n" + "="*70)
    logger.info("ABLATION: Window Size")
    logger.info("="*70)
    for ws in WINDOW_SIZES:
        ws_res = run_experiment('ag_news', 'minilm', ablation_scenarios,
                                window_size=ws, seeds=SECONDARY_SEEDS)
        for r in ws_res:
            r['ablation'] = 'window_size'
        ablation_results.extend(ws_res)
    save_incremental('Ablation: Window Size')

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

    # Block 8: Synthetic
    synth_df = run_synthetic_experiment()
    synth_df.to_csv(RESULTS_DIR / 'synthetic_results.csv', index=False)
    save_incremental('Synthetic')

    # Final saves
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'all_results.csv', index=False)
    results_df.to_json(RESULTS_DIR / 'all_results.json', orient='records', indent=2)

    ablation_df = pd.DataFrame(ablation_results)
    if len(ablation_results) > 0:
        ablation_df.to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)

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

    # Summary
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

    # Visualizations
    logger.info("\nGenerating visualizations...")
    plot_main_results(results_df, FIGURES_DIR)
    if len(synth_df) > 0:
        plot_synthetic_results(synth_df, FIGURES_DIR)
    if len(ablation_df) > 0:
        plot_ablation_results(ablation_df, FIGURES_DIR)

    embs = load_embeddings('ag_news', 'minilm')
    plot_persistence_examples(embs, FIGURES_DIR)

    t_total = time.time() - t_start
    logger.info(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("DONE")


if __name__ == '__main__':
    main()

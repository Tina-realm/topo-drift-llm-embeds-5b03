"""
Resume experiment_v3: skip already-completed blocks, run what's missing.
Checks existing results to avoid duplicates.
"""
import sys, time, json, logging
from pathlib import Path
from datetime import datetime
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

def has_results(df, dataset, model, min_rows=100):
    """Check if a dataset/model combination already has results."""
    mask = (df['dataset'] == dataset) & (df['model'] == model)
    return mask.sum() >= min_rows

def main():
    t_start = time.time()
    logger.info("="*70)
    logger.info("RESUMING Experiment v3 — skipping completed blocks")
    logger.info("="*70)

    existing_df = pd.read_csv(RESULTS_DIR / 'all_results.csv')
    all_results = existing_df.to_dict('records')
    logger.info(f"Loaded {len(all_results)} existing results")
    logger.info(f"Existing coverage: {existing_df.groupby(['dataset','model']).size().to_dict()}")

    ablation_results = []
    ablation_file = RESULTS_DIR / 'ablation_results.csv'
    if ablation_file.exists():
        abl_df = pd.read_csv(ablation_file)
        ablation_results = abl_df.to_dict('records')
        logger.info(f"Loaded {len(ablation_results)} existing ablation results")

    SECONDARY_SEEDS = [42, 123, 456]

    def save_incremental(label):
        if all_results:
            pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'all_results.csv', index=False)
        if ablation_results:
            pd.DataFrame(ablation_results).to_csv(RESULTS_DIR / 'ablation_results.csv', index=False)
        logger.info(f"  [Save after {label}: {len(all_results)} main + {len(ablation_results)} ablation]")

    ng_scenarios = ['no_drift', 'newsgroup_close', 'newsgroup_distant']

    # Block 2: 20 Newsgroups + MiniLM
    if not has_results(existing_df, '20newsgroups', 'minilm'):
        ng_results = run_experiment('20newsgroups', 'minilm', ng_scenarios)
        all_results.extend(ng_results)
        save_incremental('Exp2: 20NG MiniLM')
    else:
        logger.info("SKIP Block 2: 20NG MiniLM already done")

    # Block 3: 20 Newsgroups + MPNet-base
    if not has_results(existing_df, '20newsgroups', 'bert_base'):
        ng_bert_results = run_experiment('20newsgroups', 'bert_base', ng_scenarios,
                                         seeds=SECONDARY_SEEDS)
        all_results.extend(ng_bert_results)
        save_incremental('Exp3: 20NG MPNet')
    else:
        logger.info("SKIP Block 3: 20NG MPNet already done")

    # Block 4: AG News + MPNet-base
    if not has_results(existing_df, 'ag_news', 'bert_base'):
        ag_bert_scenarios = ['no_drift', 'abrupt_topic', 'centroid_preserving', 'subtopic_reweight']
        ag_bert_results = run_experiment('ag_news', 'bert_base', ag_bert_scenarios,
                                         seeds=SECONDARY_SEEDS)
        all_results.extend(ag_bert_results)
        save_incremental('Exp4: AG News MPNet')
    else:
        logger.info("SKIP Block 4: AG News MPNet already done")

    # Blocks 5-7: Ablations (only if not already done)
    if len(ablation_results) == 0:
        ablation_scenarios = ['no_drift', 'abrupt_topic', 'centroid_preserving']

        # Block 5: Window size
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

        # Block 6: TDA subsample size
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

        # Block 7: PCA dimensionality
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
    else:
        logger.info("SKIP Blocks 5-7: Ablation results already exist")

    # Block 8: Synthetic
    synth_file = RESULTS_DIR / 'synthetic_results.csv'
    if not synth_file.exists():
        synth_df = run_synthetic_experiment()
        synth_df.to_csv(synth_file, index=False)
        save_incremental('Synthetic')
    else:
        synth_df = pd.read_csv(synth_file)
        logger.info("SKIP Block 8: Synthetic results already exist")

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
    logger.info(f"\nTotal resume time: {t_total:.0f}s ({t_total/60:.1f}min)")
    logger.info("DONE")


if __name__ == '__main__':
    main()

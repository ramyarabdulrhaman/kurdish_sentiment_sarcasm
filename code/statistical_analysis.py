#!/usr/bin/env python3
"""
Publication Table and Statistical Analysis Generator
Reads strictly from logged experimental outputs:
  - results/step2_rigorous_results.json
  - results/predictions/*.json
  - results/human_evaluation_study_results.json
  - results/codn_intrinsic_evaluation.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TABLES_DIR = os.path.join(BASE_DIR, 'tables')
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def paired_bootstrap_test(y_true, preds_a, preds_b, n_bootstraps=10000, seed=42):
    """
    Performs two-sided paired bootstrap hypothesis test between model A and model B.
    """
    if np.array_equal(preds_a, preds_b):
        return 1.0, 0.0 # Identical predictions -> p = 1.0, diff = 0.0

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    
    acc_a = (preds_a == y_true).astype(float)
    acc_b = (preds_b == y_true).astype(float)
    obs_diff = np.mean(acc_a) - np.mean(acc_b)

    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    sub_a = acc_a[boot_indices]
    sub_b = acc_b[boot_indices]
    diffs = np.mean(sub_a, axis=1) - np.mean(sub_b, axis=1)

    # Two-sided p-value against null hypothesis (diff = 0)
    p_val = np.mean(np.abs(diffs - obs_diff) >= np.abs(obs_diff))
    return float(p_val), float(obs_diff * 100)

def main():
    print("=" * 80)
    print("GENERATING PUBLICATION TABLES FROM LOGGED STEP 2 RUNS")
    print("=" * 80)

    with open(os.path.join(RESULTS_DIR, 'step2_rigorous_results.json'), 'r') as f:
        exp_data = json.load(f)

    test_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'test.csv'))
    y_test_sent = test_df['label'].values
    y_test_sarc = test_df['is_sarcastic'].values

    # Load predictions
    preds = {}
    for pf_name in os.listdir(PRED_DIR):
        if pf_name.endswith('_predictions.json'):
            m_key = pf_name.replace('_predictions.json', '')
            with open(os.path.join(PRED_DIR, pf_name), 'r') as pf:
                preds[m_key] = json.load(pf)

    # -------------------------------------------------------------------------
    # TABLE 1: Comprehensive Comparative Performance
    # -------------------------------------------------------------------------
    print("Generating Table 1: Comparative Performance across all Models...")
    t1_rows = []
    model_order = [
        # Trivial Baselines
        'Majority_Classifier',
        'Always_Sarcastic_Baseline',
        # Surface Baselines
        'TFIDF_LinearSVC_Raw',
        'TFIDF_LinearSVC_CODN',
        'FastText_LogReg_Raw',
        'FastText_LogReg_CODN',
        'Translate_then_Classify',
        'Few_Shot_LLM_Llama3',
        # Pretrained Contextual Baselines
        'mBERT_Raw',
        'mBERT_CODN',
        'XLM_RoBERTa_Raw',
        'XLM_RoBERTa_CODN',
        'XLM_RoBERTa_large',
        'LaBSE_CODN',
        'KuBERT_Raw',
        'KuBERT_CODN',
        # Equal-Capacity DGSA Controls
        'Control_KuBERT_ExtraAttention_NoGating',
        'Control_KuBERT_DGSA_Gated',
        'Control_KuBERT_Dialect_Token_Prefix',
        'Control_KuBERT_Auxiliary_Dialect_Loss',
        # Incongruity Regularization Sweep
        'Control_KuBERT_Incongruity_beta0',
        'Control_KuBERT_Incongruity_beta0_05',
        'Control_KuBERT_Incongruity_beta0_1',
        'Control_KuBERT_Incongruity_beta0_2',
        'Control_KuBERT_Incongruity_beta0_5',
        'Control_KuBERT_Incongruity_beta1_0',
        'Control_KuBERT_ClassWeighted_CE',
        'Control_KuBERT_Oversampling',
        # Proposed
        'KurdDialect_Transformer'
    ]

    for m_name in model_order:
        if m_name not in exp_data['models']:
            continue
        m_info = exp_data['models'][m_name]
        acc_ci = m_info.get('accuracy_item_ci', [0.0, 0.0])
        f1_ci = m_info.get('macro_f1_item_ci', [0.0, 0.0])
        
        acc_str = f"{m_info['accuracy_mean']:.2f} ± {m_info['accuracy_std']:.2f} [{acc_ci[0]:.2f}, {acc_ci[1]:.2f}]"
        f1_str = f"{m_info['macro_f1_mean']:.2f} ± {m_info['macro_f1_std']:.2f} [{f1_ci[0]:.2f}, {f1_ci[1]:.2f}]"
        
        if m_info['has_sarcasm_head'] and m_info.get('sarcasm_f1_mean') != 'N/A':
            s_ci = m_info.get('sarcasm_f1_item_ci', [0.0, 0.0])
            sarc_str = f"{float(m_info['sarcasm_f1_mean']):.2f} ± {float(m_info['sarcasm_f1_std']):.2f} [{s_ci[0]:.2f}, {s_ci[1]:.2f}]"
            prauc_str = f"{float(m_info['sarcasm_prauc_mean']):.2f} ± {float(m_info['sarcasm_prauc_std']):.2f}"
        else:
            sarc_str = "N/A"
            prauc_str = "N/A"
            
        pol_inv_str = f"{m_info.get('polarity_inversion_rate_mean', 0.0):.2f}%"

        t1_rows.append({
            'Model Architecture': m_name.replace('_', ' '),
            'Sentiment Accuracy (%) [95% CI]': acc_str,
            'Sentiment Macro-F1 (%) [95% CI]': f1_str,
            'Sarcasm F1 (%) [95% CI]': sarc_str,
            'Sarcasm PR-AUC (%)': prauc_str,
            'Polarity Inversion Rate (%)': pol_inv_str
        })

    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(os.path.join(TABLES_DIR, 'table_1_comparative_performance.csv'), index=False)
    print("  -> Saved tables/table_1_comparative_performance.csv")

    # -------------------------------------------------------------------------
    # TABLE 2: Statistical Hypothesis Testing (Paired Bootstrap with Holm Correction)
    # -------------------------------------------------------------------------
    print("Generating Table 2: Paired Bootstrap Hypothesis Testing...")
    comparisons = [
        ('Control_KuBERT_DGSA_Gated', 'Control_KuBERT_ExtraAttention_NoGating', 'Incremental Value of Gating over Capacity Control'),
        ('KurdDialect_Transformer', 'Control_KuBERT_ExtraAttention_NoGating', 'Full Proposed Model vs Capacity Control'),
        ('KurdDialect_Transformer', 'Control_KuBERT_DGSA_Gated', 'Contribution of Incongruity Regularization'),
        ('Control_KuBERT_DGSA_Gated', 'Control_KuBERT_Dialect_Token_Prefix', 'Token-Wise DGSA vs Dialect Token Prefix'),
        ('Control_KuBERT_DGSA_Gated', 'Control_KuBERT_Auxiliary_Dialect_Loss', 'Token-Wise DGSA vs Auxiliary Dialect Loss'),
        ('KuBERT_CODN', 'KuBERT_Raw', 'CODN Normalization Gain on KuBERT'),
        ('mBERT_CODN', 'mBERT_Raw', 'CODN Normalization Gain on mBERT'),
        ('TFIDF_LinearSVC_CODN', 'TFIDF_LinearSVC_Raw', 'CODN Normalization Effect on TF-IDF'),
        ('KurdDialect_Transformer', 'TFIDF_LinearSVC_Raw', 'Proposed Architecture vs Surface TF-IDF Baseline'),
        ('KurdDialect_Transformer', 'Translate_then_Classify', 'Proposed Architecture vs Translate-then-Classify Baseline'),
        ('KurdDialect_Transformer', 'Few_Shot_LLM_Llama3', 'Proposed Architecture vs 5-Shot Llama-3-70B Prompting')
    ]

    t2_rows = []
    p_values = []
    
    for mod_a, mod_b, comp_name in comparisons:
        p_a = np.array(preds[mod_a]['sent_predictions'])
        p_b = np.array(preds[mod_b]['sent_predictions'])
        
        # Paired bootstrap
        p_val, diff = paired_bootstrap_test(y_test_sent, p_a, p_b, n_bootstraps=10000)
        p_values.append(p_val)
        
        # McNemar test
        tb = pd.crosstab(p_a == y_test_sent, p_b == y_test_sent)
        b = tb.iloc[0, 1] if (0 in tb.index and 1 in tb.columns) else 0
        c = tb.iloc[1, 0] if (1 in tb.index and 0 in tb.columns) else 0
        mcnemar_stat = float((abs(b - c) - 1)**2 / (b + c)) if (b + c) > 0 else 0.0

        if np.array_equal(p_a, p_b):
            verdict = "No detectable difference"
        elif p_val < 0.05:
            verdict = "Statistically Significant" if diff > 0 else "Significantly Lower"
        else:
            verdict = "Not Significant"

        t2_rows.append({
            'Comparison': comp_name,
            'Model A': mod_a.replace('_', ' '),
            'Model B': mod_b.replace('_', ' '),
            'Observed Diff (%)': round(diff, 2),
            'Raw p-value': round(p_val, 4),
            'McNemar Chi2': round(mcnemar_stat, 2),
            'Verdict': verdict
        })

    # Holm-Bonferroni correction
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    holm_p = np.zeros(m)
    for rank, idx in enumerate(sorted_indices):
        holm_p[idx] = min(1.0, p_values[idx] * (m - rank))

    for idx, row in enumerate(t2_rows):
        row['Holm-Adjusted p-value'] = round(holm_p[idx], 4)

    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_csv(os.path.join(TABLES_DIR, 'table_2_statistical_hypothesis_testing.csv'), index=False)
    print("  -> Saved tables/table_2_statistical_hypothesis_testing.csv")

    # -------------------------------------------------------------------------
    # TABLE 3: Disentangled Ablation Analysis
    # -------------------------------------------------------------------------
    print("Generating Table 3: Disentangled Ablation Study...")
    ablation_models = [
        ('KuBERT_Raw', 'Base KuBERT (Raw Informal Text)', False, False, False),
        ('KuBERT_CODN', '+ CODN Normalization (Single-Task)', True, False, False),
        ('Control_KuBERT_Incongruity_beta0', '+ Multi-Task Sarcasm Head (beta=0)', True, False, False),
        ('Control_KuBERT_ExtraAttention_NoGating', '+ Equal-Capacity Attention Layer (No Gating)', True, False, False),
        ('Control_KuBERT_DGSA_Gated', '+ Token-Wise DGSA Attention', True, True, False),
        ('Control_KuBERT_Incongruity_beta0_2', '+ Incongruity Regularization (beta=0.2, No DGSA)', True, False, True),
        ('KurdDialect_Transformer', 'Full KurdDialect-Transformer (+ All Modules)', True, True, True)
    ]

    t3_rows = []
    base_acc = exp_data['models']['KuBERT_Raw']['accuracy_mean']
    for m_key, desc, has_codn, has_dgsa, has_incon in ablation_models:
        info = exp_data['models'][m_key]
        acc_m = info['accuracy_mean']
        acc_std = info['accuracy_std']
        acc_ci = info['accuracy_item_ci']
        diff_from_base = round(acc_m - base_acc, 2)
        
        t3_rows.append({
            'Configuration': desc,
            'CODN': 'Yes' if has_codn else 'No',
            'DGSA': 'Yes' if has_dgsa else 'No',
            'Incongruity Loss': 'Yes' if has_incon else 'No',
            'Sentiment Accuracy (%) [95% CI]': f"{acc_m:.2f} ± {acc_std:.2f} [{acc_ci[0]:.2f}, {acc_ci[1]:.2f}]",
            'Delta vs Base KuBERT (%)': f"{diff_from_base:+.2f}%",
            'Sentiment Macro-F1 (%)': f"{info['macro_f1_mean']:.2f} ± {info['macro_f1_std']:.2f}",
            'Sarcasm F1 (%)': f"{info['sarcasm_f1_mean']:.2f}" if info['has_sarcasm_head'] and info['sarcasm_f1_mean'] != 'N/A' else "N/A",
            'Polarity Inversion Rate (%)': f"{info.get('polarity_inversion_rate_mean', 0.0):.2f}%"
        })

    t3_df = pd.DataFrame(t3_rows)
    t3_df.to_csv(os.path.join(TABLES_DIR, 'table_3_ablation_study.csv'), index=False)
    print("  -> Saved tables/table_3_ablation_study.csv")

    # -------------------------------------------------------------------------
    # TABLE 4: Dialect Breakdown with k/n Notation
    # -------------------------------------------------------------------------
    print("Generating Table 4: Regional Dialect Breakdown (k/n reporting)...")
    pred_prop = np.array(preds['KurdDialect_Transformer']['sent_predictions'])
    dialects = test_df['dialect'].values
    
    t4_rows = []
    for dial in ['General', 'Hawleri', 'Sulemani']:
        mask = (dialects == dial)
        n_dial = int(mask.sum())
        if n_dial > 0:
            k_correct = int(np.sum(pred_prop[mask] == y_test_sent[mask]))
            sub_acc = float(k_correct / n_dial * 100)
            
            # Bootstrap CI over the subgroup test items (B=10,000 paired resamples)
            rng = np.random.default_rng(42)
            sub_preds = pred_prop[mask]
            sub_true = y_test_sent[mask]
            boot_idxs = rng.integers(0, n_dial, size=(10000, n_dial))
            boot_scores = np.mean(sub_preds[boot_idxs] == sub_true[boot_idxs], axis=1) * 100
            ci_low = float(np.percentile(boot_scores, 2.5))
            ci_high = float(np.percentile(boot_scores, 97.5))
            
            t4_rows.append({
                'Dialect Variety': dial,
                'Test Count (N)': n_dial,
                'Correct Count (k/n)': f"{k_correct}/{n_dial}",
                'Accuracy (%) [95% CI]': f"{sub_acc:.2f}% [{ci_low:.2f}, {ci_high:.2f}]",
                'Claim Status': 'Exploratory (n < 100; CI reported)' if n_dial < 100 else 'Primary Benchmark'
            })

    t4_df = pd.DataFrame(t4_rows)
    t4_df.to_csv(os.path.join(TABLES_DIR, 'table_4_dialect_breakdown.csv'), index=False)
    print("  -> Saved tables/table_4_dialect_breakdown.csv")

    # -------------------------------------------------------------------------
    # FIGURE 4: Sarcasm Confusion Matrix
    # -------------------------------------------------------------------------
    print("Generating Figure 4: Sarcasm Confusion Matrix...")
    from sklearn.metrics import confusion_matrix
    pred_sarc = np.array(preds['KurdDialect_Transformer']['sarc_predictions'])
    cm = confusion_matrix(y_test_sarc, pred_sarc)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Literal', 'Sarcastic'],
                yticklabels=['Literal', 'Sarcastic'])
    plt.title(f'KurdDialect-Transformer Sarcasm Confusion Matrix\n(Test N={len(y_test_sarc)}, Sarcasm Positives={int(y_test_sarc.sum())})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_sarcasm_confusion_matrix.png'), dpi=300)
    plt.close()
    print("  -> Saved figures/fig4_sarcasm_confusion_matrix.png")

    print("\nAll publication tables and figures regenerated successfully from logged runs!")

if __name__ == '__main__':
    main()

# KurdDialect-Transformer: A Dialect-Aware Transformer Framework and Empirical Benchmark for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media

Official open-source repository for the practical implementation, benchmark datasets, model architectures, and experimental outputs from the study:

> **Beyond Translation: A Dialect-Aware Transformer Framework and Empirical Benchmark for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media**

---

## 👥 Authors and Affiliations

- **Ramyar A. Teimoor** $^{1, 2}$  
  - $^{1}$ Department of Computer, College of Science, University of Sulaimani, Sulaimanyah, Kurdistan Region, Iraq  
  - $^{2}$ Information Technology Department, Kurdistan Technical Institute, Sulaymaniyah, Kurdistan Region, Iraq  
  - 📧 Email: [ramyar.teimoor@univsul.edu.iq](mailto:ramyar.teimoor@univsul.edu.iq)  
  - 🆔 ORCID: [0000-0002-7016-1833](https://orcid.org/0000-0002-7016-1833)

- **Rebwar M. Nabi** $^{3, 2, *}$ *(Corresponding Author)*  
  - $^{3}$ Information Technology Department, Technical College of Informatics, Sulaimani Polytechnic University, Sulaymaniyah, Kurdistan Region, Iraq  
  - $^{2}$ Information Technology Department, Kurdistan Technical Institute, Sulaymaniyah, Kurdistan Region, Iraq  
  - 📧 Email: [rebwar.nabi@spu.edu.iq](mailto:rebwar.nabi@spu.edu.iq)  
  - 🆔 ORCID: [0000-0003-2709-7941](http://orcid.org/0000-0003-2709-7941)

---

## 📊 Dataset Provenance, Attribution, and License

- **Primary Source:** The benchmark corpus builds upon Kurdish social media commentary originally curated by **Badawi, Kazemi, and Rezaie (2024)** and **Hameed, Ahmadi, and Daneshfar (2023)**:
  > S. Badawi, A. Kazemi, and V. Rezaie, *"KurdiSent: A Kurdish Sentiment Analysis Dataset,"* *Language Resources and Evaluation*, vol. 58, pp. 815–838, 2024. DOI: [10.1007/s10579-024-09727-x](https://doi.org/10.1007/s10579-024-09727-x).
  > K. A. Hameed, S. Ahmadi, and F. Daneshfar, *"Transfer Learning for Low-Resource Sentiment Analysis in Central Kurdish,"* *ACM Transactions on Asian and Low-Resource Language Information Processing*, vol. 22, no. 6, pp. 1–21, 2023. DOI: [10.1145/3597428](https://doi.org/10.1145/3597428).
- **Hugging Face Hub:** Hosted at [`asparius/Kurdish-Sentiment`](https://huggingface.co/datasets/asparius/Kurdish-Sentiment).
- **License:** Distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
- **Institutional Ethics:** Approved by University of Sulaimani Ethics Committee (Approval No. UoS-REC-2026-042).
- **Benchmark Partitioning:**
  - **Audited 3-Way Split (7,601 posts):** Deduplicated via exact matching and MinHash Locality-Sensitive Hashing ($J \ge 0.8$) into strict zero-leakage partitions:
    - `train.csv`: 5,320 posts (70.0%)
    - `val.csv`: 760 posts (10.0%)
    - `test.csv`: 1,521 posts (20.0%)

---

## 🔬 Core Methodological Innovations

1. **Canonical Orthographic Disambiguation Normalizer (CODN):**
   - Deterministic 6-stage rule-based normalizer for Kurdish Perso-Arabic text.
   - Disambiguates Arabic vs. Kurdish glyphs, preserves phonetic consonantal Heh (`ه`) vs. vowel Ae (`ە`), harmonizes morpheme boundaries with Zero-Width Non-Joiners (ZWNJ).
   - Achieves **98.00% (490/500)** intrinsic normalization accuracy on gold Kurdish words and reduces KuBERT subword fertility from **1.44 to 1.29** tokens/word with **0.00%** sequence truncation.

2. **Token-Wise Dialect-Gated Self-Attention (DGSA):**
   - Implements fine-grained token-level gating: $\tilde{K}_j = K_j + \sum_k g_{j,k} W_{\text{dial}}[k, :]$, where $g_j = \text{softmax}(W_d x_j + b_d)$.
   - Avoids sequence-level scalar gating cancellation under softmax normalization.

3. **Incongruity-Aware Multi-Task Classification Head:**
   - Jointly predicts binary sentiment polarity and figurative sarcasm status.
   - Regularized by a label-aware hinge penalty $\mathcal{L}_{\text{incong}} = \mathbb{I}(y^{\text{sarc}}=1 \land y^{\text{sent}}=0) \cdot \max(0, p_{\text{pos}} - p_{\text{neg}} + \gamma)$ with $\beta=0.2$.
   - KurdDialect-Transformer achieves competitive affective classification performance, attaining **92.77% ± 0.09%** Sentiment Classification Accuracy [95% CI: 91.45, 94.02], **92.77% ± 0.09%** Macro-F1, **28.78% ± 0.13%** Sarcasm F1 [95% CI: 19.63, 37.70], and suppressing sarcasm polarity inversion down to **21.43%** (vs 53.57% for TF-IDF and 46.43% for FastText).

---

## 📁 Repository Directory Structure

```text
kurdish_sentiment_sarcasm/
│
├── README.md                               # Comprehensive documentation & attribution
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusions (caches, tensors, drafts)
│
├── code/                                   # Practical section source code
│   ├── model_and_algorithms.py             # DGSA attention and multi-task architecture definitions
│   ├── phase1_method_fixes.py              # Audited TokenWiseDGSA and 5-stage CODN normalizer
│   ├── preprocess_data.py                  # Normalization, dialect markers, and praise idiom heuristics
│   ├── intrinsic_codn_evaluation.py        # Intrinsic accuracy evaluation on 500 gold words
│   ├── extract_genuine_backbone_embeddings.py # Feature extraction for KuBERT, mBERT, XLM-R, LaBSE
│   ├── run_rigorous_step2_experiments.py   # Multi-seed (5 seeds) 29-arm benchmark training pipeline
│   ├── statistical_analysis.py             # Paired bootstrap significance testing & McNemar tests
│   ├── prepare_human_evaluation_step4.py   # Stratified sampling and quadratic weighted Cohen's Kappa
│   ├── measure_real_latencies.py           # CPU inference latency benchmarking
│   ├── evaluate_official_test_and_dgsa.py  # Official split evaluation & empirical DGSA diagnostics
│   └── visualize_results.py                # Publication figure generation scripts
│
├── data/                                   # Datasets and benchmarks
│   ├── train.csv                           # Clean stratified training set (n = 5,320)
│   ├── val.csv                             # Clean stratified validation set (n = 760)
│   ├── test.csv                            # Clean held-out test benchmark (n = 1,521)
│   ├── official_test_deduplicated.csv      # Deduplicated official test partition (n = 843)
│   ├── human_eval_100_sampled_ratings.csv  # 100 blind test posts evaluated by native linguists
│   ├── kubert_fixed_vocab.txt              # Fixed KuBERT vocabulary mapping
│   ├── raw/                                # Original source data files
│   ├── checksums/                          # Cryptographic SHA-256 partition checksums
│   └── documentation/                      # Data dictionaries, provenance, and quality audit
│
├── results/                                # Logged experimental outputs and predictions
│   ├── step2_rigorous_results.json         # Complete 29-arm benchmark results over 5 seeds
│   ├── codn_intrinsic_evaluation.json      # Gold 500-word CODN accuracy log
│   ├── cpu_latencies_benchmarked.json      # Real CPU latency measurements
│   ├── dgsa_empirical_diagnostics.json     # DGSA parameter norms & attention weight change logs
│   ├── human_evaluation_study_results.json # Dual-linguist plausibility and Kappa agreement
│   ├── incongruity_loss_sweep.json         # Beta hyperparameter sweep (0.0 to 1.0)
│   ├── linguistic_error_analysis_20_examples.json # 20 qualitative test case breakdowns
│   ├── official_test_eval_results.json     # Performance on official split benchmark
│   └── predictions/                        # Exact test set predictions for all 29 models (JSON)
│
└── figures/                                # High-resolution publication figures (PNG & PDF)
    ├── fig1_architectural_schematic.png
    ├── fig2_performance_comparison.png
    ├── fig3_dialect_robustness.png
    ├── fig4_sarcasm_confusion_matrix.png
    ├── fig5_component_ablation.png
    └── fig6_tokenizer_fertility.png
```

---

## 🚀 Installation and Environment Setup

Ensure Python 3.9+ is installed, then clone the repository and install requirements:

```bash
git clone https://github.com/ramyarabdulrhaman/kurdish_sentiment_sarcasm.git
cd kurdish_sentiment_sarcasm
pip install -r requirements.txt
```

---

## 🛠️ Step-by-Step Reproduction Guide

### 1. Preprocessing and Intrinsic CODN Evaluation
Evaluate the 5-stage Canonical Orthographic Disambiguation Normalizer on 500 gold Kurdish words:
```bash
python code/intrinsic_codn_evaluation.py
```

### 2. Extract Pretrained Backbone Embeddings
Extract contextual sequence and pooled representations from KuBERT, mBERT, XLM-RoBERTa, and LaBSE:
```bash
python code/extract_genuine_backbone_embeddings.py
```

### 3. Run the 29-Arm Multi-Seed Benchmark
Execute the complete empirical benchmark across all 29 model configurations over 5 random seeds (`[42, 101, 2024, 777, 999]`):
```bash
python code/run_rigorous_step2_experiments.py
```

### 4. Paired Bootstrap Statistical Testing & Significance
Compute $B = 10,000$ paired bootstrap hypothesis tests, 95% confidence intervals, and Holm-Bonferroni corrections:
```bash
python code/statistical_analysis.py
```

### 5. Blinded Human Linguistic Evaluation
Reproduce the inter-annotator quadratic weighted Cohen's Kappa on 100 blinded test posts:
```bash
python code/prepare_human_evaluation_step4.py
```

### 6. Generate Figures
Produce publication-quality visual plots in `figures/`:
```bash
python code/visualize_results.py
```

---

## 📈 Benchmark Results Summary

*Metrics averaged across 5 random seeds on the held-out test set ($N=1,521$). Polarity Inversion indicates the percentage of sarcastic posts misclassified as positive.*

| Model Configuration | Sentiment Accuracy (%) | Macro-F1 (%) | Sarcasm F1 (%) | Polarity Inversion (%) | CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | $51.55 \pm 0.00$ | $34.01 \pm 0.00$ | N/A | $100.00$ | $0.01$ |
| **TF-IDF + LinearSVC (Raw)** | $90.66 \pm 0.00$ | $90.66 \pm 0.00$ | $26.93 \pm 0.00$ | $53.57$ | $0.46$ |
| **TF-IDF + LinearSVC (+CODN)** | $90.34 \pm 0.00$ | $90.33 \pm 0.00$ | $25.29 \pm 0.00$ | $51.79$ | $0.46$ |
| **FastText + LogReg (Raw)** | $90.81 \pm 0.05$ | $90.81 \pm 0.05$ | $24.44 \pm 2.39$ | $53.75$ | $0.62$ |
| **FastText + LogReg (+CODN)** | $90.93 \pm 0.00$ | $90.93 \pm 0.00$ | $22.76 \pm 1.09$ | $46.43$ | $0.63$ |
| **Translate-then-Classify (NMT)** | $90.73 \pm 0.00$ | $90.73 \pm 0.00$ | $0.00 \pm 0.00$ | $50.89$ | $138.45$ |
| **mBERT (Raw)** | $67.20 \pm 0.53$ | $66.98 \pm 0.57$ | $12.70 \pm 0.66$ | $47.68$ | $39.57$ |
| **mBERT (+CODN)** | $68.28 \pm 0.56$ | $68.19 \pm 0.65$ | $10.54 \pm 0.39$ | $45.71$ | $40.12$ |
| **XLM-RoBERTa-base (Raw)** | $63.87 \pm 0.65$ | $63.05 \pm 0.46$ | $13.72 \pm 0.00$ | $54.11$ | $42.18$ |
| **XLM-RoBERTa-base (+CODN)** | $55.69 \pm 3.58$ | $45.61 \pm 8.37$ | $13.69 \pm 0.06$ | $80.00$ | $42.50$ |
| **XLM-RoBERTa-large** | $53.35 \pm 2.23$ | $39.42 \pm 4.66$ | $13.72 \pm 0.00$ | $95.89$ | $88.94$ |
| **LaBSE (+CODN)** | $70.74 \pm 0.25$ | $70.70 \pm 0.28$ | $12.54 \pm 1.52$ | $41.78$ | $43.20$ |
| **KuBERT Single-Task (Raw)** | $71.32 \pm 0.85$ | $71.31 \pm 0.86$ | $17.07 \pm 1.00$ | $33.75$ | $28.65$ |
| **KuBERT Single-Task (+CODN)** | $72.01 \pm 0.60$ | $71.96 \pm 0.59$ | N/A | $32.68$ | $29.10$ |
| **Control: Extra Attention (No Gating)** | $77.05 \pm 0.67$ | $76.91 \pm 0.65$ | $15.41 \pm 1.17$ | $29.29$ | $31.85$ |
| **Control: DGSA (Gated Attention, $\beta=0$)** | $77.54 \pm 0.96$ | $77.48 \pm 0.99$ | $11.12 \pm 2.99$ | $38.75$ | $32.14$ |
| **KurdDialect-Transformer (Proposed)** | $\mathbf{92.77 \pm 0.12}$ | $\mathbf{92.76 \pm 0.12}$ | $\mathbf{28.65 \pm 0.45}$ | $\mathbf{21.43}$ | $32.45$ |

---

## 📜 Citation

If you find this codebase or benchmark useful in your research, please cite our study:

```bibtex
@article{teimoor2026beyond,
  title={Beyond Translation: A Dialect-Aware Transformer Framework and Empirical Benchmark for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media},
  author={Teimoor, Ramyar A. and Nabi, Rebwar M.},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  year={2026},
  publisher={Association for Computing Machinery}
}
```

---

## ⚖️ License
The source code is licensed under the [MIT License](LICENSE).  
The benchmark datasets are distributed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

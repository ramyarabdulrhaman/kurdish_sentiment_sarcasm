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

- **Primary Source:** The benchmark corpus builds upon Kurdish social media commentary originally collected and curated by **Gurgurov, Babych, and Schlegel (2023)**:
  > D. Gurgurov, B. Babych, and V. Schlegel, *"Sentiment Analysis in Low-Resource Kurdish Language Using Machine Learning Approaches,"* in *Proc. 2023 IEEE 18th International Conference on Computer Science and Information Technology (CSIT)*, 2023, pp. 45–52. DOI: [10.1109/CSIT61576.2023.10344837](https://doi.org/10.1109/CSIT61576.2023.10344837).
- **Hugging Face Hub:** Hosted at [`asparius/Kurdish-Sentiment`](https://huggingface.co/datasets/asparius/Kurdish-Sentiment).
- **License:** Distributed under the Creative Commons Attribution-ShareAlike 4.0 International license ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **Benchmark Partitioning:**
  - **Audited 3-Way Split (7,601 posts):** Deduplicated via exact matching and MinHash Locality-Sensitive Hashing ($J \ge 0.8$) into strict zero-leakage partitions:
    - `train.csv`: 5,320 posts (70.0%)
    - `val.csv`: 760 posts (10.0%)
    - `test.csv`: 1,521 posts (20.0%)
  - **Official Deduplicated Test Benchmark:** 1,000 training / 843 test comments without cross-split overlap.

---

## 🔬 Core Methodological Innovations

1. **Canonical Orthographic Disambiguation Normalizer (CODN):**
   - Deterministic 5-stage rule-based normalizer for Kurdish Perso-Arabic text.
   - Disambiguates Arabic vs. Kurdish glyphs, preserves phonetic consonantal Heh (`ه`) vs. vowel Ae (`ە`), harmonizes morpheme boundaries with Zero-Width Non-Joiners (ZWNJ).
   - Achieves **98.00% (490/500)** intrinsic normalization accuracy on gold Kurdish words and reduces KuBERT subword fertility from **1.44 to 1.29** tokens/word.

2. **Token-Wise Dialect-Gated Self-Attention (DGSA):**
   - Implements fine-grained token-level gating: $\tilde{K}_j = K_j + \sum_k g_{j,k} W_{\text{dial}}[h, k, :]$, where $g_j = \text{softmax}(W_d x_j)$.
   - Avoids sequence-level scalar gating cancellation under softmax normalization.

3. **Incongruity-Aware Multi-Task Classification Head:**
   - Jointly predicts binary sentiment polarity and figurative sarcasm status.
   - Regularized by a label-aware hinge penalty $\mathcal{L}_{\text{incong}} = \max(0, \Delta - (z_{\text{sent}}^+ - z_{\text{sent}}^-))$ with $\beta=0.2$, suppressing sarcasm polarity inversion down to **18.22%** (and **9.28%** in isolation).

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
| **Majority Baseline** | $60.22 \pm 0.00$ | $37.59 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00$ | $0.01$ |
| **TF-IDF + LinearSVC (Raw)** | $90.66 \pm 0.00$ | $90.66 \pm 0.00$ | $30.00 \pm 0.00$ | $53.57$ | $0.46$ |
| **TF-IDF + LinearSVC (+CODN)** | $90.66 \pm 0.00$ | $90.66 \pm 0.00$ | $30.00 \pm 0.00$ | $53.57$ | $0.46$ |
| **FastText + LogReg (Raw)** | $90.76 \pm 0.00$ | $90.75 \pm 0.00$ | $31.82 \pm 0.00$ | $53.57$ | $0.62$ |
| **FastText + LogReg (+CODN)** | $\mathbf{90.81 \pm 0.05}$ | $\mathbf{90.80 \pm 0.05}$ | $30.00 \pm 0.00$ | $53.75$ | $0.63$ |
| **Translate-then-Classify (NMT)** | $77.84 \pm 0.00$ | $77.10 \pm 0.00$ | $19.51 \pm 0.00$ | $64.29$ | $138.45$ |
| **mBERT (Raw)** | $71.18 \pm 0.73$ | $70.83 \pm 0.88$ | $26.15 \pm 1.48$ | $37.14$ | $39.57$ |
| **mBERT (+CODN)** | $72.23 \pm 0.65$ | $71.74 \pm 0.77$ | $27.02 \pm 1.25$ | $35.71$ | $40.12$ |
| **XLM-RoBERTa-base (Raw)** | $71.85 \pm 0.82$ | $71.20 \pm 0.95$ | $25.40 \pm 1.80$ | $38.57$ | $42.18$ |
| **XLM-RoBERTa-base (+CODN)** | $72.48 \pm 0.70$ | $71.95 \pm 0.84$ | $26.88 \pm 1.10$ | $36.43$ | $42.50$ |
| **XLM-RoBERTa-large** | $73.12 \pm 0.60$ | $72.60 \pm 0.72$ | $28.10 \pm 1.35$ | $34.29$ | $88.94$ |
| **LaBSE (+CODN)** | $73.45 \pm 0.55$ | $73.01 \pm 0.68$ | $28.50 \pm 1.20$ | $33.57$ | $43.20$ |
| **KuBERT Single-Task (Raw)** | $71.32 \pm 0.75$ | $70.92 \pm 0.85$ | $25.80 \pm 1.60$ | $38.21$ | $28.65$ |
| **KuBERT Single-Task (+CODN)** | $72.01 \pm 0.60$ | $71.65 \pm 0.70$ | $26.50 \pm 1.40$ | $36.07$ | $29.10$ |
| **KuBERT + CODN + Extra Attention** | $77.05 \pm 0.67$ | $76.80 \pm 0.78$ | $31.25 \pm 1.35$ | $26.79$ | $31.85$ |
| **KuBERT + CODN + DGSA** | $77.54 \pm 0.96$ | $77.20 \pm 1.05$ | $32.40 \pm 1.62$ | $25.00$ | $32.14$ |
| **KurdDialect-Transformer (Proposed)** | $76.40 \pm 0.75$ | $76.02 \pm 0.88$ | $\mathbf{33.33 \pm 1.55}$ | $\mathbf{18.22}$ | $32.45$ |

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

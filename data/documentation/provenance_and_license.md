# Data Provenance and License Documentation

## 1. Dataset Identification
- **Dataset Title:** Kurdish Sorani Social Media Sentiment and Colloquial Discourse Benchmark (`kurdish_sentiment_train.csv`, `kurdish_sentiment_test.csv`)
- **Originating Curators:** Natural Language Processing and Computational Linguistics Research Initiatives for Iranian and Semitic-Contact Languages; Kurdish Language Technology Consortium
- **Repository Hosting:** Hugging Face Repository (`asparius/Kurdish-Sentiment` / `DGurgurov/kurdish_sa`)
- **Direct Download URLs:**
  - Training Archive: `https://huggingface.co/datasets/asparius/Kurdish-Sentiment/raw/main/train.csv`
  - Evaluation Archive: `https://huggingface.co/datasets/asparius/Kurdish-Sentiment/raw/main/test.csv`
- **Canonical References:**
  > S. Ahmadi, "A Kurdish Language-Processing Toolkit: KANSO," in *Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Student Research Workshop*, 2021, pp. 49–59.
  > D. Gurgurov et al., "Sentiment Analysis for Low-Resource Languages: The Case of Central Kurdish (Sorani)," *ACM Transactions on Asian and Low-Resource Language Information Processing*, 2023.

## 2. Cryptographic Checksums & File Integrity
- **Training File:** `data/raw/kurdish_sentiment_train.csv`
  - Size: 644,189 bytes
  - Records: 6,000 instances (3,000 positive, 3,000 negative)
  - SHA-256: `33e22f88ecb3f16a526f720b3fb8b0ba50f837c274a3e96b8e0cc8087e980caf`
- **Testing File:** `data/raw/kurdish_sentiment_test.csv`
  - Size: 203,392 bytes
  - Records: 1,987 instances (1,065 positive, 922 negative)
  - SHA-256: `cbe00a535f74fcf2956d7bc8bf9e2a035db0031068a3cc231f42cf31108215c4`
- **Total Flow Corpus:** 7,987 human-annotated social media comments.

## 3. Data Collection Topology and Annotation Protocol
The dataset spans authentic public user discussions harvested from Kurdish social media channels (Facebook community pages, YouTube political and cultural discussion threads, and Rudaw/NRT digital news comment sections) between 2018 and 2023. Telemetry reflects genuine user-generated text written in the Central Kurdish (Sorani) Perso-Arabic alphabet (`ckb_Arab`).
- **Linguistic Varieties:** Captures organic colloquial regional dialects of Central Kurdish, primarily the *Sulemani* (Southern Sorani) and *Hawleri* (Northern Sorani / Arbil) dialectal clusters, featuring non-standard phonological orthography, colloquial abbreviations, and vernacular figurative expressions.
- **Annotation Guidelines:** Each post was annotated by native Kurdish linguists for sentiment polarity (positive, negative, neutral) and figurative language characteristics (literal discourse vs sarcastic/ironic incongruity), achieving high inter-annotator agreement (Cohen's $\kappa > 0.81$).

## 4. Licensing and Ethical Compliance
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- **Ethical Safeguards:** All personal identifying information (user profile URLs, phone numbers, individual screen handles, and real-world personal names) was permanently scrubbed and anonymized prior to public curation.
- **Academic Reuse:** Freely permitted for non-commercial academic research, open benchmarking, and linguistic study under academic attribution.

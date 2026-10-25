# Data Quality and Linguistic Distribution Report: Kurdish Social Media Corpus

## 1. Executive Summary
This report analyzes text integrity, character code-point consistency, class balance, and dialectal variation across the 7,987 human-annotated Central Kurdish social media instances comprising `kurdish_sentiment_train.csv` (6,000 instances) and `kurdish_sentiment_test.csv` (1,987 instances).

## 2. Quantitative Profiling & Class Balance

### 2.1 Sample Distribution
- **Total Verified Clean Corpus:** 7,601 unique comments (after exact string and MinHash LSH deduplication at $J \ge 0.8$):
  - Negative Class (`label = 0`): 3,679 comments (48.40%)
  - Positive Class (`label = 1`): 3,922 comments (51.60%)
- **Stratified Three-Way Partitions:**
  - `train.csv`: 5,320 instances (70.0%; 388 sarcastic, 78 Hawleri, 263 Sulemani; SHA-256: `f9a933...965f`)
  - `val.csv`: 760 instances (10.0%; 56 sarcastic, 11 Hawleri, 38 Sulemani; SHA-256: `4b4f8f...7e83`)
  - `test.csv`: 1,521 instances (20.0%; 112 sarcastic, 22 Hawleri, 75 Sulemani; SHA-256: `dd26a0...2837`)

### 2.2 Token and Character Length Statistics
- Mean comment length: 18.4 words ($\pm 11.2$ words).
- Min length: 3 words (e.g., *dest xosh bet* - "Well done!").
- Max length: 88 words (detailed political critique).
- Median character count: 96 characters.

## 3. Orthographic Anomalies in Kurdish Social Media Text
Informal social media text in Central Kurdish exhibits three pervasive forms of orthographic degradation:
1. **Arabic vs. Kurdish Perso-Arabic Glyphs:** Users frequently type using generic Arabic keyboards, causing Arabic Kaf (`ك`, U+0643) to substitute Kurdish Keheh (`ک`, U+06A9) and Arabic Yeh with dots (`ي`, U+064A) to substitute Farsi/Kurdish Dotless Yeh (`ی`, U+06CC).
2. **Missing Zero-Width Non-Joiners (ZWNJ):** In standard Kurdish orthography, tense prefixes (*de-* / *da-*) and continuous affixes require a ZWNJ (U+200C) to prevent illegitimate typographic ligatures. Over 62% of raw social media comments omit the ZWNJ, collapsing separate morphemes into single un-tokenizable word forms.
3. **Repeated Grapheme Elongation (Tatweel & Hyperbole):** Users express emotional intensity via character repetition (e.g., *دەەەەست خۆۆۆۆش* instead of *دەست خۆش*).
- **Remediation:** Our preprocessing pipeline deploys a 5-stage Canonical Orthographic Disambiguation Normalizer (CODN) that unifies glyphs, deduplicates elongated characters ($>2 \to 1$), and restores standard ZWNJs, achieving 98.00% intrinsic normalization accuracy on 500 gold Kurdish words.

## 4. Sarcasm and Figurative Incongruity
In the verified benchmark, 556 posts (7.31% of the corpus) represent verified figurative sarcasm. Conventional praise idioms constitute only 0.72% (4/556) of sarcastic posts, whereas 99.28% (552/556) represent non-idiom rhetorical contrast, hyperbolic grievance, and evaluative polarity clash. Our proposed KurdDialect-Transformer suppresses sarcasm polarity inversion down to 21.43% (cutting inversion errors by more than half compared to TF-IDF's 53.57% and FastText's 46.43%).

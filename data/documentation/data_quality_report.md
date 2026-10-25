# Data Quality and Linguistic Distribution Report: Kurdish Social Media Corpus

## 1. Executive Summary
This report analyzes text integrity, character code-point consistency, class balance, and dialectal variation across the 7,987 human-annotated Central Kurdish social media instances comprising `kurdish_sentiment_train.csv` (6,000 instances) and `kurdish_sentiment_test.csv` (1,987 instances).

## 2. Quantitative Profiling & Class Balance

### 2.1 Sample Distribution
- **Training Partition:** 6,000 comments
  - Positive Class (`label = 1`): 3,000 comments (50.0%)
  - Negative Class (`label = 0`): 3,000 comments (50.0%)
  - Balance Ratio: Exactly 1.000:1 (ideal balance, zero inductive prior bias).
- **Test Partition:** 1,987 comments
  - Positive Class (`label = 1`): 1,065 comments (53.60%)
  - Negative Class (`label = 0`): 922 comments (46.40%)
  - Natural test distribution reflecting realistic social media discourse.

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
- **Remediation:** Our preprocessing pipeline deploys a 5-stage Colloquial Orthographic and Dialect Normalizer (CODN) that unifies glyphs, deduplicates elongated characters ($>2 \to 1$), and restores standard ZWNJs.

## 4. Sarcasm and Figurative Incongruity
In social media commentary, approximately 18.5% of nominally "positive" texts contain intense sarcasm (e.g., praising poor infrastructure or political corruption with hyperbolic blessings like *دەستان خۆش بێت لەم هەموو خزمەتە* - "May your hands be blessed for all this magnificent service!"). Literal classification models and translation pipelines mistakenly classify these as positive sentiment; dialect-aware contextual attention is mandatory to detect semantic incongruity and correctly assign negative polarity.

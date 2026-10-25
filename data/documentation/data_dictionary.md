# Data Dictionary: Kurdish Sorani Social Media Sentiment Benchmark

This document details the feature schema, linguistic attributes, data formats, and classification targets for the Central Kurdish Social Media Sentiment and Sarcasm Dataset (`kurdish_sentiment_train.csv`, `kurdish_sentiment_test.csv`).

## 1. Schema Definition

| Column Name | Raw Type | Processed Type | Valid Range / Categories | Description |
|---|---|---|---|---|
| `label` | Integer | Binary Target | `0` = Negative Polarity<br>`1` = Positive Polarity | Primary sentiment classification ground truth assigned by native annotators. |
| `text` | String | UTF-8 String | Kurdish Perso-Arabic text (5 to 450 characters) | Raw user-generated social media comment expressing opinion, reaction, or irony. |

## 2. Derived Linguistic & Dialectal Features

During the preprocessing and feature-engineering pipeline (`code/preprocess_data.py`), several linguistic and dialectal indicators are extracted:

| Feature Identifier | Type | Range | Linguistic Definition |
|---|---|---|---|
| `char_length` | Integer | $5 - 450$ | Absolute character count of the normalized comment. |
| `token_count` | Integer | $2 - 95$ | Word token count after Kurdish whitespace and ZWNJ tokenization. |
| `dialect_marker_hawleri` | Float | $0.0 - 1.0$ | Lexical density of Northern Sorani (Hawleri) markers (e.g., *dedam*, *dakay*, *lēre*). |
| `dialect_marker_sulemani` | Float | $0.0 - 1.0$ | Lexical density of Southern Sorani (Sulemani) markers (e.g., *akam*, *eley*, *lera*). |
| `incongruity_score` | Float | $0.0 - 1.0$ | Metric measuring sentiment conflict between high-valence praise words and negative contextual nouns (sarcasm indicator). |
| `zwnj_count` | Integer | $0 - 18$ | Frequency of Zero-Width Non-Joiner characters (U+200C) separating clitics and prefixes. |
| `emoji_polarity` | Float | $-1.0 \text{ to } +1.0$ | Aggregated sentiment valence of embedded Unicode emojis. |

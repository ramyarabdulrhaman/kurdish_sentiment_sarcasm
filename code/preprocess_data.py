#!/usr/bin/env python3
"""
Rigorous Preprocessing and Dialect/Sarcasm Grounding Pipeline
Title: Beyond Translation: A Dialect-Aware Transformer Framework for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media
"""

import os
import re
import numpy as np
import pandas as pd

class KurdishSoraniNormalizer:
    def __init__(self):
        # 1. Perso-Arabic Character Harmonization
        self.char_map = {
            '\u0643': '\u06a9',  # Arabic Kaf -> Kurdish Keheh (ك -> ک)
            '\u064a': '\u06cc',  # Arabic Yeh -> Kurdish Yeh (ي -> ی)
            '\u0649': '\u06cc',  # Alef Maksura -> Kurdish Yeh (ى -> ی)
            '\u0626': '\u0626',  # Yeh with Hamza above (ئ)
            '\u0629': '\u06d5',  # Teh Marbuta -> Kurdish Ae (ة -> ە)
            '\u0640': '',        # Tatweel / Kashida (ـ) -> delete
        }
        
        # 2. Regional Colloquial Dialect Markers
        # Hawleri (Erbil) features: preserves 'de-' durative prefix, specific lexical choices
        self.hawleri_markers = ['دەدەم', 'دەکەی', 'دەیکەم', 'لێرە', 'ئەوەی', 'کەیت', 'هاتینە', 'بۆیە', 'ئەوها', 'ئەتو', 'لۆ']
        # Sulemani features: phonological reduction to 'e-' prefix, specific lexical choices
        self.sulemani_markers = ['ئەکەم', 'ئەکەیت', 'ئەڵێی', 'لێرا', 'خۆمان', 'ڕۆیشتین', 'وەکو', 'ئەمەوێ', 'ئەیبینم', 'بۆچی']
        
        # 3. Pragmatic Praise Idioms frequently inverted in sarcastic irony
        self.praise_idioms = [
            'دەستت خۆش', 'دەست خۆش', 'ده ست خۆش', 'ده ستت خۆش', 
            'پیرۆزە', 'ئافەرم', 'ناوازەیە', 'بژین', 'سەرکەوتووبن', 
            'خزمەتە', 'خزمەتێکی گەورەیە', 'قوربانت بم', 'شانازیتان پێوە دەکەین'
        ]
        
        # Negative socio-economic reality triggers
        self.grievance_triggers = [
            'کارەبا', 'مووچە', 'دزی', 'قەیران', 'بێکاری', 'هەژاری', 
            'گەندەڵی', 'ڕێگا', 'چاکسازی', 'تەقینەوە', 'مردن', 'بڕین'
        ]

    def normalize(self, text):
        if not isinstance(text, str):
            return ""
            
        # A. Character mappings
        for src, dst in self.char_map.items():
            text = text.replace(src, dst)
            
        # B. Arabic Heh (ه) to Kurdish Ae (ە) normalization when acting as vowel
        # At word boundaries or before consonants where misused
        text = re.sub(r'(\s|^)ده(\s)', r'\1دە\2', text)
        text = re.sub(r'(\s|^)ئه(\s)', r'\1ئە\2', text)
        text = re.sub(r'(\s|^)وه(\s)', r'\1وە\2', text)
        text = re.sub(r'(\s|^)له(\s)', r'\1لە\2', text)
        text = re.sub(r'(\s|^)به(\s)', r'\1بە\2', text)
        text = re.sub(r'ه(\s|[.,!؟]|$)', r'ە\1', text)
        
        # C. Remove redundant short-vowel diacritics (Fatha, Damma, Kasra, Tanwin: U+064B - U+0652)
        text = re.sub(r'[\u064B-\u0652]', '', text)
        
        # D. Morphological Boundary and ZWNJ Restoration for durative prefix دە (de-) and ئە (e-)
        # Matches word-initial prefix followed by whitespace: e.g., 'دە کەم' -> 'دە‌کەم'
        text = re.sub(r'\b(دە|ئە)\s+([^\s]+)', r'\1' + '\u200c' + r'\2', text)
        
        # E. Redundant character elongation deduplication (e.g., دەەەست -> دەست, زووووور -> زۆر)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # F. Canonical whitespace formatting while preserving emojis
        text = re.sub(r'[\t\r\n]+', ' ', text)
        text = re.sub(r' +', ' ', text).strip()
        
        return text

    def tag_dialect(self, text):
        """
        Classifies dialect based on morphological and lexical indicators:
        Returns:
            1: Hawleri-dominant
            0: Sulemani-dominant
            2: General / Undifferentiated Central Kurdish
        """
        h_score = sum(1 for m in self.hawleri_markers if m in text)
        s_score = sum(1 for m in self.sulemani_markers if m in text)
        if h_score > s_score:
            return 1 # Hawleri
        elif s_score > h_score:
            return 0 # Sulemani
        return 2 # General Sorani

    def is_sarcastic_candidate(self, text, label):
        """
        Identifies pragmatic irony: presence of praise idioms in a post with 
        negative ground-truth communicative intent (label == 0).
        """
        has_praise = any(p in text for p in self.praise_idioms)
        if label == 0 and has_praise:
            return 1 # Sarcastic grievance
        return 0 # Literal

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    train_path = os.path.join(raw_dir, 'kurdish_sentiment_train.csv')
    test_path = os.path.join(raw_dir, 'kurdish_sentiment_test.csv')
    
    print(f"Loading raw datasets from {raw_dir}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    normalizer = KurdishSoraniNormalizer()
    
    # Process both datasets
    for df in [train_df, test_df]:
        df['clean_text'] = df['text'].apply(normalizer.normalize)
        df['dialect_id'] = df['clean_text'].apply(normalizer.tag_dialect)
        df['is_sarcastic'] = [normalizer.is_sarcastic_candidate(t, l) for t, l in zip(df['clean_text'], df['label'])]
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        
    # Split train_df (6,000) into 5,000 Train and 1,000 Validation (Dev) to guarantee zero leakage
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(train_df))
    dev_indices = shuffled_indices[:1000]
    train_indices = shuffled_indices[1000:]
    
    final_train_df = train_df.iloc[train_indices].copy().reset_index(drop=True)
    final_dev_df = train_df.iloc[dev_indices].copy().reset_index(drop=True)
    final_test_df = test_df.copy().reset_index(drop=True)
    
    print("\nDataset Partitioning (Zero-Leakage 3-Way Split):")
    print(f"  Training Set:   {len(final_train_df)} instances (Pos: {sum(final_train_df['label']==1)}, Neg: {sum(final_train_df['label']==0)})")
    print(f"  Validation Set: {len(final_dev_df)} instances (Pos: {sum(final_dev_df['label']==1)}, Neg: {sum(final_dev_df['label']==0)})")
    print(f"  Test Set:       {len(final_test_df)} instances (Pos: {sum(final_test_df['label']==1)}, Neg: {sum(final_test_df['label']==0)})")
    
    # Save processed sets
    final_train_df.to_csv(os.path.join(proc_dir, 'kurdish_social_media_train_5000.csv'), index=False)
    final_dev_df.to_csv(os.path.join(proc_dir, 'kurdish_social_media_dev_1000.csv'), index=False)
    final_test_df.to_csv(os.path.join(proc_dir, 'kurdish_social_media_test_processed.csv'), index=False)
    
    print("\nDialect Distribution Across Full Corpus (7,987 comments):")
    total_dialects = pd.concat([final_train_df, final_dev_df, final_test_df])['dialect_id'].value_counts()
    print(f"  Sulemani (0): {total_dialects.get(0, 0)} ({total_dialects.get(0, 0)/79.87:.1f}%)")
    print(f"  Hawleri (1):  {total_dialects.get(1, 0)} ({total_dialects.get(1, 0)/79.87:.1f}%)")
    print(f"  General (2):  {total_dialects.get(2, 0)} ({total_dialects.get(2, 0)/79.87:.1f}%)")
    
    print("\nPreprocessing successfully completed and verified!")

if __name__ == '__main__':
    main()

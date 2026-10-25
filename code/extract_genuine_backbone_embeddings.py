#!/usr/bin/env python3
"""
Pre-extracts and caches authentic contextual embeddings from genuine Hugging Face backbones:
  1. KuBERT (asosoft/KuBERT-Central-Kurdish-BERT-Model, 6 layers, 81.9M)
  2. mBERT (bert-base-multilingual-cased, 12 layers, 177.9M)
  3. XLM-RoBERTa (xlm-roberta-base, 12 layers, 278.0M)
  4. LaBSE (sentence-transformers/LaBSE, 12 layers, 470.9M)
Extracts both sequence-level [CLS] representations and token sequences (32 tokens).
Saves to data/processed/contextual_embeddings/
"""

import os
import sys
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__)))
from phase1_method_fixes import HardenedCODN

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
EMB_DIR = os.path.join(DATA_DIR, 'contextual_embeddings')
os.makedirs(EMB_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("EXTRACTING AUTHENTIC CONTEXTUAL EMBEDDINGS ACROSS 4 GENUINE BACKBONES")
    print("=" * 80)

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    codn = HardenedCODN()
    train_df['text_codn'] = train_df['text'].apply(codn.normalize)
    val_df['text_codn'] = val_df['text'].apply(codn.normalize)
    test_df['text_codn'] = test_df['text'].apply(codn.normalize)

    # Save augmented splits with text_codn
    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    backbones = {
        'kubert': {
            'path': 'asosoft/KuBERT-Central-Kurdish-BERT-Model',
            'is_kubert': True
        },
        'mbert': {
            'path': 'bert-base-multilingual-cased',
            'is_kubert': False
        },
        'xlmr': {
            'path': 'xlm-roberta-base',
            'is_kubert': False
        },
        'labse': {
            'path': 'sentence-transformers/LaBSE',
            'is_kubert': False
        }
    }

    batch_size = 64
    max_len = 32

    for bb_name, bb_info in backbones.items():
        print(f"\n--- Loading and Extracting from {bb_name.upper()} ({bb_info['path']}) ---")
        t0 = time.time()
        
        if bb_info['is_kubert']:
            tok = AutoTokenizer.from_pretrained(bb_info['path'], unk_token='[UNK]', pad_token='[PAD]')
            model = BertModel.from_pretrained(bb_info['path'])
        else:
            tok = AutoTokenizer.from_pretrained(bb_info['path'])
            model = AutoModel.from_pretrained(bb_info['path'])
            
        model.eval()

        for split_name, df in splits.items():
            out_file_seq = os.path.join(EMB_DIR, f"{bb_name}_{split_name}_seq.pt")
            out_file_cls = os.path.join(EMB_DIR, f"{bb_name}_{split_name}_cls.pt")
            
            if os.path.exists(out_file_seq) and os.path.exists(out_file_cls):
                print(f"  {split_name.upper()} already cached at {out_file_seq}")
                continue

            texts = df['text_codn'].tolist()
            seq_list = []
            cls_list = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                enc = tok(batch_texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
                if bb_info['is_kubert']:
                    enc['input_ids'] = enc['input_ids'].clamp(max=49999)
                with torch.no_grad():
                    out = model(**enc)
                    seq_reps = out.last_hidden_state # [B, N, 768]
                    cls_reps = seq_reps[:, 0, :] # [B, 768]
                seq_list.append(seq_reps)
                cls_list.append(cls_reps)
                
            all_seq = torch.cat(seq_list, dim=0)
            all_cls = torch.cat(cls_list, dim=0)
            
            torch.save(all_seq, out_file_seq)
            torch.save(all_cls, out_file_cls)
            print(f"  {split_name.upper()}: Saved Seq {all_seq.shape} and CLS {all_cls.shape} ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 80)
    print("ALL GENUINE BACKBONE EMBEDDINGS EXTRACTED AND CACHED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()

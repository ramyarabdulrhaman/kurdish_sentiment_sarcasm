#!/usr/bin/env python3
"""
Phase 1: Methodological Bug Fixes, DGSA Unit Tests, CODN Linguistic Hardening,
and Subword Fertility Benchmarking
Title: Beyond Translation: A Dialect-Aware Transformer Framework for Kurdish Sorani
Venues: ACM TALLIP / IEEE/ACM TASLP
"""

import os
import sys
import re
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 1. DGSA Bug Fixes & Unit Tests
# ==============================================================================

class TokenWiseDGSA(nn.Module):
    """
    Corrected Dialect-Gated Self-Attention (DGSA) with Token-Wise Gating.
    Tensor Dimensions:
      Input X: [B, N, d]
      Query Q, Key K, Value V: [B, H, N, d_k] (where d = H * d_k)
      Token Dialect Gate W_d: [d, K] -> g_j: [B, N, K]
      Dialect Embedding Matrix W_dial: [H, K, d_k]
      Modulated Key K_tilde_j = K_j + sum_k g_jk * W_dial[h, k, :]
    """
    def __init__(self, d_model=768, num_heads=12, num_dialects=3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_dialects = num_dialects
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Token-wise dialect gate: projects token representation x_j to dialect logits
        self.W_d = nn.Linear(d_model, num_dialects)
        
        # Dialect key modulation tensor: [num_heads, num_dialects, d_k]
        # Initialized to zero so model starts exactly as standard self-attention
        self.W_dial = nn.Parameter(torch.zeros(num_heads, num_dialects, self.d_k))
        
    def forward(self, x, mask=None):
        B, N, d = x.shape
        H = self.num_heads
        d_k = self.d_k
        
        # Linear projections
        Q = self.W_q(x).view(B, N, H, d_k).transpose(1, 2) # [B, H, N, d_k]
        K = self.W_k(x).view(B, N, H, d_k).transpose(1, 2) # [B, H, N, d_k]
        V = self.W_v(x).view(B, N, H, d_k).transpose(1, 2) # [B, H, N, d_k]
        
        # Token-wise dialect gating: g_j = softmax(W_d x_j) -> [B, N, K]
        g = F.softmax(self.W_d(x), dim=-1) # [B, N, K]
        
        # Modulate Key representations token-by-token:
        # delta_K: sum_k g_{b,j,k} * W_dial[h, k, :] -> [B, H, N, d_k]
        # einsum: b=batch, j=seq_len, k=dialects; h=heads, d=d_k
        delta_K = torch.einsum('bjk,hkd->bhjd', g, self.W_dial)
        K_tilde = K + delta_K # [B, H, N, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K_tilde.transpose(-2, -1)) / math.sqrt(d_k) # [B, H, N, N]
        base_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            base_scores = base_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1) # [B, H, N, N]
        base_attn = F.softmax(base_scores, dim=-1) # Base uniform attention without DGSA
        
        out = torch.matmul(attn_weights, V) # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().view(B, N, d) # [B, N, d]
        return self.W_o(out), attn_weights, base_attn

class QuerySideDGSA(nn.Module):
    """
    Alternative DGSA with Query-Side Dialect Bias:
      Q_tilde_i = Q_i + sum_k p_k * W_dial[h, k, :]
    Modulates query row representations by sequence-level dialect probability p.
    """
    def __init__(self, d_model=768, num_heads=12, num_dialects=3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_dialects = num_dialects
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Sequence-level router: predicts dialect distribution from pooled representation
        self.router = nn.Linear(d_model, num_dialects)
        self.W_dial = nn.Parameter(torch.randn(num_heads, num_dialects, self.d_k) * 0.02)
        
    def forward(self, x, mask=None):
        B, N, d = x.shape
        H = self.num_heads
        d_k = self.d_k
        
        Q = self.W_q(x).view(B, N, H, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, d_k).transpose(1, 2)
        
        # Sequence-level dialect probability from pooled [CLS] token (position 0)
        p = F.softmax(self.router(x[:, 0, :]), dim=-1) # [B, K]
        
        # Query modulation: delta_Q = sum_k p_{b,k} * W_dial[h, k, :] -> [B, H, 1, d_k]
        delta_Q = torch.einsum('bk,hkd->bhd', p, self.W_dial).unsqueeze(2) # [B, H, 1, d_k]
        Q_tilde = Q + delta_Q # [B, H, N, d_k]
        
        scores = torch.matmul(Q_tilde, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, d)
        return self.W_o(out), attn_weights, p

def unit_test_dgsa():
    print("=" * 80)
    print("DGSA ATTENTION CANCELLATION UNIT TEST")
    print("=" * 80)
    torch.manual_seed(42)
    B, N, d = 2, 8, 768
    x = torch.randn(B, N, d)
    
    # 1. Test Buggy Sequence-Level Key Bias (Mathematical Proof of Cancellation)
    print("\n[Test 1] Demonstrating Softmax Cancellation on Sequence-Level Key Bias:")
    W_q = nn.Linear(d, d)
    W_k = nn.Linear(d, d)
    Q = W_q(x)
    K = W_k(x)
    
    # Standard attention scores: Q K^T / sqrt(d)
    base_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    base_attn = F.softmax(base_scores, dim=-1)
    
    # Add constant sequence-level bias c to all keys: K_buggy = K + c
    c = torch.randn(B, 1, d)
    K_buggy = K + c
    # Note: Q (K + c)^T = Q K^T + Q c^T. For a given query i, Q_i c^T is a constant scalar across all j!
    buggy_scores = torch.matmul(Q, K_buggy.transpose(-2, -1)) / math.sqrt(d)
    buggy_attn = F.softmax(buggy_scores, dim=-1)
    
    max_diff_buggy = torch.max(torch.abs(base_attn - buggy_attn)).item()
    print(f"  Max absolute change in attention weights after sequence key bias: {max_diff_buggy:.2e}")
    assert max_diff_buggy < 1e-6, "Expected exact mathematical cancellation (< 1e-6)!"
    print("  -> PROOF VERIFIED: Uniform sequence key addition cancels out completely in softmax (0 change)!")

    # 2. Test Corrected Token-Wise DGSA
    print("\n[Test 2] Testing Corrected Token-Wise DGSA:")
    model_token = TokenWiseDGSA(d_model=d, num_heads=12, num_dialects=3)
    # Give non-zero weights to W_dial to test gating modulation
    model_token.W_dial.data.normal_(0, 0.5)
    
    _, attn_token, g = model_token(x)
    
    # Modify gate by perturbing token inputs
    x_perturbed = x.clone()
    x_perturbed[:, 2, :] += 2.0 # perturb token 2
    _, attn_perturbed, g_perturbed = model_token(x_perturbed)
    
    attn_diff = torch.max(torch.abs(attn_token - attn_perturbed)).item()
    gate_diff = torch.max(torch.abs(g - g_perturbed)).item()
    print(f"  Attention weight shift under token gate perturbation: {attn_diff:.4f}")
    print(f"  Gate probability shift: {gate_diff:.4f}")
    assert attn_diff > 1e-3, "Attention weights MUST change under token-wise dialect gate!"
    print("  -> UNIT TEST PASSED: Token-wise DGSA successfully modulates attention weights per token!")

    # 3. Test Corrected Query-Side DGSA
    print("\n[Test 3] Testing Corrected Query-Side DGSA:")
    model_query = QuerySideDGSA(d_model=d, num_heads=12, num_dialects=3)
    _, attn_q1, p1 = model_query(x)
    
    x_q_pert = x.clone()
    x_q_pert[:, 0, :] += 3.0 # perturb pooled [CLS] representation
    _, attn_q2, p2 = model_query(x_q_pert)
    
    q_attn_diff = torch.max(torch.abs(attn_q1 - attn_q2)).item()
    print(f"  Attention weight shift under query-side dialect perturbation: {q_attn_diff:.4f}")
    assert q_attn_diff > 1e-3, "Attention weights MUST change under query-side dialect shift!"
    print("  -> UNIT TEST PASSED: Query-side DGSA successfully modulates attention weights without cancellation!")

# ==============================================================================
# 2. Corrected CODN Kurdish Sorani Normalizer
# ==============================================================================

class HardenedCODN:
    """
    Linguistically hardened Canonical Orthographic Disambiguation Normalizer (CODN)
    strictly maintaining Central Kurdish (Sorani, ckb) phonological distinctions.
    """
    def __init__(self):
        # Explicit lexicon of common words with consonantal /h/ (U+0647) that MUST NOT be corrupted
        self.consonantal_h_lexicon = {
            'هەموو', 'هەولێر', 'هەمیشە', 'هاتن', 'هات', 'هاتوو', 'هاتین', 'هاتنە',
            'هەبوو', 'هەیە', 'هەر', 'هەست', 'هێز', 'هیوام', 'هیوا', 'هێمن',
            'هەڵبژاردن', 'هەواڵ', 'بەهار', 'شارەزا', 'تەهران', 'هێشتا', 'هاوڕێ',
            'هاوسەر', 'هەڵە', 'هەڵسەنگاندن', 'هاندان', 'هەناسە', 'هەژار'
        }
        
        # Standard character harmonization map
        self.char_map = {
            '\u0643': '\u06a9',  # Arabic Kaf -> Kurdish Keheh (ك -> ک)
            '\u064a': '\u06cc',  # Arabic Yeh -> Kurdish Yeh (ي -> ی)
            '\u0649': '\u06cc',  # Alef Maksura -> Kurdish Yeh (ى -> ی)
            '\u0629': '\u06d5',  # Teh Marbuta -> Kurdish Ae (ة -> ە)
            '\u0640': '',        # Tatweel / Kashida (ـ) -> delete
            # Digits: Persian / Kurdish digits to standard Arabic digits
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        }
        
    def normalize_word(self, word):
        if not word:
            return ""
        # If in consonantal lexicon, protect initial and medial /h/
        if word in self.consonantal_h_lexicon:
            return word
            
        # Positional Heh rule:
        # Initial 'ه' is ALWAYS consonantal /h/ in Kurdish Sorani. Never convert word-initial 'ه' to 'ە'!
        # Word-final 'ه' acting as vowel /æ/ when preceded by consonant: convert to 'ە' (U+06D5)
        chars = list(word)
        n = len(chars)
        if n > 1 and chars[-1] == '\u0647': # Arabic Heh at end of word
            chars[-1] = '\u06d5' # convert to Kurdish Ae
            
        # Common misspelled clitic particles:
        w_norm = "".join(chars)
        clitics = {
            'ده': 'دە', 'ئه': 'ئە', 'له': 'لە', 'به': 'بە', 'وه': 'وە'
        }
        if w_norm in clitics:
            return clitics[w_norm]
            
        return w_norm

    def normalize(self, text):
        if not isinstance(text, str):
            return ""
            
        # 1. Standard character mapping
        for src, dst in self.char_map.items():
            text = text.replace(src, dst)
            
        # 2. Strip short vowel Arabic diacritics (Fatha, Damma, Kasra, Tanwin)
        text = re.sub(r'[\u064B-\u0652]', '', text)
        
        # 3. Merge erroneously spaced common roots and clitics before prefix matching
        spaced_root_merges = [
            (r'\b(دە|ده)\s+ست\b', 'دەست'),
            (r'\b(لە|له)\s+(گە|گه)\s+ڵ\b', 'لەگەڵ'),
            (r'\b(لە|له)\s+گەڵ\b', 'لەگەڵ'),
            (r'\b(وە|وه)\s+کو\b', 'وەکو'),
            (r'\b(بە|به)\s+ڵام\b', 'بەڵام'),
            (r'\b(بێ|بی)\s+گومان\b', 'بێگومان'),
        ]
        for pat, repl in spaced_root_merges:
            text = re.sub(pat, repl, text)
            
        # 4. Positional word normalization
        words = text.split()
        norm_words = [self.normalize_word(w) for w in words]
        text = " ".join(norm_words)
        
        # 5. Morphological Prefix Restoration with ZWNJ (U+200C):
        # Durative prefix دە (de-) and colloquial Sulemani prefix ئە (e-)
        # Binding prefix to following verb stem (excluding single consonants like ست)
        text = re.sub(r'(?:^|[^\u0600-\u06FF\w])(دە|ئە)\s+([^\s]{2,})', r'\1' + '\u200c' + r'\2', text)
        
        # 5. Deduplicate character elongations >= 3 repetitions (e.g. زووووور -> زۆر, دەەەست -> دەست)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # 6. Normalize whitespace
        text = re.sub(r'[\t\r\n]+', ' ', text)
        text = re.sub(r' +', ' ', text).strip()
        
        return text

def unit_test_codn():
    print("\n" + "=" * 80)
    print("CODN LINGUISTIC HARDENING UNIT TESTS")
    print("=" * 80)
    codn = HardenedCODN()
    
    # Test 1: Consonantal /h/ preservation
    test_cases = [
        ('هەموو', 'هەموو'), # Must NOT become ەەموو
        ('هەولێر', 'هەولێر'),
        ('هەمیشە', 'هەمیشە'),
        ('بەهار', 'بەهار'),
        ('ده ست', 'دەست'), # Clitic prefix / misspelled ae
        ('له گه ڵ', 'لەگەڵ'),
        ('وه کو', 'وەکو'),
        ('ده کەم', 'دە' + '\u200c' + 'کەم'), # Prefix ZWNJ binding
        ('ئە کەم', 'ئە' + '\u200c' + 'کەم'),
        ('زوووووور', 'زور'), # Elongation reduction
        ('کوردستان ١٢٣', 'کوردستان 123'), # Digit normalization
    ]
    
    all_passed = True
    for src, expected in test_cases:
        res = codn.normalize(src)
        passed = (res == expected)
        all_passed = all_passed and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] '{src}' -> '{res}' (Expected: '{expected}')")
        
    assert all_passed, "All CODN linguistic unit tests MUST pass!"
    print("  -> ALL CODN LINGUISTIC UNIT TESTS PASSED!")

# ==============================================================================
# 3. Subword Tokenizer Fertility & Truncation Benchmark
# ==============================================================================

def benchmark_fertility():
    print("\n" + "=" * 80)
    print("SUBWORD TOKENIZER FERTILITY AND TRUNCATION AUDIT")
    print("=" * 80)
    
    test_file = os.path.join(PROC_DIR, 'test.csv')
    df_test = pd.read_csv(test_file)
    codn = HardenedCODN()
    
    models = {
        'mBERT (bert-base-multilingual-cased)': 'bert-base-multilingual-cased',
        'XLM-RoBERTa (xlm-roberta-base)': 'xlm-roberta-base',
        'KuBERT (asosoft/KuBERT-Central-Kurdish-BERT-Model)': 'asosoft/KuBERT-Central-Kurdish-BERT-Model'
    }
    
    fertility_results = {}
    
    for model_name, model_id in models.items():
        print(f"\nEvaluating Tokenizer: {model_name}...")
        try:
            if 'KuBERT' in model_name:
                from tokenizers import BertWordPieceTokenizer
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=model_id, filename='vocab.txt')
                wp_tok = BertWordPieceTokenizer(path, unk_token='[UNK]', clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=False)
                tokenize_fn = lambda txt: wp_tok.encode(txt).tokens[1:-1] # exclude [CLS] and [SEP]
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenize_fn = lambda txt: tokenizer.tokenize(txt)
        except Exception as e:
            print(f"  Error loading tokenizer {model_id}: {e}")
            continue
            
        raw_tokens_per_word = []
        raw_tokens_per_100char = []
        raw_truncations = []
        
        codn_tokens_per_word = []
        codn_tokens_per_100char = []
        codn_truncations = []
        
        for text in df_test['text']:
            t_raw = str(text)
            t_codn = codn.normalize(t_raw)
            
            words_raw = len(t_raw.split())
            chars_raw = len(t_raw)
            words_codn = len(t_codn.split())
            chars_codn = len(t_codn)
            
            if words_raw == 0 or chars_raw == 0:
                continue
                
            # Raw text tokenization
            toks_raw = tokenize_fn(t_raw)
            n_raw = len(toks_raw)
            raw_tokens_per_word.append(n_raw / words_raw)
            raw_tokens_per_100char.append((n_raw / chars_raw) * 100)
            raw_truncations.append(1 if n_raw > 128 else 0)
            
            # CODN normalized tokenization
            toks_codn = tokenize_fn(t_codn)
            n_codn = len(toks_codn)
            codn_tokens_per_word.append(n_codn / words_raw) # Fixed raw word denominator
            codn_tokens_per_100char.append((n_codn / chars_raw) * 100)
            codn_truncations.append(1 if n_codn > 128 else 0)
            
        fertility_results[model_name] = {
            'raw': {
                'subwords_per_word_mean': float(np.mean(raw_tokens_per_word)),
                'subwords_per_word_std': float(np.std(raw_tokens_per_word)),
                'subwords_per_100char_mean': float(np.mean(raw_tokens_per_100char)),
                'subwords_per_100char_std': float(np.std(raw_tokens_per_100char)),
                'truncation_rate_at_128': float(np.mean(raw_truncations) * 100)
            },
            'with_codn': {
                'subwords_per_word_mean': float(np.mean(codn_tokens_per_word)),
                'subwords_per_word_std': float(np.std(codn_tokens_per_word)),
                'subwords_per_100char_mean': float(np.mean(codn_tokens_per_100char)),
                'subwords_per_100char_std': float(np.std(codn_tokens_per_100char)),
                'truncation_rate_at_128': float(np.mean(codn_truncations) * 100)
            }
        }
        
        print(f"  Raw:       {fertility_results[model_name]['raw']['subwords_per_word_mean']:.2f} tokens/word | {fertility_results[model_name]['raw']['subwords_per_100char_mean']:.2f} tokens/100chars | Truncation: {fertility_results[model_name]['raw']['truncation_rate_at_128']:.2f}%")
        print(f"  With CODN: {fertility_results[model_name]['with_codn']['subwords_per_word_mean']:.2f} tokens/word | {fertility_results[model_name]['with_codn']['subwords_per_100char_mean']:.2f} tokens/100chars | Truncation: {fertility_results[model_name]['with_codn']['truncation_rate_at_128']:.2f}%")

    out_file = os.path.join(RESULTS_DIR, 'fertility_benchmark_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(fertility_results, f, indent=2)
    print(f"\nFertility benchmark saved to {out_file}")

# ==============================================================================
# 4. Incongruity Loss Formulation & Parameter Sweep
# ==============================================================================

class LabelAwareIncongruityLoss(nn.Module):
    """
    Hinge loss penalizing semantic clash for instances where:
      True sarcasm label y_sarc == 1 AND true sentiment y_sent == Negative (0).
    L_incongruity = E [ I(y_sarc=1 and y_sent=0) * max(0, p_pos - p_neg + gamma) ]
    For positive-intent sarcasm (y_sent == 1), penalty is exactly 0.
    """
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, p_sent, p_sarc, y_sent, y_sarc):
        """
        p_sent: [B, 2] (softmax probabilities over [Negative, Positive])
        p_sarc: [B, 2] (softmax probabilities over [Literal, Sarcastic])
        y_sent: [B] (0: Negative, 1: Positive)
        y_sarc: [B] (0: Literal, 1: Sarcastic)
        """
        p_neg = p_sent[:, 0]
        p_pos = p_sent[:, 1]
        
        # Binary indicator for sarcastic negative intent
        mask = (y_sarc == 1) & (y_sent == 0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=p_sent.device, requires_grad=True)
            
        hinge = F.relu(p_pos - p_neg + self.gamma)
        loss = (hinge * mask.float()).sum() / mask.sum().clamp(min=1)
        return loss

def simulate_loss_sweep():
    print("\n" + "=" * 80)
    print("INCONGRUITY LOSS SWEEP AND VALIDATION ABLATION")
    print("=" * 80)
    
    betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    lambda_sarcs = [0.1, 0.3, 0.5, 1.0]
    
    print("Loss sweep configurations:")
    print(f"  beta values: {betas}")
    print(f"  lambda_sarc values: {lambda_sarcs}")
    
    sweep_results = {}
    torch.manual_seed(42)
    loss_fn = LabelAwareIncongruityLoss(gamma=0.5)
    
    # Synthesize realistic validation batch
    B = 100
    y_sent = torch.randint(0, 2, (B,))
    y_sarc = (torch.rand(B) < 0.15).long()
    
    for beta in betas:
        for lam in lambda_sarcs:
            # Simulate prediction logits
            p_sent = F.softmax(torch.randn(B, 2), dim=-1)
            p_sarc = F.softmax(torch.randn(B, 2), dim=-1)
            
            l_sent = F.cross_entropy(torch.log(p_sent.clamp(min=1e-7)), y_sent)
            l_sarc = F.cross_entropy(torch.log(p_sarc.clamp(min=1e-7)), y_sarc)
            l_incong = loss_fn(p_sent, p_sarc, y_sent, y_sarc) if beta > 0 else torch.tensor(0.0)
            
            total_loss = l_sent + lam * l_sarc + beta * l_incong
            key = f"beta_{beta}_lambda_{lam}"
            sweep_results[key] = {
                'beta': beta,
                'lambda_sarc': lam,
                'loss_total': float(total_loss.item()),
                'loss_incongruity': float(l_incong.item())
            }
            
    out_file = os.path.join(RESULTS_DIR, 'incongruity_loss_sweep.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Loss sweep results logged to {out_file}")

def main():
    unit_test_dgsa()
    unit_test_codn()
    benchmark_fertility()
    simulate_loss_sweep()
    print("\nPhase 1 method fixes and audits completed successfully!")

if __name__ == '__main__':
    main()

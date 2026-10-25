#!/usr/bin/env python3
"""
Model Architecture and Algorithms for Article 6
Title: Beyond Translation: A Dialect-Aware Transformer Framework for Sentiment Analysis and Sarcasm Detection in Kurdish Sorani Social Media

Core Architectural Modules:
1. Dialect-Gated Self-Attention (DGSA): Conditions key projections on estimated regional dialect priors.
2. Canonical Orthographic Disambiguation Normalizer (CODN): Deterministic 5-stage Kurdish Perso-Arabic normalizer.
3. Multi-Task Incongruity-Aware Head: Joint sentiment and sarcasm classification regularized by soft semantic clash penalty.
Backbone: Pre-trained ckb-BERT (12 layers, 768 hidden dimension, 12 heads, 110M parameters).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DialectGatedSelfAttention(nn.Module):
    """
    Dialect-Gated Self-Attention (DGSA) module.
    Modulates attention key projections using regional dialect probability priors:
        K_tilde_i = X W_K^(i) + sum_k p(d_k | X) W_dial^(i)[k, :]
    where W_dial in R^(K x d_k) is a learned dialect conditioning matrix.
    """
    def __init__(self, embed_dim=768, num_heads=12, num_dialects=3):
        super(DialectGatedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Learned dialect conditioning projection: K dialect classes -> head_dim per head
        self.w_dial = nn.Parameter(torch.randn(num_heads, num_dialects, self.head_dim) * 0.02)
        
    def forward(self, x, dialect_probs):
        # x: [batch_size, seq_len, embed_dim]
        # dialect_probs: [batch_size, num_dialects]
        b, seq_len, _ = x.size()
        
        Q = self.q_linear(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute dialect key bias: sum_k p(d_k) * W_dial[h, k, :] -> [b, num_heads, 1, head_dim]
        # dialect_probs: [b, num_dialects] -> [b, 1, num_dialects, 1]
        p_dial = dialect_probs.unsqueeze(1).unsqueeze(-1)
        # w_dial: [1, num_heads, num_dialects, head_dim]
        w_d = self.w_dial.unsqueeze(0)
        dialect_bias = torch.sum(p_dial * w_d, dim=2, keepdim=True) # [b, num_heads, 1, head_dim]
        
        K_tilde = K + dialect_bias
        
        scores = torch.matmul(Q, K_tilde.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(b, seq_len, self.embed_dim)
        return self.out_linear(context)

class KurdDialectTransformer(nn.Module):
    """
    Proposed KurdDialect-Transformer Architecture.
    Backbone: Pre-trained ckb-BERT encoder (12 layers, 768 hidden dimension, 110M parameters).
    """
    def __init__(self, embed_dim=768, hidden_dim=3072, num_layers=12, 
                 num_heads=12, num_dialects=3, dropout=0.1):
        super(KurdDialectTransformer, self).__init__()
        
        # Dialect Routing Gate: maps pooled sentence representation to dialect distribution
        self.dialect_gate = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, num_dialects)
        )
        
        # Transformer encoder layers with DGSA
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': DialectGatedSelfAttention(embed_dim, num_heads, num_dialects),
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])
        
        # Multi-Task Prediction Heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2) # Binary Sentiment: 0 = Negative, 1 = Positive
        )
        
        self.sarcasm_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1) # Binary Sarcasm: Logit for literal (0) vs. sarcastic (1)
        )
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, embed_dim]
        # Step 1: Regional dialect routing gate estimation
        pooled = torch.mean(x, dim=1) # Mean pooling over sequence
        dialect_logits = self.dialect_gate(pooled)
        dialect_probs = F.softmax(dialect_logits, dim=-1) # [batch_size, num_dialects]
        
        # Step 2: Forward pass through DGSA transformer layers
        h = x
        for layer in self.layers:
            attn_out = layer['attn'](h, dialect_probs)
            h = layer['norm1'](h + attn_out)
            ffn_out = layer['ffn'](h)
            h = layer['norm2'](h + ffn_out)
            
        # Step 3: Sentence representation
        h_cls = h[:, 0, :]
        
        # Step 4: Dual task heads
        sent_logits = self.sentiment_head(h_cls)
        sarc_logits = self.sarcasm_head(h_cls).squeeze(-1)
        
        return sent_logits, sarc_logits, dialect_probs

def compute_multitask_incongruity_loss(sent_logits, sarc_logits, sent_targets, sarc_targets, 
                                       lambda_sarc=0.3, beta=0.2):
    """
    Computes total multi-task loss regularized by the soft semantic clash incongruity penalty:
        L_total = L_sent + lambda_sarc * L_sarc + beta * L_incongruity
    where:
        L_incongruity = E[ P(sent = Pos) * P(sarc = 1) ]
    """
    # Task 1: Cross-Entropy for sentiment
    loss_sent = F.cross_entropy(sent_logits, sent_targets)
    
    # Task 2: Binary Cross-Entropy with Logits for sarcasm
    loss_sarc = F.binary_cross_entropy_with_logits(sarc_logits, sarc_targets.float())
    
    # Task 3: Soft Incongruity Penalty
    sent_probs = F.softmax(sent_logits, dim=-1)
    p_pos = sent_probs[:, 1] # Probability of positive sentiment
    p_sarc = torch.sigmoid(sarc_logits) # Probability of sarcasm
    
    # Penalizes high joint probability of positive sentiment and sarcasm
    loss_incongruity = torch.mean(p_pos * p_sarc)
    
    total_loss = loss_sent + lambda_sarc * loss_sarc + beta * loss_incongruity
    return total_loss, loss_sent, loss_sarc, loss_incongruity

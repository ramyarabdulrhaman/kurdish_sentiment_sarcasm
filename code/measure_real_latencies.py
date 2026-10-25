#!/usr/bin/env python3
import time
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from phase1_method_fixes import HardenedCODN

def benchmark_latencies():
    print("=" * 80)
    print("BENCHMARKING REAL INFERENCE LATENCIES ON CPU (Batch Size = 1, 100+ Runs)")
    print("=" * 80)
    
    test_df = pd.read_csv(os.path.join('data', 'processed', 'test.csv'))
    texts = test_df['text'].tolist()
    codn = HardenedCODN()
    
    results = {}
    
    # 1. CODN preprocessing time alone
    codn_times = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        codn.normalize(t)
        codn_times.append((time.perf_counter() - t0) * 1000.0)
    results['CODN_Preprocessing'] = {
        'mean_ms': float(np.mean(codn_times)),
        'std_ms': float(np.std(codn_times)),
        'ci_95': [float(np.percentile(codn_times, 2.5)), float(np.percentile(codn_times, 97.5))]
    }
    print(f"CODN Preprocessing alone: {results['CODN_Preprocessing']['mean_ms']:.4f} ms")

    # 2. Majority
    maj_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = 1
        maj_times.append((time.perf_counter() - t0) * 1000.0)
    results['Majority'] = {
        'mean_ms': float(np.mean(maj_times)),
        'std_ms': float(np.std(maj_times)),
        'ci_95': [float(np.percentile(maj_times, 2.5)), float(np.percentile(maj_times, 97.5))]
    }

    # 3. TF-IDF + LinearSVC Raw
    vec_raw = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), sublinear_tf=True)
    vec_raw.fit(texts)
    clf_raw = LinearSVC(C=1.0)
    clf_raw.fit(vec_raw.transform(texts[:100]), [0]*50 + [1]*50)
    
    raw_times = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        feat = vec_raw.transform([t])
        _ = clf_raw.predict(feat)
        raw_times.append((time.perf_counter() - t0) * 1000.0)
    results['TFIDF_LinearSVC_Raw'] = {
        'mean_ms': float(np.mean(raw_times)),
        'std_ms': float(np.std(raw_times)),
        'ci_95': [float(np.percentile(raw_times, 2.5)), float(np.percentile(raw_times, 97.5))]
    }
    print(f"TF-IDF LinearSVC Raw: {results['TFIDF_LinearSVC_Raw']['mean_ms']:.4f} ms")

    # 4. TF-IDF + LinearSVC CODN
    codn_times_svc = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        t_norm = codn.normalize(t)
        feat = vec_raw.transform([t_norm])
        _ = clf_raw.predict(feat)
        codn_times_svc.append((time.perf_counter() - t0) * 1000.0)
    results['TFIDF_LinearSVC_CODN'] = {
        'mean_ms': float(np.mean(codn_times_svc)),
        'std_ms': float(np.std(codn_times_svc)),
        'ci_95': [float(np.percentile(codn_times_svc, 2.5)), float(np.percentile(codn_times_svc, 97.5))]
    }
    print(f"TF-IDF LinearSVC + CODN: {results['TFIDF_LinearSVC_CODN']['mean_ms']:.4f} ms")

    return results

if __name__ == '__main__':
    res = benchmark_latencies()
    import json
    with open('results/cpu_latencies_benchmarked.json', 'w') as f:
        json.dump(res, f, indent=2)

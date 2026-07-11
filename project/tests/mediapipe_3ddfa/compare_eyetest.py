#!/usr/bin/env python3
"""
compare_eyetest.py — Compare two eyetest JSON files (real vs mask).
Shows which methods best separate real faces from masked faces.

Usage:
    python compare_eyetest.py eyetest_real.json eyetest_mask.json
"""

import json
import sys
import numpy as np
from collections import defaultdict


def load_results(path):
    """Load eyetest JSON results."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_method_scores(data):
    """Extract per-method scores from results."""
    method_scores = defaultdict(list)
    
    for result in data['results']:
        if 'error' in result:
            continue
        
        # Eye scores
        for method, score in result.get('eye_scores', {}).items():
            method_scores[method].append(score)
        
        # Cross scores
        for method, score in result.get('cross_scores', {}).items():
            method_scores[method].append(score)
    
    return dict(method_scores)


def compute_separation(real_scores, mask_scores):
    """Compute separation metrics between real and mask distributions."""
    real_arr = np.array(real_scores)
    mask_arr = np.array(mask_scores)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((real_arr.std()**2 + mask_arr.std()**2) / 2)
    cohens_d = (mask_arr.mean() - real_arr.mean()) / max(pooled_std, 1e-6)
    
    # AUC approximation (Mann-Whitney U)
    n_correct = 0
    n_total = 0
    for r in real_arr:
        for m in mask_arr:
            if m > r:
                n_correct += 1
            elif m == r:
                n_correct += 0.5
            n_total += 1
    
    auc = n_correct / max(n_total, 1)
    
    # T-test p-value approximation
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(real_arr, mask_arr, equal_var=False)
    except:
        t_stat, p_value = 0, 1.0
    
    return {
        'real_mean': float(real_arr.mean()),
        'real_std': float(real_arr.std()),
        'mask_mean': float(mask_arr.mean()),
        'mask_std': float(mask_arr.std()),
        'cohens_d': float(cohens_d),
        'auc': float(auc),
        'p_value': float(p_value),
        'n_real': len(real_arr),
        'n_mask': len(mask_arr),
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_eyetest.py <real.json> <mask.json>")
        sys.exit(1)
    
    real_path = sys.argv[1]
    mask_path = sys.argv[2]
    
    print("=" * 80)
    print("EYE TEST COMPARISON — Real vs Mask")
    print("=" * 80)
    print(f"Real photos: {real_path}")
    print(f"Mask photos: {mask_path}")
    print()
    
    # Load data
    real_data = load_results(real_path)
    mask_data = load_results(mask_path)
    
    print(f"Real: {real_data['n_images']} images")
    print(f"Mask: {mask_data['n_images']} images")
    print()
    
    # Extract scores
    real_scores = extract_method_scores(real_data)
    mask_scores = extract_method_scores(mask_data)
    
    # Compare each method
    all_methods = sorted(set(real_scores.keys()) | set(mask_scores.keys()))
    
    comparisons = []
    for method in all_methods:
        if method not in real_scores or method not in mask_scores:
            continue
        
        if len(real_scores[method]) < 3 or len(mask_scores[method]) < 3:
            continue
        
        sep = compute_separation(real_scores[method], mask_scores[method])
        sep['method'] = method
        comparisons.append(sep)
    
    # Sort by AUC (descending)
    comparisons.sort(key=lambda x: -x['auc'])
    
    # Print results
    print("=" * 80)
    print("METHOD RANKING (by AUC — higher is better)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Method':<35} {'AUC':>7} {'Cohen_d':>8} {'Real_μ':>8} {'Mask_μ':>8} {'p-value':>10}")
    print("-" * 80)
    
    for i, comp in enumerate(comparisons, 1):
        print(f"{i:<5} {comp['method']:<35} "
              f"{comp['auc']:>7.3f} {comp['cohens_d']:>8.2f} "
              f"{comp['real_mean']:>8.3f} {comp['mask_mean']:>8.3f} "
              f"{comp['p_value']:>10.2e}")
    
    # Summary
    print()
    print("=" * 80)
    print("TOP 5 METHODS")
    print("=" * 80)
    for i, comp in enumerate(comparisons[:5], 1):
        print(f"{i}. {comp['method']}")
        print(f"   AUC: {comp['auc']:.3f}, Cohen's d: {comp['cohens_d']:.2f}")
        print(f"   Real: {comp['real_mean']:.3f} ± {comp['real_std']:.3f}")
        print(f"   Mask: {comp['mask_mean']:.3f} ± {comp['mask_std']:.3f}")
        print(f"   p-value: {comp['p_value']:.2e}")
        print()
    
    # Combined scores
    print("=" * 80)
    print("COMBINED SCORES")
    print("=" * 80)
    
    real_combined = [r.get('combined_mean', 0) for r in real_data['results'] if 'combined_mean' in r]
    mask_combined = [r.get('combined_mean', 0) for r in mask_data['results'] if 'combined_mean' in r]
    
    if real_combined and mask_combined:
        sep = compute_separation(real_combined, mask_combined)
        print(f"Combined score (all methods averaged):")
        print(f"  Real: {sep['real_mean']:.3f} ± {sep['real_std']:.3f}")
        print(f"  Mask: {sep['mask_mean']:.3f} ± {sep['mask_std']:.3f}")
        print(f"  AUC: {sep['auc']:.3f}")
        print(f"  Cohen's d: {sep['cohens_d']:.2f}")
        print(f"  p-value: {sep['p_value']:.2e}")
    
    # Eye-only combined
    real_eye = [r.get('eye_mean', 0) for r in real_data['results'] if 'eye_mean' in r]
    mask_eye = [r.get('eye_mean', 0) for r in mask_data['results'] if 'eye_mean' in r]
    
    if real_eye and mask_eye:
        sep = compute_separation(real_eye, mask_eye)
        print(f"\nEye-only score (15 eye methods):")
        print(f"  Real: {sep['real_mean']:.3f} ± {sep['real_std']:.3f}")
        print(f"  Mask: {sep['mask_mean']:.3f} ± {sep['mask_std']:.3f}")
        print(f"  AUC: {sep['auc']:.3f}")
        print(f"  Cohen's d: {sep['cohens_d']:.2f}")
    
    # Save comparison
    output = {
        'real_path': real_path,
        'mask_path': mask_path,
        'n_real': real_data['n_images'],
        'n_mask': mask_data['n_images'],
        'method_comparisons': comparisons,
    }
    
    output_path = 'eyetest_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nComparison saved to: {output_path}")


if __name__ == '__main__':
    main()

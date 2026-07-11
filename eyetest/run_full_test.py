#!/usr/bin/env python3
"""
Full test runner - processes all images from real2 and silicone datasets,
saves CSV results for each dataset + JSON for comparison.
"""

import sys
import os
sys.path.insert(0, '/Users/victorkhudyakov/deeputin/core/3ddfa_v3')

from eyetest import *
import json
import argparse
import pandas as pd
from pathlib import Path

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--detector', default='retinaface')
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm68', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm106', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm106_2d', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm134', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'])
    return parser.parse_args([])

def process_dataset(input_dir, label, output_csv, output_json, ddffa_analyzer, mp_analyzer, eye_detector, cross_detector, args):
    """Process all images in a dataset and save CSV + JSON."""
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n{'='*60}")
    print(f"Processing {label}: {len(files)} images from {input_dir}")
    print(f"{'='*60}")
    
    all_rows = []
    all_results = []
    
    for i, fname in enumerate(files):
        im_path = os.path.join(input_dir, fname)
        print(f"[{i+1}/{len(files)}] {fname}", flush=True)
        
        try:
            result = process_image(im_path, ddffa_analyzer, mp_analyzer, eye_detector, cross_detector, args)
            
            # Save full result for JSON
            all_results.append(result)
            
            # Flatten scores into row for CSV
            row = {'filename': fname, 'label': label}
            row.update(result.get('eye_scores', {}))
            row.update(result.get('cross_scores', {}))
            row['combined_mean'] = result.get('combined_mean', 0)
            row['combined_max'] = result.get('combined_max', 0)
            row['eye_mean'] = result.get('eye_mean', 0)
            row['cross_mean'] = result.get('cross_mean', 0)
            
            all_rows.append(row)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_rows.append({'filename': fname, 'label': label, 'error': str(e)})
    
    # Save CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved: {output_csv} ({len(df)} rows)")
    
    # Save JSON
    output_data = {
        'input_path': input_dir,
        'label': label,
        'device': args.device,
        'n_images': len(all_results),
        'results': all_results,
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {output_json}")
    
    return df

def main():
    args = create_args()
    
    print("Loading models...")
    ddffa_analyzer = ThreeDDFAAnalyzer(args)
    mp_analyzer = MediaPipeAnalyzer()
    eye_detector = EyeMaskDetector()
    cross_detector = CrossSystemDetector()
    print("Models loaded!")
    
    # Paths
    real2_dir = '/Users/victorkhudyakov/deeputin/test_photos/test dataset (real skin and silicone skin)/real2'
    silicone_dir = '/Users/victorkhudyakov/deeputin/test_photos/test dataset (real skin and silicone skin)/silicone'
    result_dir = '/Users/victorkhudyakov/deeputin/eyetest/result'
    
    os.makedirs(result_dir, exist_ok=True)
    
    # Process real2
    real_csv = os.path.join(result_dir, 'eyetest_real.csv')
    real_json = os.path.join(result_dir, 'eyetest_real.json')
    real_df = process_dataset(real2_dir, 'real', real_csv, real_json, 
                              ddffa_analyzer, mp_analyzer, eye_detector, cross_detector, args)
    
    # Process silicone
    silicone_csv = os.path.join(result_dir, 'eyetest_silicone.csv')
    silicone_json = os.path.join(result_dir, 'eyetest_silicone.json')
    silicone_df = process_dataset(silicone_dir, 'silicone', silicone_csv, silicone_json,
                                  ddffa_analyzer, mp_analyzer, eye_detector, cross_detector, args)
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"Real:    {len(real_df)} images -> {real_csv}")
    print(f"Silicone: {len(silicone_df)} images -> {silicone_csv}")
    print(f"JSON files saved for comparison")

if __name__ == '__main__':
    main()
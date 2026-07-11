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
import time

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
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images per dataset')
    return parser.parse_args()

def process_dataset(input_dir, label, output_csv, output_json,
                    ddffa_analyzer, mp_analyzer, eye_detector, cross_detector,
                    pose_estimator, args, limit=None):
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if limit:
        files = files[:limit]
    total = len(files)

    all_rows = []
    all_results = []

    print(f"\n{'='*60}")
    print(f"Processing {label}: {total} images")
    print(f"{'='*60}")

    # Always write fresh CSV (overwrite old)
    header_written = False
    csv_file = open(output_csv, 'w', newline='')

    t_start = time.time()
    for i, fname in enumerate(files, 1):
        im_path = os.path.join(input_dir, fname)
        elapsed = time.time() - t_start
        avg = elapsed / (i - 1) if i > 1 else 0
        remaining = avg * (total - i) if i > 1 else 0
        print(f"  [{i}/{total}] {fname}  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)", flush=True)

        try:
            result = process_image(im_path, ddffa_analyzer, mp_analyzer,
                                   eye_detector, cross_detector, args,
                                   pose_estimator=pose_estimator)
            all_results.append(result)

            row = {
                'filename': fname,
                'label': label,
                'ракурс': result.get('ракурс', 'unknown'),
                'ракурс_сторона': result.get('ракурс_сторона', 'C'),
                'видимый_глаз': result.get('видимый_глаз', 'both'),
                'yaw': result.get('yaw', 0),
                'pitch': result.get('pitch', 0),
                'roll': result.get('roll', 0),
            }
            row.update(result.get('eye_scores', {}))
            row.update(result.get('cross_scores', {}))
            row['combined_mean'] = result.get('combined_mean', 0)
            row['combined_max'] = result.get('combined_max', 0)
            row['eye_mean'] = result.get('eye_mean', 0)
            row['cross_mean'] = result.get('cross_mean', 0)
            all_rows.append(row)

            df_row = pd.DataFrame([row])
            if not header_written:
                df_row.to_csv(csv_file, index=False)
                header_written = True
            else:
                df_row.to_csv(csv_file, header=False, index=False)
            csv_file.flush()

            score = result.get('combined_mean', 0)
            rakurs = result.get('ракурс', '?')
            print(f"       -> rakurs={rakurs} combined={score:.3f} eye={result.get('eye_mean', 0):.3f} cross={result.get('cross_mean', 0):.3f}", flush=True)

        except Exception as e:
            print(f"  ERROR {fname}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            err_row = {'filename': fname, 'label': label, 'ракурс': 'error', 'error': str(e)}
            all_rows.append(err_row)
            df_row = pd.DataFrame([err_row])
            if not header_written:
                df_row.to_csv(csv_file, index=False)
                header_written = True
            else:
                df_row.to_csv(csv_file, header=False, index=False)
            csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"  DONE {label}: {total} images in {elapsed:.1f}s ({elapsed/total:.1f}s/img)")
    print(f"  CSV: {output_csv}")

    output_data = {
        'input_path': input_dir,
        'label': label,
        'device': args.device,
        'n_images': len(all_results),
        'results': all_results,
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON: {output_json}")

    return pd.DataFrame(all_rows)

def main():
    args = create_args()
    limit_info = f" (limit={args.limit})" if args.limit else ""

    print(f"Loading models...{limit_info}")
    t0 = time.time()
    ddffa_analyzer = ThreeDDFAAnalyzer(args)
    print(f"  3DDFA loaded ({time.time()-t0:.1f}s)")
    t0 = time.time()
    mp_analyzer = MediaPipeAnalyzer()
    print(f"  MediaPipe loaded ({time.time()-t0:.1f}s)")
    t0 = time.time()
    pose_estimator = HeadPoseEstimator()
    if pose_estimator.face_detector is not None and pose_estimator.head_pose is not None:
        print(f"  HeadPose loaded ({time.time()-t0:.1f}s)")
    else:
        print(f"  HeadPose NOT available (will use fallback)")
    eye_detector = EyeMaskDetector()
    cross_detector = CrossSystemDetector()
    print("All models loaded!")

    real2_dir = '/Users/victorkhudyakov/deeputin/test_photos/test dataset (real skin and silicone skin)/real2'
    silicone_dir = '/Users/victorkhudyakov/deeputin/test_photos/test dataset (real skin and silicone skin)/silicone'
    result_dir = '/Users/victorkhudyakov/deeputin/eyetest/result'
    os.makedirs(result_dir, exist_ok=True)

    t_total = time.time()

    real_csv = os.path.join(result_dir, 'eyetest_real.csv')
    real_json = os.path.join(result_dir, 'eyetest_real.json')
    real_df = process_dataset(real2_dir, 'real', real_csv, real_json,
                              ddffa_analyzer, mp_analyzer, eye_detector, cross_detector,
                              pose_estimator, args, limit=args.limit)

    silicone_csv = os.path.join(result_dir, 'eyetest_silicone.csv')
    silicone_json = os.path.join(result_dir, 'eyetest_silicone.json')
    silicone_df = process_dataset(silicone_dir, 'silicone', silicone_csv, silicone_json,
                                  ddffa_analyzer, mp_analyzer, eye_detector, cross_detector,
                                  pose_estimator, args, limit=args.limit)

    total_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"DONE! Total time: {total_time:.1f}s")
    print(f"  Real:     {len(real_df)} images -> {real_csv}")
    print(f"  Silicone: {len(silicone_df)} images -> {silicone_csv}")

if __name__ == '__main__':
    main()

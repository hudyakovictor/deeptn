#!/usr/bin/env python3
"""
Full test runner - processes all images from real2 and silicone datasets.
Saves per-image JSON immediately, then builds combined CSV + summary tables.
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

RESULT_DIR = '/Users/victorkhudyakov/deeputin/eyetest/result'

METRIC_NAMES = [
    'aperture_shrinkage', 'eyelid_thickness', 'lid_smoothness',
    'eyelid_edge_sharpness', 'eye_symmetry_anomaly', 'sclera_iris_boundary',
    'pupil_apparent_size', 'orbit_area_ratio', 'periocular_lbp_entropy',
    'eyelid_sss', 'iris_visible_ratio', 'iris_center_offset',
    'sclera_asymmetry', 'landmark_discrepancy', 'specular_brdf',
    'eye_contour_divergence', 'subsurface_violation', 'ear_texture_cliff',
    'skin_tone_mismatch',
]

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


def build_row(fname, label, result):
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
    row['combined_median'] = result.get('combined_median', 0)
    row['combined_max'] = result.get('combined_max', 0)
    row['eye_mean'] = result.get('eye_mean', 0)
    row['cross_mean'] = result.get('cross_mean', 0)

    all_scores = {}
    all_scores.update(result.get('eye_scores', {}))
    all_scores.update(result.get('cross_scores', {}))
    if all_scores:
        vals = list(all_scores.values())
        zero_count = sum(1 for v in vals if v == 0.0)
        saturated = sum(1 for v in vals if v >= 0.99)
        non_zero = [v for v in vals if v > 0.0]
        high_anomaly = sum(1 for v in non_zero if v > 0.7) if non_zero else 0
        row['zero_count'] = zero_count
        row['saturated_count'] = saturated
        row['high_anomaly'] = high_anomaly
        row['metrics_active'] = len(vals) - zero_count
        row['natural_flag'] = 'NATURAL' if row['combined_mean'] < 0.5 and high_anomaly <= 3 else 'UNNATURAL'
    else:
        row['zero_count'] = 0
        row['saturated_count'] = 0
        row['high_anomaly'] = 0
        row['metrics_active'] = 0
        row['natural_flag'] = 'UNKNOWN'
    return row


def process_dataset(input_dir, label, args,
                    ddffa_analyzer, mp_analyzer, eye_detector, cross_detector,
                    pose_estimator, limit=None):
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if limit:
        files = files[:limit]
    total = len(files)

    per_image_dir = os.path.join(RESULT_DIR, 'per_image', label)
    os.makedirs(per_image_dir, exist_ok=True)

    all_rows = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"Processing {label}: {total} images")
    print(f"  Per-image JSONs -> {per_image_dir}/")
    print(f"{'='*60}")

    for i, fname in enumerate(files, 1):
        elapsed = time.time() - t_start
        avg = elapsed / (i - 1) if i > 1 else 0
        remaining = avg * (total - i) if i > 1 else 0
        print(f"  [{i}/{total}] {fname}  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)", flush=True)

        json_path = os.path.join(per_image_dir, fname.rsplit('.', 1)[0] + '.json')

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    saved = json.load(f)
                if saved.get('_complete'):
                    row = build_row(fname, label, saved)
                    all_rows.append(row)
                    score = saved.get('combined_mean', 0)
                    median = saved.get('combined_median', 0)
                    rakurs = saved.get('ракурс', '?')
                    flag = row.get('natural_flag', '?')
                    zeros = row.get('zero_count', 0)
                    active = row.get('metrics_active', 0)
                    print(f"       CACHED rakurs={rakurs} mean={score:.3f} median={median:.3f} active={active}/19 zeros={zeros} [{flag}]", flush=True)
                    continue
            except Exception:
                pass

        try:
            im_path = os.path.join(input_dir, fname)
            result = process_image(im_path, ddffa_analyzer, mp_analyzer,
                                   eye_detector, cross_detector, args,
                                   pose_estimator=pose_estimator)
            result['_complete'] = True
            result['_filename'] = fname
            result['_label'] = label

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            row = build_row(fname, label, result)
            all_rows.append(row)

            score = result.get('combined_mean', 0)
            median = result.get('combined_median', 0)
            rakurs = result.get('ракурс', '?')
            flag = row.get('natural_flag', '?')
            zeros = row.get('zero_count', 0)
            active = row.get('metrics_active', 0)
            print(f"       -> rakurs={rakurs} mean={score:.3f} median={median:.3f} active={active}/19 zeros={zeros} [{flag}]", flush=True)

        except Exception as e:
            print(f"  ERROR {fname}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            err_row = {'filename': fname, 'label': label, 'ракурс': 'error', 'error': str(e)}
            all_rows.append(err_row)

    elapsed = time.time() - t_start
    print(f"  DONE {label}: {total} images in {elapsed:.1f}s ({elapsed/total:.1f}s/img)")

    if all_rows:
        natural = sum(1 for r in all_rows if r.get('natural_flag') == 'NATURAL')
        unnatural = sum(1 for r in all_rows if r.get('natural_flag') == 'UNNATURAL')
        avg_zeros = np.mean([r.get('zero_count', 0) for r in all_rows])
        avg_active = np.mean([r.get('metrics_active', 0) for r in all_rows])
        print(f"  ANALYSIS: NATURAL={natural}/{total}, UNNATURAL={unnatural}/{total}")
        print(f"  METRICS: avg_active={avg_active:.1f}/19, avg_zeros={avg_zeros:.1f}")

    return pd.DataFrame(all_rows)


def build_combined_csv(real_df, silicone_df, result_dir):
    combined = pd.concat([real_df, silicone_df], ignore_index=True)
    path = os.path.join(result_dir, 'eyetest_combined.csv')
    combined.to_csv(path, index=False)
    print(f"  Combined CSV: {path}")
    return combined


def build_summary_tables(combined_df, result_dir):
    score_cols = [c for c in METRIC_NAMES if c in combined_df.columns]
    stat_cols = score_cols + ['combined_mean', 'combined_median', 'combined_max', 'eye_mean', 'cross_mean']

    summary_rows = []
    for label in ['real', 'silicone']:
        subset = combined_df[combined_df['label'] == label]
        row = {'label': label, 'n': len(subset)}
        for col in stat_cols:
            if col in subset.columns:
                vals = subset[col].dropna()
                row[f'{col}_mean'] = round(vals.mean(), 4) if len(vals) else None
                row[f'{col}_std'] = round(vals.std(), 4) if len(vals) else None
                row[f'{col}_median'] = round(vals.median(), 4) if len(vals) else None
        natural = (subset['natural_flag'] == 'NATURAL').sum() if 'natural_flag' in subset.columns else 0
        row['n_natural'] = int(natural)
        row['n_unnatural'] = int(len(subset) - natural)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    path = os.path.join(result_dir, 'summary_by_label.csv')
    summary_df.to_csv(path, index=False)
    print(f"  Summary by label: {path}")

    rakurs_rows = []
    for rakurs in combined_df['ракурс'].unique():
        if rakurs in ('error', 'unknown'):
            continue
        for label in ['real', 'silicone']:
            subset = combined_df[(combined_df['ракурс'] == rakurs) & (combined_df['label'] == label)]
            if len(subset) == 0:
                continue
            row = {'ракурс': rakurs, 'label': label, 'n': len(subset)}
            for col in ['combined_mean', 'combined_median']:
                if col in subset.columns:
                    vals = subset[col].dropna()
                    row[f'{col}_mean'] = round(vals.mean(), 4) if len(vals) else None
                    row[f'{col}_median'] = round(vals.median(), 4) if len(vals) else None
            rakurs_rows.append(row)

    rakurs_df = pd.DataFrame(rakurs_rows)
    path = os.path.join(result_dir, 'summary_by_rakurs.csv')
    rakurs_df.to_csv(path, index=False)
    print(f"  Summary by rakurs: {path}")

    metric_rows = []
    for col in METRIC_NAMES:
        if col not in combined_df.columns:
            continue
        real_vals = combined_df[combined_df['label'] == 'real'][col].dropna()
        sil_vals = combined_df[combined_df['label'] == 'silicone'][col].dropna()
        row = {
            'metric': col,
            'real_mean': round(real_vals.mean(), 4) if len(real_vals) else None,
            'real_median': round(real_vals.median(), 4) if len(real_vals) else None,
            'real_std': round(real_vals.std(), 4) if len(real_vals) else None,
            'silicone_mean': round(sil_vals.mean(), 4) if len(sil_vals) else None,
            'silicone_median': round(sil_vals.median(), 4) if len(sil_vals) else None,
            'silicone_std': round(sil_vals.std(), 4) if len(sil_vals) else None,
        }
        if row['real_mean'] is not None and row['silicone_mean'] is not None:
            diff = row['silicone_mean'] - row['real_mean']
            row['diff'] = round(diff, 4)
            row['diff_pct'] = round(diff / row['real_mean'] * 100, 1) if row['real_mean'] != 0 else None
        metric_rows.append(row)

    metric_df = pd.DataFrame(metric_rows)
    metric_df = metric_df.sort_values('diff', key=abs, ascending=False, na_position='last')
    path = os.path.join(result_dir, 'summary_by_metric.csv')
    metric_df.to_csv(path, index=False)
    print(f"  Summary by metric: {path}")


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
    os.makedirs(RESULT_DIR, exist_ok=True)

    t_total = time.time()

    real_df = process_dataset(real2_dir, 'real', args,
                              ddffa_analyzer, mp_analyzer, eye_detector, cross_detector,
                              pose_estimator, limit=args.limit)
    real_csv = os.path.join(RESULT_DIR, 'eyetest_real.csv')
    real_df.to_csv(real_csv, index=False)
    print(f"  CSV saved: {real_csv}")

    silicone_df = process_dataset(silicone_dir, 'silicone', args,
                                  ddffa_analyzer, mp_analyzer, eye_detector, cross_detector,
                                  pose_estimator, limit=args.limit)
    silicone_csv = os.path.join(RESULT_DIR, 'eyetest_silicone.csv')
    silicone_df.to_csv(silicone_csv, index=False)
    print(f"  CSV saved: {silicone_csv}")

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"BUILDING SUMMARY TABLES...")

    combined_df = build_combined_csv(real_df, silicone_df, RESULT_DIR)
    build_summary_tables(combined_df, RESULT_DIR)

    print(f"\n{'='*60}")
    print(f"DONE! Total time: {total_time:.1f}s")
    print(f"  Real:     {len(real_df)} images")
    print(f"  Silicone: {len(silicone_df)} images")
    print(f"  All files in: {RESULT_DIR}/")
    print(f"    per_image/real/     - per-image JSONs (real)")
    print(f"    per_image/silicone/ - per-image JSONs (silicone)")
    print(f"    eyetest_real.csv")
    print(f"    eyetest_silicone.csv")
    print(f"    eyetest_combined.csv")
    print(f"    summary_by_label.csv")
    print(f"    summary_by_rakurs.csv")
    print(f"    summary_by_metric.csv")


if __name__ == '__main__':
    main()

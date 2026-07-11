from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import TextureExtractor without importing project.s2_metrics package __init__
_tex_path = REPO_ROOT / "project" / "s2_metrics" / "modules" / "texture" / "texture_extractor.py"
_spec = importlib.util.spec_from_file_location("texture_extractor_module", _tex_path)
_tex_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_tex_module)
TextureExtractor = _tex_module.TextureExtractor


class Ctx:
    def __init__(self, face_mask_path: Path):
        self.face_mask_path = str(face_mask_path)
        img = cv2.imread(str(face_mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(f"Cannot read RGBA face mask: {face_mask_path}")
        self.image_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        ys, xs = np.where(img[:, :, 3] > 10)
        if len(xs) > 0:
            self.face_bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            self.face_min_dim = min(self.face_bbox[2], self.face_bbox[3])
        else:
            self.face_bbox = []
            self.face_min_dim = None
        self.pp_iod = None


def iter_masks(data_root: Path):
    for label, folder in [("real", "output_real30"), ("silicone", "output_silicone_30")]:
        root = data_root / folder
        for p in sorted(root.glob("*/face_mask.png")):
            yield label, p.parent.name, p


def clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = float(v)
        elif isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            out[k] = float(v)
    return out


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    y = (df["label"] == "silicone").astype(int).values
    rows = []
    for c in df.columns:
        if c in {"label", "photo_id", "path"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        vals = df[c].astype(float).replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() != len(vals) or vals.nunique(dropna=True) <= 1:
            continue
        auc = roc_auc_score(y, vals.values)
        r = df[df.label == "real"][c].astype(float)
        s = df[df.label == "silicone"][c].astype(float)
        direction = "higher_in_silicone" if auc >= 0.5 else "lower_in_silicone"
        rows.append({
            "metric": c,
            "auc": float(auc),
            "auc_abs": float(max(auc, 1.0 - auc)),
            "direction": direction,
            "real_mean": float(r.mean()),
            "silicone_mean": float(s.mean()),
            "real_median": float(r.median()),
            "silicone_median": float(s.median()),
            "real_std": float(r.std()),
            "silicone_std": float(s.std()),
        })
    return pd.DataFrame(rows).sort_values("auc_abs", ascending=False)


def best_thresholds(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    y = (df["label"] == "silicone").astype(int).values
    rows = []
    for _, row in table.iterrows():
        c = row["metric"]
        vals = df[c].astype(float).values
        uniq = np.unique(vals)
        if len(uniq) < 2:
            continue
        candidates = (uniq[:-1] + uniq[1:]) / 2.0
        best = None
        for t in candidates:
            if row["direction"] == "higher_in_silicone":
                pred = vals >= t
                rule = f">= {t:.6g}"
            else:
                pred = vals <= t
                rule = f"<= {t:.6g}"
            acc = float((pred.astype(int) == y).mean())
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            item = (acc, tp + tn, -fp, -fn, rule, tp, tn, fp, fn)
            if best is None or item > best:
                best = item
        if best:
            acc, _, _, _, rule, tp, tn, fp, fn = best
            rows.append({
                "metric": c,
                "rule": rule,
                "accuracy": acc,
                "tp_silicone": tp,
                "tn_real": tn,
                "fp_real_as_silicone": fp,
                "fn_silicone_as_real": fn,
            })
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, table: pd.DataFrame, thresholds: pd.DataFrame, out_md: Path) -> None:
    top = table.head(20).copy()
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)

    lines: List[str] = []
    lines.append("# Аудит метрик, пересчитанных напрямую из `face_mask.png`\n")
    lines.append("Пересчёт выполнен текущим `TextureExtractor` из `project/s2_metrics/modules/texture/texture_extractor.py`.\n")
    lines.append(f"Всего файлов: `{len(df)}`; real: `{(df.label == 'real').sum()}`; silicone: `{(df.label == 'silicone').sum()}`.\n")
    lines.append("Важно: этот пересчёт использует только `face_mask.png`. Landmark-dependent physical features из `physical_features.py` (`seam_score`, `sss_index`, `specular_sharpness`) здесь не пересчитываются, если нет `reconstruction.pkl`.\n")

    lines.append("## Точный набор числовых метрик, доступных из `face_mask.png`\n")
    metric_cols = [c for c in df.columns if c not in {"label", "photo_id", "path"} and pd.api.types.is_numeric_dtype(df[c])]
    lines.append("\n".join(f"- `{c}`" for c in metric_cols))
    lines.append("\n")

    lines.append("## Самые разделяющие метрики\n")
    lines.append("| # | metric | AUC abs | направление silicone | real median | silicone median | real mean | silicone mean |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(top.itertuples(index=False), 1):
        lines.append(
            f"| {i} | `{r.metric}` | {r.auc_abs:.3f} | {r.direction} | {fmt(r.real_median)} | {fmt(r.silicone_median)} | {fmt(r.real_mean)} | {fmt(r.silicone_mean)} |"
        )

    lines.append("\n## Лучшие одиночные пороги по текущей выборке\n")
    lines.append("Не использовать как финальный verdict в одиночку; это диагностические пороги для понимания направления метрик.\n")
    lines.append("| metric | rule for silicone | accuracy | FP real→silicone | FN silicone→real |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in thresholds.head(15).itertuples(index=False):
        lines.append(f"| `{r.metric}` | `{r.rule}` | {r.accuracy:.3f} | {r.fp_real_as_silicone} | {r.fn_silicone_as_real} |")

    lines.append("\n## Вывод\n")
    lines.append("На текущих `face_mask.png` наиболее устойчивый rule-based каркас дают семейства: FFT micro-frequency, spectral slope, pore/blob density, WLD entropy, GLRLM/GLSZM и GLCM anisotropy. Вердикт лучше строить не по одному порогу, а по сумме вкладов нескольких независимых семейств.\n")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=REPO_ROOT / "project" / "data", type=Path)
    ap.add_argument("--out-csv", default=REPO_ROOT / "face_mask_recomputed_texture_metrics.csv", type=Path)
    ap.add_argument("--out-md", default=REPO_ROOT / "FACE_MASK_METRICS_AUDIT.md", type=Path)
    args = ap.parse_args()

    extractor = TextureExtractor()
    rows = []
    for label, photo_id, p in iter_masks(args.data_root):
        ctx = Ctx(p)
        metrics = clean_metrics(extractor.extract(ctx, exclude_sensitive=False))
        metrics.update({"label": label, "photo_id": photo_id, "path": str(p.relative_to(REPO_ROOT))})
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    table = auc_table(df)
    table.to_csv(args.out_csv.with_name(args.out_csv.stem + "_auc.csv"), index=False)
    thresholds = best_thresholds(df, table)
    thresholds.to_csv(args.out_csv.with_name(args.out_csv.stem + "_thresholds.csv"), index=False)
    write_report(df, table, thresholds, args.out_md)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_csv.with_name(args.out_csv.stem + '_auc.csv')}")
    print(f"Wrote: {args.out_csv.with_name(args.out_csv.stem + '_thresholds.csv')}")
    print(f"Wrote: {args.out_md}")
    print(table.head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

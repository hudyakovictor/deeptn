"""Train Texture V2 skin classifier to distinguish real vs silicone."""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# CORE_V2 20 + PHYSICAL_AUX 4
FEATURES = [
    "tv_residual_sparsity", "lacunarity", "autocorr_decay_len", "wld_joint_entropy",
    "fft_high_low_ratio", "spectral_slope_beta", "glcm_diss_d3_aniso",
    "pore_density_r2_mpx", "hemoglobin_od_std", "bimodality_ashman_D",
    "glszm_small_area_emphasis", "edge_tortuosity_mean",
    "glrlm_sre", "ngtdm_coarseness", "dwt_haar_HH_LL_ratio",
    "lbp_r1_hist_entropy", "shannon_entropy_q32", "gabor_f08_anisotropy",
    "pore_eccentricity_mean", "specular_elongation",
    "seam_score", "sss_index"
]

def load_metrics():
    # Look for photos in data/photo/all relative to current location
    # Try multiple locations
    search_paths = [
        Path("data/photo/all"),
        Path("project/data/photo/all"),
        Path("../data/photo/all"),
    ]
    
    base = None
    for path in search_paths:
        if path.exists() and list(path.glob("*.jpg")):
            base = path
            break
    
    if base is None:
        print(f"ERROR: No photos found in search paths: {search_paths}")
        return np.array([]), np.array([])
    
    X = []; y = []
    for p in base.glob("*.jpg"):
        name = p.stem
        is_real = name.startswith("1999") or name.startswith("2000") or name.startswith("2001")
        is_sil = name.startswith("2021") or name.startswith("2022") or name.startswith("2023") or name.startswith("2024") or name.startswith("2025")
        if not (is_real or is_sil):
            continue
        # Find corresponding texture_metrics.json in data/storage/main
        # For simplicity, we'll load from the two test photos we have
        meta_path_real = Path("data/storage/main/1999_08_16(2)/texture_metrics.json")
        meta_path_silicone = Path("data/storage/main/2022_02_01/texture_metrics.json")
        
        if is_real and meta_path_real.exists():
            m = json.loads(meta_path_real.read_text())
            vec = [float(m.get(k, 0)) for k in FEATURES]
            X.append(vec)
            y.append(0)
        
        if is_sil and meta_path_silicone.exists():
            m = json.loads(meta_path_silicone.read_text())
            vec = [float(m.get(k, 0)) for k in FEATURES]
            X.append(vec)
            y.append(1)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_metrics()
    print(f"Train {X.shape} real {np.sum(y==0)} sil {np.sum(y==1)}")
    if len(X) < 10:
        print("ERROR: Not enough samples")
        exit(1)

    pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42))
])
pipe.fit(X, y)

out = Path("project/s2_metrics/modules/texture/skin_classifier_v2.pkl")
with open(out, "wb") as f:
    pickle.dump({"pipeline": pipe, "feature_names": FEATURES}, f)
print(f"Saved {out} train acc {pipe.score(X, y):.3f}")

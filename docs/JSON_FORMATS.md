# DEEPUTIN JSON Output Formats

## Overview

Each processed photo generates a directory with standardized JSON files:

```
storage/main/2020_01_15/
├── info.json              # Stage 1: metadata, pose, quality
├── texture_metrics.json   # Stage 2: texture features
├── geometry_metrics.json  # Stage 2: geometry features
├── face_mask.png          # RGBA face mask (424x500)
├── face_crop.jpg          # BGR face preview
├── thumb.jpg              # 100x100 thumbnail
├── reconstruction.pkl     # 3DDFA reconstruction (binary)
├── uv_texture.png         # UV texture map (optional)
├── uv_confidence.png      # UV confidence map (optional)
├── mesh.obj               # 3D mesh (optional)
└── mesh.mtl               # Material file (optional)
```

---

## info.json

**Stage**: S1 Extraction  
**Purpose**: Core metadata for each photo

### Structure

```json
{
  "photo_id": "2020_01_15",
  "dataset": "main",
  "source_path": "/path/to/original.jpg",
  "date": "2020-01-15T00:00:00",
  "age_years": 67.5,
  
  "pose": {
    "photo_id": "2020_01_15",
    "date": "2020-01-15T00:00:00",
    "age_years": 67.5,
    "yaw": 12.3,
    "pitch": -5.2,
    "roll": 2.1,
    "bucket": "frontal",
    "pose_source": "3ddfa_v3",
    "confidence": 0.8
  },
  
  "quality": {
    "overall_quality": 0.75,
    "blur_value": 45.2,
    "noise_level": 12.3,
    "jpeg_blockiness": 0.8,
    "sharpness_score": 234.5
  },
  
  "face_bbox": [120, 80, 300, 400],
  "face_mask_path": "/path/to/face_mask.png",
  "reconstruction_path": "/path/to/reconstruction.pkl",
  "image_size": [1920, 1080],
  
  "expression_flags": {
    "smile_excluded": false,
    "jaw_excluded": false,
    "neutralized": false
  },
  
  "readiness": {
    "geometry": "available",
    "texture": "available"
  }
}
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `photo_id` | string | Unique identifier (from filename) |
| `dataset` | string | "main" or "calibration" |
| `date` | ISO datetime | Parsed from filename |
| `age_years` | float | Subject age at photo date |
| `pose.bucket` | string | Pose category (see below) |
| `quality.overall_quality` | float | 0.0-1.0 composite quality score |
| `face_bbox` | [x, y, w, h] | Face bounding box in original image |

### Pose Buckets

- `frontal`: |yaw| < 15°
- `left_threequarter_light`: -30° ≤ yaw < -15°
- `right_threequarter_light`: 15° ≤ yaw < 30°
- `left_threequarter_medium`: -45° ≤ yaw < -30°
- `right_threequarter_medium`: 30° ≤ yaw < 45°
- `left_threequarter_deep`: -60° ≤ yaw < -45°
- `right_threequarter_deep`: 45° ≤ yaw < 60°
- `left_profile`: yaw < -60°
- `right_profile`: yaw > 60°
- `unknown`: pose estimation failed

---

## texture_metrics.json

**Stage**: S2 Metrics  
**Purpose**: Texture features for skin analysis

### Structure

```json
{
  "tv_residual_sparsity": 0.8234,
  "edge_tortuosity_mean": 1.0423,
  "autocorr_decay_len": 23.5,
  "specular_elongation": 2.1,
  "glcm_diss_d3_aniso": 0.0856,
  "lacunarity": 2.34,
  "spectral_slope_beta": 2.78,
  "glrlm_sre": 0.4521,
  "glszm_small_area_emphasis": 0.5634,
  "fft_high_low_ratio": 0.1234,
  "wld_joint_entropy": 6.234,
  "lbp_r1_hist_entropy": 3.456,
  "pore_density_r2_mpx": 28500.0,
  "hemoglobin_od_std": 0.0823,
  "bimodality_ashman_D": 3.21,
  "pore_eccentricity_mean": 0.856,
  "ngtdm_coarseness": 0.0234,
  "seam_score": 0.0123,
  "melanin_hemo_slope": 4.56,
  "shannon_entropy_q32": 4.89,
  
  "overall_quality": 0.75,
  "sharpness_score": 234.5,
  "noise_level": 12.3,
  "jpeg_blockiness": 0.8,
  
  "texture_assessability": "eligible",
  "texture_unreliable": false,
  "texture_skin_hint": "real",
  "texture_skin_confidence": 0.87,
  "texture_real_prob": 0.87,
  "texture_silicone_prob": 0.13
}
```

### Metric Categories

#### Tier 1 Core Metrics (12)
Primary texture features used for classification:

1. **tv_residual_sparsity** (0.5-0.95): Total variation residual sparsity
   - Lower = smoother/more uniform (silicone indicator)
   
2. **edge_tortuosity_mean** (0.8-1.3): Edge complexity
   - Lower = straighter edges (silicone indicator)
   
3. **autocorr_decay_len** (5.0-50.0): Autocorrelation decay
   - Higher = more uniform texture (silicone indicator)
   
4. **specular_elongation** (0.5-30.0): Specular highlight shape
   - Higher = elongated highlights (silicone indicator)
   
5. **glcm_diss_d3_aniso** (0.0-0.5): GLCM dissimilarity anisotropy
6. **lacunarity** (1.0-6.0): Texture gap distribution
7. **spectral_slope_beta** (1.0-5.0): Power spectrum slope
8. **glrlm_sre** (0.1-0.8): Gray-level run length short run emphasis
9. **glszm_small_area_emphasis** (0.2-0.8): Gray-level size zone small area emphasis
10. **fft_high_low_ratio** (0.0-2.0): FFT high/low frequency ratio
11. **wld_joint_entropy** (4.0-7.5): Weber local descriptor joint entropy
12. **lbp_r1_hist_entropy** (2.0-4.5): Local binary pattern histogram entropy

#### Tier 2 Metrics (8)
Secondary features (quality-dependent):

13. **pore_density_r2_mpx** (5000-60000): Pore density per megapixel
14. **hemoglobin_od_std** (0.0-0.5): Hemoglobin optical density std
15. **bimodality_ashman_D** (1.0-6.0): Bimodality coefficient
16. **pore_eccentricity_mean** (0.5-1.0): Pore shape eccentricity
17. **ngtdm_coarseness** (0.0-1.0): Neighborhood gray tone difference matrix coarseness
18. **seam_score** (0.0-0.5): Seam detection score
19. **melanin_hemo_slope** (-2.0-8.0): Melanin-hemoglobin slope
20. **shannon_entropy_q32** (3.0-6.0): Shannon entropy (32 bins)

#### Quality Metrics
- **overall_quality** (0.0-1.0): Composite quality score
- **sharpness_score** (0.0-10000.0): Laplacian variance
- **noise_level** (0.0-15.0): Noise estimate
- **jpeg_blockiness** (0.5-5.0): JPEG compression artifacts

#### Classification Results
- **texture_skin_hint**: "real" | "silicone" | "unknown"
- **texture_skin_confidence** (0.0-1.0): Classification confidence
- **texture_real_prob** (0.0-1.0): Probability of real skin
- **texture_silicone_prob** (0.0-1.0): Probability of silicone

### Clean Output

The `clean_texture_metrics()` function removes:
- Internal debug fields (`texture_feature_weights_json`)
- Duplicate fields (`texture_noise_sigma`)
- Always-zero fields (`specular_dispersion`, `specular_sharpness`)
- Tier2 weight indicators (`*_weight`)
- Internal fields (starting with `_`)

---

## geometry_metrics.json

**Stage**: S2 Metrics  
**Purpose**: 3D geometry features from 3DDFA reconstruction

### Structure

```json
{
  "zone_nose_bridge_tip_depth_std_ratio": 0.0234,
  "zone_cheekbone_L_span_lateral_ratio": 0.1456,
  "zone_cheekbone_R_span_lateral_ratio": 0.1423,
  "zone_chin_depth_std_ratio": 0.0189,
  "zone_forehead_span_vertical_ratio": 0.2345,
  "zone_jaw_L_span_lateral_ratio": 0.0987,
  "zone_jaw_R_span_lateral_ratio": 0.0956,
  "zone_orbit_L_bbox_area_ratio": 0.0123,
  "zone_orbit_R_bbox_area_ratio": 0.0119,
  
  "bone_nasion_depth": 0.456,
  "bone_zygomatic_width": 0.234,
  "bone_asymmetry_x": 0.012,
  
  "jaw_taper_index": 1.234,
  "bigonial_to_bizygomatic_ratio": 0.856
}
```

### Clean Output

The `clean_geometry_metrics()` function removes:
- NaN/Inf values
- None values
- Internal fields (starting with `_`)

**Note**: Many geometry metrics may be missing if the geometry extractor is disabled or fails. This is normal — texture metrics are the primary classification features.

---

## Validation

All JSON files are validated using soft checks:

### texture_metrics.json
- Checks for required metrics (tv, et, ac)
- Validates value ranges
- Reports NaN/Inf values
- Logs out-of-range values as warnings

### geometry_metrics.json
- Reports percentage of missing metrics
- Validates numeric types
- Logs if >50% metrics are missing

### info.json
- Validates required fields
- Checks pose bucket values
- Validates angle ranges
- Checks quality metric ranges

### Example Validation Output

```
[2020_01_15] tv_residual_sparsity: Outside expected range [0.5, 0.95] (got 0.4500)
[2020_01_15] pose.yaw: Angle > 180° (got 185.3°)
[2020_01_15] geometry: 67% of geometry metrics are NaN/None (234/350)
```

---

## Reading JSON Files

### Python

```python
import json
from pathlib import Path

photo_dir = Path("storage/main/2020_01_15")

# Load texture metrics
with open(photo_dir / "texture_metrics.json") as f:
    texture = json.load(f)

print(f"Skin hint: {texture['texture_skin_hint']}")
print(f"Confidence: {texture['texture_skin_confidence']:.2%}")
print(f"TV sparsity: {texture['tv_residual_sparsity']:.4f}")
```

### Command Line

```bash
# Pretty-print texture metrics
cat storage/main/2020_01_15/texture_metrics.json | python -m json.tool

# Extract specific field
jq '.texture_skin_hint' storage/main/2020_01_15/texture_metrics.json

# Find all silicone classifications
find storage/main -name "texture_metrics.json" -exec jq -l '.texture_skin_hint == "silicone"' {} \;
```

---

## Schema Versioning

Current schema version: **1.0** (no explicit version field)

Future versions should add:
```json
{
  "_schema_version": "1.1",
  "_pipeline_version": "0.5.0",
  "_generated_at": "2024-01-15T14:30:00Z"
}
```

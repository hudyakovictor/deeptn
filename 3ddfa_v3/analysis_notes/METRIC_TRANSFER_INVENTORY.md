# Metric transfer inventory (ITER10)

Date: 2026-07-14

## Geometry — newapp `extract_macro_bone_metrics` (scoring.py)

- Referenced keys in scoring.py: **48**
- Present in lib extract output: **43/48**
- Finite values on synthetic mesh: **61**

| metric | in lib |
|--------|--------|
| `bigonial_width_ratio` | YES |
| `bizygomatic_depth_asymmetry` | YES |
| `brow_asymmetry_deg` | YES |
| `canthal_tilt_3d_L` | YES |
| `canthal_tilt_3d_R` | YES |
| `canthal_tilt_L` | YES |
| `canthal_tilt_R` | YES |
| `canthal_tilt_asymmetry_deg` | YES |
| `canthal_tilt_mean_deg` | YES |
| `chin_offset_asymmetry` | YES |
| `chin_projection_ratio` | YES |
| `expression_severity` | YES |
| `eye_aspect_ratio_L` | NO |
| `eye_aspect_ratio_R` | NO |
| `forehead_slope_index` | YES |
| `glabella_nasion_projection_angle` | YES |
| `gnathion_midline_deviation_ratio` | YES |
| `gonial_angle_L` | YES |
| `gonial_angle_R` | YES |
| `gonial_width_asymmetry` | YES |
| `intercanthal_width_ratio` | YES |
| `interorbital_ratio` | YES |
| `lip_thickness_ratio` | NO |
| `lower_lip_height_ratio` | NO |
| `mandibular_ramus_length` | YES |
| `midface_compactness` | YES |
| `midface_depth_index` | YES |
| `nasal_frontal_index` | YES |
| `nasal_length_ratio` | YES |
| `nasion_zone_depth_ratio` | YES |
| `nasofacial_angle_ratio` | YES |
| `nose_bridge_length_ratio` | YES |
| `nose_projection_ratio` | YES |
| `nose_width_ratio` | YES |
| `orbit_centroid_ratio` | YES |
| `orbit_depth_L_ratio` | YES |
| `orbit_depth_R_ratio` | YES |
| `orbit_depth_asymmetry_ratio` | YES |
| `orbit_skull_ratio` | YES |
| `orbit_vertical_asymmetry_ratio` | YES |
| `orbit_vertical_signed_ratio` | YES |
| `orbital_asymmetry_index` | YES |
| `orbital_height_signed` | YES |
| `orbital_perimeter_symmetry` | YES |
| `palpebral_fissure_asymmetry_ratio` | YES |
| `skull_depth_asymmetry_index` | YES |
| `subnasale_projection_ratio` | YES |
| `upper_lip_height_ratio` | NO |

### Extra keys from lib (20)

- `brow_ridge_projection_L_ratio`
- `brow_ridge_projection_R_ratio`
- `cranial_face_index`
- `jaw_width_ratio`
- `ligament_orbital_L_depth_ratio`
- `ligament_orbital_R_depth_ratio`
- `mandibular_body_length_L_ratio`
- `mandibular_body_length_R_ratio`
- `orbit_fossa_spread_L`
- `orbit_fossa_spread_R`
- `orbit_width_L_ratio`
- `orbit_width_R_ratio`
- `palpebral_fissure_length_L_ratio`
- `palpebral_fissure_length_R_ratio`
- `ramus_height_L_ratio`
- `ramus_height_R_ratio`
- `temporal_depth_L_ratio`
- `temporal_depth_R_ratio`
- `zygomatic_arch_height_L_ratio`
- `zygomatic_arch_height_R_ratio`

## Geometry — project GEOMETRY_CORE_METRICS catalog

**NOT ported:** 234 zone-shape metrics (convexity, bbox volume, plane residuals, L_/R_ proxies).

## Texture — newapp TextureMetrics fields

| field | in lib |
|-------|--------|
| `lbp_uniformity` | YES |
| `lbp_entropy` | YES |
| `glcm_contrast` | YES |
| `glcm_energy` | YES |
| `glcm_homogeneity` | YES |
| `glcm_correlation` | YES |
| `gabor_mean` | YES |
| `gabor_std` | YES |
| `laplacian_energy` | YES |
| `specular_gloss` | YES |
| `pigmentation_index` | YES |
| `glcm_contrast_ratio` | YES |
| `skin_micro_contrast` | YES |
| `skin_shannon_entropy` | YES |
| `quality_sharpness_score` | YES |
| `quality_noise_score` | YES |
| `quality_index` | YES |

### Lib-only extras

- `fractal_dimension`
- `autocorrelation_decay`
- `pore_proxy`
- `matte_uniformity`
- `synthetic_prob`
- `raw_synthetic_prob`
- `quality_adjusted_synthetic_prob`
- `natural_score`

## Systems still NOT ported

| component | status |
|-----------|--------|
| util.geometry_metrics (scoring bone ratios) | YES |
| util.zones zone errors | YES |
| util.texture field-parity + heuristic score | YES (expanded) |
| skin_authenticity full blocks + reference classifiers | NO |
| periocular_metrics full suite | NO |
| GEOMETRY_CORE_METRICS 234 catalog | NO |

## Summary

- Bone ratios (scoring): **43/48**
- Zone-shape catalog: **0/234**
- TextureMetrics fields: **17/17**

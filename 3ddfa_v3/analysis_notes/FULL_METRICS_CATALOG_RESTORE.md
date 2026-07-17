# Full metrics catalog restore

Date: 2026-07-14

## Truth from old project (`legacy_metrics/metric_coverage_report.json`)

- **Full catalog / registry: 2199 metrics**
- Chronology metrics: 19
- Stability-pruned allowed: **955**
- Stability-pruned excluded: **268**
- Names recovered without original CSV: **1251**
- Names still missing file list for (need metric_catalog_active.csv original): **948**

### Single-photo metrics by view (old system)

| bucket | metrics |
|--------|--------:|
| frontal | 1814 |
| left_threequarter_light | 1187 |
| right_threequarter_light | 1187 |
| left_threequarter_mid | 1025 |
| right_threequarter_mid | 1025 |
| left_threequarter_deep | 1025 |
| right_threequarter_deep | 1025 |
| left_profile | 1025 |
| right_profile | 1025 |

### Pair metrics by view

| bucket | metrics |
|--------|--------:|
| frontal | 360 |
| left_threequarter_light | 210 |
| right_threequarter_light | 210 |
| left_threequarter_mid | 210 |
| right_threequarter_mid | 210 |
| left_threequarter_deep | 210 |
| right_threequarter_deep | 210 |
| left_profile | 210 |
| right_profile | 210 |

### By family

| family | count |
|--------|------:|
| F_zone | 483 |
| F_pair_zone | 230 |
| F3 | 216 |
| F6 | 156 |
| F12 | 135 |
| F9 | 130 |
| F11 | 112 |
| F4 | 108 |
| F1 | 104 |
| F7 | 104 |
| F0 | 100 |
| F5 | 59 |
| F_orbit | 46 |
| F10 | 45 |
| F_nose | 31 |
| F_periocular | 26 |
| F_zyg_temp | 24 |
| F2 | 23 |
| F_mandible | 21 |
| F13 | 19 |
| F_brow | 16 |
| F8 | 11 |

## What is in lib NOW

| layer | count |
|-------|------:|
| util.geometry_metrics (bone ratios) | ~60 |
| util.texture TextureMetrics fields | 21 |
| util.zones zone sets | 25 |
| GEOMETRY_CORE_METRICS subset list | 234 |
| restored name catalog (partial CSV) | 1251 |
| **target full catalog** | **2199** |

## Distribution on 9 views

Old system: YES — `single_by_bucket` / `pair_by_bucket` + `identity_scoring_config` core/experimental/disabled per view.

Lib before this restore: pose classification only, no metric whitelist per view.

Lib after this step: catalog + per-view counts API restored from artifacts; full compute of all 2199 still requires original CSV metric definitions OR re-export from project.

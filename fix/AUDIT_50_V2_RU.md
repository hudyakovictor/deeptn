# DEEPUTIN — Аудит 2: 50+ ключевых находок после первой итерации исправлений

Дата: 2026-07-12  
Репозиторий: `hudyakovictor/deeptn`, `master`  
Область: S1–S5, реконструкция, geometry, texture, compare, verdict, calibration, chronology, report.

## 0. Резюме новой итерации

После предыдущего аудита (`AUDIT_50_RU.md`) часть дефектов была исправлена, но в коде остались:

1. **Эра-prior всё ещё встроен в classifier** — главный методологический дефект, не позволяющий считать результаты объективными.
2. **`pose_yaw/pitch/roll` загрязняют `geometry_metrics`** и через `_normalized_distance` ломают попарные расстояния в S3.
3. **`id_params` / `exp_params` сохраняются в reconstruction-dict как top-level, но `GeometryExtractor` ищет их в `recon["payload"]`** — параметры экспрессии и identity-вектор **никогда не попадают** в `geometry_metrics.json` (`id_norm`, `id_mean`, `id_std`, `exp_magnitude` пусты).
4. **ChronologyAnalyzer смешивает все bucket-ы в один timeline** — флаги `IMPOSSIBLE_BONE_CHANGE`, `age_inversion`, `spike:*` возникают между соседними фото разных ракурсов одного человека.
5. **Platt scaling не обучен** (`fit()` не вызывается), а `y_true` для `calibration_report.json` строится из самих предсказаний — отчёт ECE/MCE невалиден.
6. **Pose-aware калибровка (S2) строится, но не используется в S3** — `pose_calibration_models.json` пишется, но `CompareEngine` его не читает.
7. **`physical_extractor.extract()` получает landmarks в world/model space, но ROI строит в image space** (224x224) — все физические метрики (SSS, specular, pores) считаются **в чужой coordinate system** и врут.
8. **Калибровочный каталог и active extractor используют разные неймспейсы метрик** — overlap = 0; resolver возвращает `UNCERTAIN` по умолчанию.
9. **`skin_center` / `vertices_canonical = vertices_raw` исправлены**, но сохраняемый в `reconstruction.pkl` ключ `vertices_canonical` всё ещё равен `vertices_camera` (engine.py:706-712) — S3 читает неправильный mesh.
10. **`_normalized_distance` путает geometry и texture**: `if key.startswith("texture_")` берёт weight=0.85, но также не исключает `pose_yaw/pitch/roll`, `mesh_vertex_count`, `face_scale` из identity-вычислений.

Подробный список ниже.

---

# Блок A. S1 — extraction и metrics (10 находок)

### A1. КРИТИЧНО: `id_params` / `exp_params` ищутся в неправильном месте

**Код:**
- `project/s1_extraction/engine.py:744-745` — `id_params` / `exp_params` сохраняются как **top-level** в reconstruction dict.
- `project/s1_extraction/metrics/modules/geometry_extractor.py:384, 392` — `reconstruction.get("payload", {}).get("exp_params", [])` ищет **в `payload`**.

**Следствие:** `exp_magnitude`, `exp_jaw_open`, `exp_smile`, `id_norm`, `id_mean`, `id_std` **никогда не заполняются** в `geometry_metrics.json`.

**Sanity test:**
```python
test_dict = {"id_params": [0.1]*10, "payload": {"id_params": [0.2]*10}}
test_dict.get("payload", {}).get("id_params")  # → [0.2]*10 (НЕ [0.1]*10)
```

**Исправление:** либо сохранять `payload: {id_params, exp_params}` в `_reconstruction_to_dict`, либо в `GeometryExtractor` читать top-level `reconstruction.get("id_params", [])`.

### A2. КРИТИЧНО: `pose_yaw / pose_pitch / pose_roll` загрязняют `geometry_metrics`

**Код:** `project/s1_extraction/metrics/modules/geometry_extractor.py:401-403`

`pose_yaw` записывается как **degrees** (например 22.5). В S3 `_normalized_distance` (`project/s3_compare/engine.py:467-498`) эти поля попадают в общий перечень с `key_weight = 1.0`. 

**Sanity test:**
- Два фото, отличающихся по yaw на 0.5°: `weighted.append(0.5 * 1.0) = 0.5`.
- Два фото, отличающихся по `bone_nasion_depth` на 0.01 (типично): `weighted.append(0.0115)` (с weight 1.15).
- **Pose-шум превышает bone-сигнал в ~43 раза**.

**Исправление:** завести allow-list: `identity_metrics` / `pose_nuisance` / `quality_nuisance` / `raw_texture`, и в S3 использовать только identity для geometry distance.

### A3. `face_scale` (= x_span) попадает в geometry и доминирует в distance

**Код:** `project/s1_extraction/metrics/modules/geometry_extractor.py:148, 197`

`face_scale = max(x_span, 1.0)` сохраняется как `metrics["face_scale"]`. Любая пара фото, снятых с разных расстояний, даёт **ненулевую разность** `face_scale`. В S3 это попадает в geometry distance с `key_weight = 1.0`. В cohort фотографий разных лет и с разных камер face_scale реально меняется.

**Исправление:** исключить `face_scale` из identity-сравнения; использовать только как нормализатор внутри pipeline.

### A4. `mesh_vertex_count` — служебная метрика попадает в identity distance

**Код:** `project/s1_extraction/metrics/modules/geometry_extractor.py:189`

`mesh_vertex_count` — техническое поле, одинаковое для всех фото (фиксированная 3DMM), но **попадает в comparison**. Если 3DDFA вернёт другое число vertices (другая модель), оно сломает расстояния.

**Исправление:** исключить из active output.

### A5. `bone_asymmetry_x` использует naive reflection — некорректное соответствие

**Код:** `project/s1_extraction/metrics/modules/geometry_extractor.py:225-270` (`compute_asymmetry`)

Алгоритм отражает весь mesh по YZ, потом Procrustes, и считает residual. Это **не то же самое**, что correspondence "vertex i ↔ vertex j (mirror)". Если vertices не симметрично нумерованы, residual теряет смысл.

**Исправление:** нужна явная mid-sagittal plane estimation и попарное соответствие (см. `project/s1_extraction/metrics/modules/geometry/legacy_metrics/mid_sagittal.py`).

### A6. PCA normal имеет произвольный знак

**Код:** `project/s1_extraction/metrics/modules/geometry_extractor.py:90-110`

`vh[-1]` из SVD даёт normal `n` или `-n` в зависимости от нумерации. Без ориентации относительно camera/outward reference, `normal_mean_x/y/z` может инвертироваться между фото одного человека.

**Исправление:** align normal с `view_dir` или с `centroid_to_face_center`.

### A7. `text` encoding проблема в коде S4 chronology

**Код:** `project/s4_verdict/modules/chronology.py:325`

```python
g_was偏离 = np.mean(np.abs(g_norm[idx-2:idx])) > 1.0
```

Используются **китайские иероглифы** в имени переменной (`偏离` = "отклонение"). Python это допускает, но:
- некоторые IDE и linters не распознают;
- смешивает кодировки в журналах расследования (русский + китайский);
- нарушает читаемость кода для будущих ревьюеров.

**Исправление:** заменить на `g_was_deviated` / `g_was_off`.

### A8. Удаление валидных метрик в `clean_texture_metrics`

**Код:** `project/shared/validation.py:354-360`

`TEXTURE_NOISE_FIELDS` помечает `specular_dispersion` и `specular_sharpness` как "Always 0.0" — что **неверно**. `PhysicalTextureExtractor._compute_specular_sharpness_dispersion` возвращает реальные значения, и они важны для H1-детектора силикона. Эти метрики **вычисляются, но удаляются** перед сохранением → H1 не видит evidence.

**Исправление:** убрать specular_sharpness/dispersion из `TEXTURE_NOISE_FIELDS`.

### A9. `physical_extractor` получает landmarks в чужой coordinate system

**Код:**
- `project/s1_extraction/engine.py:241-250` — `landmarks_68 = reconstruction.get("landmarks_68")` или `landmarks_106`. В `reconstruction.py:283-284` `landmarks_106 = np.asarray(result["ldm106"])[0]` — это **2D координаты на исходном изображении**, не на face_crop 424×500.
- `project/s1_extraction/metrics/physical_features.py:80-89` — `image` это RGBA crop 424×500, но `landmarks` берутся в исходных координатах (тысячи пикселей).

**Следствие:** ROI для `ear`, `cheek`, `forehead`, `nose`, `jaw` строятся в **чужом** пространстве → попадают в случайные области face crop → **SSS, specular, pores, seam_score** — недостоверны.

**Исправление:** либо transform landmarks в face_crop space через `trans_params`, либо работать с исходным изображением.

### A10. `vertices_canonical` в reconstruction.pkl = `vertices_camera`, а не canonical

**Код:** `project/s1_extraction/engine.py:706-712` (если там есть), или reconstruction_adapter

Из предыдущего аудита (A9): `vertices_canonical` сохраняется в `reconstruction.pkl` под неверным именем. Если S3 читает `verts = recon.get("vertices_canonical", recon.get("vertices", []))` и получает `vertices_camera` (в camera space, не в canonical pose) — Procrustes alignment сравнивает разные coord systems.

**Проверить и исправить:** в `_reconstruction_to_dict` сохранять `verts_canon = canonicalize_vertices_for_bucket(verts, ...)` отдельно.

---

# Блок B. S2 — calibration (8 находок)

### B1. КРИТИЧНО: Pose-aware calibration строится, но не используется в S3

**Код:**
- `project/s2_identity/calibration_builder.py:11-50` — строит `PoseNoiseModel` per bucket с интерцептом/slope/curvature.
- `project/s2_identity/engine.py:106-114` — сохраняет в `pose_calibration_models.json`.
- `project/s3_compare/engine.py:471-548` — **читает только `reference.pairwise_noise` и `reference.global_stats`**, не открывает `pose_calibration_models.json`.

**Следствие:** вся pose-aware калибровка (ради чего и был рефакторинг) лежит мертвым грузом. S3 использует **только общий `pairwise_noise[metric]` MAD** — без зависимости от pose_gap.

**Исправление:** `CompareEngine._normalized_distance` должен учитывать pose_gap и читать per-bucket PoseNoiseModel.

### B2. `_load_stage2_records` в S2 не использует `stage2_manifest.json`, если manifest есть — только если записи валидны

**Код:** `project/s2_identity/engine.py:217-258`

`Stage2Record` строится из `info.json + geometry_metrics.json + texture_metrics.json`, но **в manifest уже есть правильно собранные Stage2Record с `metric_notes`, `texture_assessability`, `texture_skin_hint`**, которые вычисляются в `InlineMetricsExtractor`. Если S2 пересобирает Stage2Record из info + json, он теряет:
- `metric_notes` (texture_anomaly_score, cohort_key, physical_*) — нужны для H1 / S4
- `texture_assessability` (eligible/low_confidence/not_assessable)
- `quality_summary` с `quality_sensitive_excluded`

**Следствие:** identity_hint в S4 считается без этих данных → `texture_skin_hint` = "unknown" для большинства фото, что делает H1 boost слабым.

**Исправление:** в `_load_stage2_records` всегда сначала пытаться загрузить `stage2_manifest.json` и валидировать Pydantic.

### B3. `_merge_metric_maps` склеивает geometry и texture без разграничения

**Код:** `project/s2_identity/engine.py:260-267`

```python
for key, value in {**record.geometry, **record.texture}.items():
```

В `pairwise_noise[bucket]` и `global_stats` попадают **все** метрики вперемешку, включая `pose_yaw/pitch/roll`, `face_scale`, `mesh_vertex_count`. Когда S3 `_normalized_distance` смотрит `ref = ref_stats.get(key, {})` и берёт `scale = max(ref.get("mad", 0.0) or ref.get("std", 0.0) or 1.0, 1e-6)` — для `pose_yaw` (degrees) MAD ~ 5-10, а для `bone_*` MAD ~ 0.01-0.05. То есть **scale для pose_yaw доминирует** в сравнении.

**Исправление:** раздельные `geometry_global_stats` и `texture_global_stats` + filter на identity-only metrics.

### B4. `_build_thresholds` использует все texture_* ключи, включая флаги классификатора

**Код:** `project/s2_identity/engine.py:271-279`

`texture_keys = [k for k in global_stats if k.startswith("texture_")]` — в эту выборку попадают и реальные текстурные метрики (`tv_residual_sparsity`), и `texture_silicone_prob`, `texture_real_prob` — **которые сами являются outputs классификатора**, а не измерениями. Использование их в `texture_suspicion = mean + std` приводит к порочной калибровке (мы калибруем на основе значений, которые сами зависят от эра-prior).

**Исправление:** allow-list для raw texture (Tier1+Tier2 core) и **отдельный** slot для classifier outputs.

### B5. `subject_age_years_at` использует `SUBJECT_BIRTHDATE = date(1952, 10, 7)` (Путин) для calibration фото

**Код:** `project/shared/utils.py:24`

Калибровочные фото (т.е. **ваши** фото, как я понял из ТЗ) получают возраст Путина. age_profiles и chronology используют `age_years` который отличается от реального возраста на ~30 лет для ваших фото.

**Исправление:** брать дату рождения из конфига или из EXIF; либо разделить `age_years_main` и `age_years_calibration` (и не использовать второе для ageing model).

### B6. S2 не использует построенный pose-aware reference при annotate_main_dataset

**Код:** `project/s2_identity/engine.py:124-200`

`annotate_main_dataset` использует `CalibrationReference` (с `global_stats`, `pairwise_noise`), но **не использует `pose_calibration_models`**, которые только что построил.

**Исправление:** identity_hint = `PUT/UDMURT/VAS` должен считаться с учётом pose-aware калибровки (как в исходном ТЗ).

### B7. `_build_pairwise_noise` считает дельты **между соседними** по дате, а не все-пары

**Код:** `project/s2_identity/engine.py:344-365`

```python
for left, right in zip(ordered[:-1], ordered[1:]):
```

Это соседние по дате, не **все возможные** пары. В калибровке с 30 фото получается 29 пар, а должно быть 435.

**Следствие:** MAD по `pairwise_noise` слишком оптимистичный (учитывает только ближайших соседей), а в S3 `_normalized_distance` этот MAD используется как scale — то есть scale **занижен** → distances **завышены** → ложные H2-флаги.

**Исправление:** all-pairs.

### B8. `CalibrationEngine._distance_to_reference` не учитывает bucket

**Код:** `project/s2_identity/engine.py:283-307`

Использует `stats = reference.global_stats` (cross-bucket), а `pairwise_noise[record.bucket.value]` — только как additive noise discount. **Bucket-specific MAD** игнорируется. То есть сравнение с использованием общей calibration не учитывает, что разные ракурсы дают разные диапазоны.

---

# Блок C. S3 — compare (8 находок)

### C1. КРИТИЧНО: ICP alignment выбрасывается

**Код:** `project/s3_compare/engine.py:413-425`

```python
# Align using shared visible vertices
src = verts_a[shared_vis]
tgt = verts_b[shared_vis]
aligned_src, R, t, scale = procrustes_align(src, tgt, allow_scale=False)

# Apply to full mesh — НО использует ДРУГОЙ procrustes на всех вершинах
full_aligned = procrustes_align(verts_a, verts_b, allow_scale=False)[0]
```

**Следствие:** R/t из shared-visibility alignment **не применяется** к full mesh. Вместо этого снова делается Procrustes на всех вершинах. Hidden vertices (затылок, невидимая часть) **опять влияют** на alignment — visibility filtering не работает.

**Исправление:** применить сохранённые R/t к full mesh:
```python
full_aligned = (verts_a @ R.T) * scale + t
```

### C2. ICP distance не попадает в evidence score

**Код:** `project/s3_compare/engine.py:327, 363-386`

`icp_distance` вычисляется, но сохраняется только как **строка** в `anomaly_flags` (`f"icp_dist={icp_distance:.3f}"`). `heatmap` тоже не сериализуется в `PairEvidence`.

**Исправление:** добавить `icp_distance: float` и `heatmap: dict` в schema `PairEvidence`.

### C3. `compute_heatmap` использует threshold 0.002 / 0.005 без учёта реальной face_scale

**Код:** `project/s3_compare/engine.py:73-90`

`threshold = 0.002 if is_bone else 0.005` — это hardcoded 2‰/5‰ от face_scale. Для лица с x_span=0.1 model units (3DDFA), 0.002 = 0.0002 model units. Любая реальная неточность Procrustes (~0.001-0.01) делает heatmap "перенасыщенным" красным.

**Исправление:** рассчитывать threshold per-pair на основе residual shared-vis Procrustes alignment.

### C4. Пары adjacent и anchor могут дублироваться

**Код:** `project/s3_compare/engine.py:163-220`

`pair_id = f"{a.photo_id}__{b.photo_id}"`. Если photo сравнивается и как adjacent и как anchor — id **одинаковый**, но оба раза добавляется в `evidence` (без дедупликации).

**Следствие:** одна и та же пара учитывается дважды при усреднении в S4. S4 `avg_geom = mean(p.geometry_distance for p in pairs)` — двойной учёт.

**Исправление:** дедупликация `pair_id` через `set` или check перед `append`.

### C5. `expression_flags` берутся из Stage1.expression_flags (только 3 булевых)

**Код:** `project/s1_extraction/engine.py:763-776` (engine) vs `project/s3_compare/engine.py:570-600` (compare)

Stage1 сохраняет `expression_flags = {smile_excluded, jaw_excluded, neutralized}` — 3 boolean. Compare ожидает структуру `{"intensities": ..., "excluded_zones": ...}` (см. `ExpressionAnalyzer3D.analyze`).

**Следствие:** `anomaly_flags.append(f"expr_excluded_zones={...}")` никогда не срабатывает.

**Исправление:** Stage1 должен сохранять **полный** `ExpressionAnalysis` (intensities, excluded_zones, neutral_score).

### C6. `_get_expression_flags` пытается читать `stage1.exp_vector`, но такого поля в Stage1Record нет

**Код:** `project/s3_compare/engine.py:570-600`

`exp_vector = getattr(stage1, 'exp_vector', None)` → всегда None. `ExpressionAnalyzer3D` никогда не вызывается в S3.

**Исправление:** сохранять `exp_vector` (или `exp_params`) в Stage1Record, и в S3 восстанавливать через `ExpressionAnalyzer.analyze`.

### C7. В `silicone_suspicion` weight для `texture_*` keys = 0.85, но качество и classifier output тоже начинаются с `texture_`

**Код:** `project/s3_compare/engine.py:480-484`

```python
if key.startswith("texture_"):
    key_weight = 0.85
```

Сюда попадают:
- `texture_silicone_prob` (output classifier — не raw feature)
- `texture_real_prob`
- `texture_skin_confidence`
- `texture_unreliable` (bool)
- `texture_assessability` (string)
- `q_tenengrad`, `q_noise_sigma` (нет, не начинаются с texture)

**Следствие:** `texture_silicone_prob` ≈ 0.5 ± 0.3 (в диапазоне); при сравнении фото одного человека с разными priors или разным quality получается ложная дельта.

**Исправление:** отделить `texture_raw_*` (Tier1+Tier2 core) от `texture_meta_*` (outputs).

### C8. `_normalized_distance` возвращает `noise_discount = 0.0` если calibration пустая

**Код:** `project/s3_compare/engine.py:498-499`

`noise_discount = float(np.mean(discounts) if discounts else 0.0)`. Без calibration `noise_discount = 0.0` → `raw_distance - min(raw_distance * 0.7, 0.0) = raw_distance`. То есть **никакой компенсации шума** при отсутствии calibration.

**Исправление:** при отсутствии calibration возвращать `noise_discount = 0.0` И пометить `noise_calibrated = False` в evidence.

---

# Блок D. S4 — verdict (8 находок)

### D1. КРИТИЧНО: ChronologyAnalyzer строит единый timeline для всех bucket-ов

**Код:** `project/s4_verdict/modules/chronology.py:67-90`

`ordered` сортирует **все** фото по дате независимо от bucket. Затем `per_metric_series` собирает значения метрик по всем фото подряд.

**Sanity test:**
```
[photo_A_front (2000-01-01), photo_B_left (2000-06-01), photo_C_front (2001-01-01), photo_D_left (2001-06-01)]
Все 4 фото идут в один timeline. Сравнение photo_B_left → photo_C_front
сравнивает bone_nasion_depth left_threequarter_light vs frontal — разные
абсолютные значения даже для одного человека.
```

**Следствие:** `IMPOSSIBLE_BONE_CHANGE`, `spike:bone_nasion_depth`, `age_inversion:bone_*` флаги генерируются на ложных переходах между ракурсами. Эти флаги подаются в Bayesian engine, повышая H2 likelihood.

**Исправление:** строить chronology **per bucket**, внутри bucket ещё фильтровать по pose_gap <= 5°.

### D2. КРИТИЧНО: ERA_PRIORS всё ещё жёстко вшит в classifier

**Код:** `project/s1_extraction/metrics/modules/texture/classifier_v5.py:73-77`

```python
ERA_PRIORS = {
    "pre_2012": 0.05,      # Original Putin era — silicone very unlikely
    "2012_2021": 0.40,     # Udmurt era — possible silicone
    "post_2021": 0.60,     # Vasilich era — silicone likely
}
```

**Sanity test:**
```python
clf.classify(metrics, year=2022)  # → prior 0.60 silicone
clf.classify(metrics, year=2005)  # → prior 0.05 silicone
# Один и тот же человек с одним и тем же лицом, но probability различается в 12 раз
```

**Следствие:** это встраивает расследуемую гипотезу в классификатор. Любой вывод "post_2021 silicone likely" — circular evidence, не независимое наблюдение. **Это главный методологический блокер**, на который указывал прошлый аудит, и он не исправлен.

**Исправление:** хранить priors в **отдельном** файле с явным комментарием, не применять в production classification; использовать raw texture evidence отдельно от prior-adjusted scores.

### D3. КРИТИЧНО: Platt scaling не обучен

**Код:** `project/s4_verdict/engine.py:52-54` и `calibration_analysis.py:142-160`

```python
self._calibrator = FittedPlattCalibrator()
# ... далее calibrate() вызывается без fit()
```

**Sanity test (static analysis):**
```
calibrator.fit() calls in s4/engine.py: 0
calibrate() calls: 1
```

**Следствие:** `calibrate()` использует fallback `scale = 0.5 + 0.5 * quality` — не реальный Platt fit. `calibration_report.json` строится с `y_true = 1 if predicted == H0 else 0` — **т.е. ECE считается между предсказанием и самим предсказанием** = garbage. Статус "WELL CALIBRATED" в отчёте — фикция.

**Исправление:** либо убрать Platt слой, либо явно обучить на размеченном ground-truth.

### D4. KNN для H0 likelihood зависит от quality `0.65 + 0.35 * quality + identity_boost`

**Код:** `project/s4_verdict/engine.py:264`

Для фото с quality=0 → H0 likelihood = 0.65 * H0_raw. Для quality=1 → 1.0. То есть **H0 likelihood зависит от качества фото**, что вводит bias: высококачественные фото автоматически считаются "тем же человеком".

**Следствие:** при анализе 2024-2025 фото (высокое качество) — boost H0. При 1999-2005 фото (низкое качество) — ослабление H0. Если гипотеза "ранние = Путин, поздние = Vasilich" верна, то **H0-флаги для ранних** частично из-за bias, а не из-за реальной идентичности.

**Исправление:** likelihood не должен зависеть от quality; только prior на evidence.

### D5. `_detect_h1_synthetic` использует `anchor_comparison["heatmap_max"]` и `["bone_zone_violations"]`, но они не вычисляются в S3

**Код:** `project/s4_verdict/h1_engine.py:106-115`

```python
violations = anchor_comparison.get("bone_zone_violations", 0)
heatmap_max = anchor_comparison.get("heatmap_max", 0)
if violations >= 3 and heatmap_max > 0.7:
    return {...}
```

В `_detect_h1_synthetic` (s4/engine.py:599-620) `anchor_comparison` строится с `bone_zone_violations` и `heatmap_max` по отсутствующим ключам (default 0). Срабатывание `prosthetic_zone_tension` **никогда** не произойдёт.

**Исправление:** вычислять heatmap_max и bone_zone_violations в S3 ICP и передавать в S4.

### D6. `texture_silicone_prob` уже использует era prior, затем S4 ещё раз его бустит

**Код:** `project/s4_verdict/engine.py:243-254` и `project/s1_extraction/metrics/modules/texture/classifier_v5.py:203-211`

Двойной prior: classifier уже учёл `prior=0.60` для post_2021; затем S4 берёт `texture_silicone_prob` (с учётом prior) и добавляет в `silicone_physical_boost`. **Prior умножается дважды** — bias квадратично усиливается.

**Исправление:** передавать `posterior_unbiased` и `posterior_with_era_prior` отдельно.

### D7. `_verdict_for_photo` использует texture_unreliable = `record.quality_summary.get("texture_unreliable", False)`

**Код:** `project/s4_verdict/engine.py:248-252`

Но `Stage2Record.quality_summary` (см. schemas.py) — это **QualityMetrics**-like dict, который InlineMetricsExtractor заполняет:
```python
quality_summary={
    "overall_quality": ...,
    "blur_value": ...,
    "noise_level": ...,
    "jpeg_blockiness": ...,
    "sharpness_score": ...,
    "quality_sensitive_excluded": False,
},
```

**`texture_unreliable` НЕ заполняется** в InlineMetricsExtractor. S4 берёт `False` для всех фото, что делает этот фильтр бесполезным.

**Исправление:** S4 должен читать `record.texture.get("texture_unreliable", False)` или передавать `texture_assessability`.

### D8. `_check_biological_constraints` запускается для **каждой** пары календарных соседей

**Код:** `project/s4_verdict/engine.py:498-516`

Вложенный цикл `for record in stage2_records: ... for other_id, other_stage1 in stage1_records.items()`. На 1700 фото получается 1700 × 1700 = 2.89M вызовов. На калибровке 200 фото — 40K вызовов, что медленно. Для main 1700 фото — может занять **часы**.

**Следствие:** либо pipeline падает по таймауту, либо S4 работает неприемлемо медленно, и пользователь его отключает.

**Исправление:** предварительно индексировать stage2_records по bucket, итерировать только по тому же bucket.

---

# Блок E. S5 — report (6 находок)

### E1. `_build_journalist_findings` создаёт pair_id `f"{v.photo_id}__anchor"`, которого нет в `pairs.json`

**Код:** `project/s5_report/engine.py:269`

`pair_ids.append(f"{v.photo_id}__anchor")`. В S3 `pair_id = f"{a.photo_id}__{b.photo_id}"`. Значение `__anchor` нигде не существует как реальный pair_id, и `EvidenceLinker.link` не сможет найти такой pair (строка 280). Возвращает None.

**Следствие:** `evidence_links` в journalist_theses содержит **только** photo links, без comparison references.

**Исправление:** искать реальный pair_id через `pair_index[v.photo_id]`.

### E2. `executive_summary` всегда говорит "X критических, Y высоких аномалий" независимо от того, насколько они поддержаны

**Код:** `project/s5_report/journalist_engine.py:230-244`

```python
lines.append(
    f"Система выявила **{critical} критических** и **{high} высоких** аномалий, "
    "несовместимых с гипотезой об единственном человеке на всех фотографиях."
)
```

Эта фраза **предрешает вывод** до анализа — не зависит от evidence.

**Исправление:** выводить 3 сценария: (а) поддерживает теорию двойников, (б) не подтверждает, (в) недостаточно данных.

### E3. `_build_personas` через HDBSCAN может дать **противоречивые** результаты

**Код:** `project/s5_report/engine.py:170-180`

`HDBSCAN(min_cluster_size=max(3, len(X) // 10), min_samples=2)`. При разных `len(X)` — разные параметры, что делает personas несравнимыми между запусками. На маленькой calibration выборке получится 1-2 persona.

**Исправление:** фиксированные параметры HDBSCAN (например min_cluster_size=10, min_samples=3) для всех запусков.

### E4. `noise_normalized: 0.015` в journalist findings — **захардкожено**

**Код:** `project/s5_report/engine.py:292`

```python
"noise_normalized": 0.015,
"ratio": evidence.get("geometry", 0) / 0.015 if evidence.get("geometry", 0) > 0 else 1.0,
```

0.015 — это **фиксированное число**, не вычисленное из calibration. Сравнение "geometry X в N раз больше noise" ложно, если реальный noise — другой.

**Исправление:** брать noise из `reference.pairwise_noise[bucket][metric_name]["mad"]` или возвращать `None`.

### E5. `top_anomalies` сортируется по `anomaly_score`, но `anomaly_score` в S4 завышен текстурным boost

**Код:** `project/s4_verdict/engine.py:67-69` (anomaly = avg_anomaly + avg_tex * 0.5)

Для фото с `texture_silicone_prob = 0.9` (с учётом ERA_PRIOR!) `avg_tex` будет высоким → anomaly завышен → попадает в top_anomalies → цитируется в journalist theses.

**Следствие:** весь pipeline отчёта "доказывает" то, что уже заложено в ERA_PRIOR.

**Исправление:** топ-аномалии считать по **chronology_score** (биологические ограничения), а не по текстуре.

### E6. `_limitations_section` утверждает "вероятности калиброваны через Platt scaling" — **ложь**

**Код:** `project/s5_report/engine.py:444`

```python
"Вероятности калиброваны через Platt scaling (Platt-like)..."
```

Platt не обучен (D3). Эта строка вводит в заблуждение.

**Исправление:** убрать или заменить на "вероятности эвристические, не калиброванные".

---

# Блок F. Texture/Physical/Classifier (6 находок)

### F1. `TextureRules.evaluate` ищет `flags["specular_ratio"]` и `flags["lbp_entropy_r1"]`, но они не вычисляются

**Код:** `project/s4_verdict/h1_engine.py:128-150`

Правила ссылаются на `flags["specular_ratio"]`, `flags["lbp_entropy_r1"]`, `flags["fft_peak_regularity"]`, но в `feature_flags` из texture_anomaly приходят **другие** ключи (`tv_residual_sparsity`, `edge_tortuosity_mean`, ...).

**Следствие:** `texture_silicone_signature` и `texture_regular_microrelief` **никогда не срабатывают**.

### F2. `cohort_detector.feature_flags` пуст в H1_detector

**Код:** `project/s4_verdict/engine.py:617-620`

```python
texture_anomaly = {
    "anomaly_score": float(...),
    "feature_flags": {},
}
```

Хардкод `feature_flags = {}` → все `feature_flags.get(key, default)` в CrossModalTextureRules возвращают default → большинство правил не срабатывают (см. F1).

**Исправление:** брать реальные `feature_flags` из `cohort_detector.score()` через `stage2_records[photo_id].metric_notes["texture_anomaly_flags"]`.

### F3. `cohort_key` = "early_scan"/"udmurt_era"/"vas_era" — hypothesis label встроен в cohort

**Код:** `project/s1_extraction/metrics/texture_anomaly.py:35-45`

Cohort key использует **имена гипотез** (Vas, Udmurt). Это не нейтральное название эпохи, а **маркировка предполагаемой персоны**. Дальнейший anomaly detection **внутри** cohort "vas_era" уже presupposes, что эти фото — Vasilich.

**Исправление:** нейтральные имена ("era_1999_2005", "era_2005_2012", ...).

### F4. `EARLY_UDMURT_MASK` rule в CrossModalTextureRules требует `sss_index < 0.05` и `specular_sharpness > 2.5`

**Код:** `project/s1_extraction/metrics/cross_modal_rules.py:35-43`

Однако F2 показывает, что `feature_flags` пуст → `feature_flags.get("sss_index", 1.0) > 0.05` → return False. Правило никогда не сработает. Аналогично для всех 6 rules.

**Исправление:** загружать `feature_flags` из S1 cohort_detector output.

### F5. `clean_texture_metrics` не убирает `texture_feature_weights_json` (только `TEXTURE_NOISE_FIELDS`)

**Код:** `project/shared/validation.py:354-360`

`texture_feature_weights_json` есть в `TEXTURE_NOISE_FIELDS`, но `texture_noise_sigma` помечен как "Duplicate of noise_level" — что неверно (texture noise вычисляется по-другому, на albedo).

### F6. `spectral_slope` (Tier3) в S4 verdict интерпретируется в диапазоне [2, 4] → [0, 1]

**Код:** `project/s4_verdict/engine.py:262-265`

```python
_raw_spectral = phys.get("spectral_slope", 2.5)
_norm_spectral = float(np.clip((_raw_spectral - 2.0) / 2.0, 0.0, 1.0))
```

Но `_raw_spectral` = `phys.get("spectral_slope", 2.5)`. В `PhysicalTextureFeatures` поле называется `spectral_slope`, в metric_notes записывается как `physical_spectral_slope`. То есть доступ должен быть `phys["spectral_slope"]` (после strip префикса) — что и есть в коде. ОК.

**Но:** `specular_sharpness` и `seam_score` могут быть 0.0 если ROI пустые (см. A9 — physical_extractor возвращает мусор). Тогда `silicone_physical_boost = 0` независимо от реального silicone, что:
- для лиц с **пустыми ROI** (часто профильные) → 0 boost → не flag
- для лиц с ROI в правильном месте → реальный boost

**Исправление:** `silicone_physical_boost` должен наказывать фото с missing/empty physical features (quality penalty).

---

# Блок G. Конфигурация и инфраструктура (4 находки)

### G1. `validate_config` отвергает `s2` в stages

**Код:** `project/shared/config_validation.py:24-30`

```python
valid_stages = {"s1", "s3", "s4", "s5", "s6"}
```

Но `PipelineRunner.DEFAULT_STAGES = ("s1", "s2", "s3", "s4", "s5")`. Если в `pipeline.yaml` указан `stages: [s1, s2, s3, s4, s5]` — config validator ругается "Invalid stage 's2'".

**Следствие:** пользователь не может задать `--config pipeline.yaml` без warning о s2.

**Исправление:** заменить `valid_stages` на `{"s1", "s2", "s3", "s4", "s5"}`.

### G2. `s2` конфиг ищется как `s3` в `engine_v2.py` (мёртвый код)

**Код:** `project/s3_compare/engine_v2.py:8-15` 

`engine_v2.py` не импортируется в `run.py` — мёртвый код, но в нём `min_calibration_pairs` в `s3` config (старая нумерация до рефакторинга).

**Исправление:** удалить engine_v2.py или явно задепрекейтить.

### G3. Hardcoded путь к DPTN_DATA_ROOT

**Код:** `project/run.py:54-58`

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DPTN_DATA_ROOT", PROJECT_ROOT / "data"))
```

`parents[1]` = `deeptn/`, не `deeptn/project/`. То есть `PROJECT_ROOT / "data"` — это `deeptn/data`, что **противоречит** ТЗ (где `data/photo/all` и `data/storage/main`). Однако текущая команда `python -m project.run` всё равно ожидает default = `deeptn/data` — **ошибка** для пользователя.

**Исправление:** `PROJECT_ROOT = Path(__file__).resolve().parents[1]` корректно, но default должен быть `PROJECT_ROOT / "photo" / "all"` (как в ТЗ). Сейчас это `PROJECT_ROOT / "data" / "photo" / "all"` что неверно при `PROJECT_ROOT = deeptn`.

### G4. `s4/comparisons` конфиг читается как `s3` в runner

**Код:** `project/run.py:163-174`

`compare_engine = CompareEngine(config=self.config.get("s3", {}))` — нормально, CompareEngine это S3. Но также `verdict_engine = VerdictEngine(config=self.config.get("s4", {}))` — нормально. Эта область ОК.

**Однако:** `config = self.config.get("s1", {})` (строка 211) — `ReconstructionAdapter` берёт `device`, `detector_device`, `backbone` из `s1` config. А `InlineMetricsExtractor` берёт `self.config.get("s2", {})` (строка 218) — то есть **config для s2 передаётся в InlineMetricsExtractor** (а не в CalibrationEngine!). Это переименование — `s2` в config это `inline_metrics` config, а не `calibration` config. **Семантическая путаница**.

---

# Блок H. Hypothesis bias (4 находки, МЕТОДОЛОГИЧЕСКИЕ)

### H1. КРИТИЧНО: `_infer_era` в alpha_tracker использует hypothesis labels

**Код:** `project/s4_verdict/alpha_tracker.py:104-114`

```python
def _infer_era(self, dates):
    if avg_year < 2005: return "early_putin"
    elif avg_year < 2012: return "mature_putin"
    elif avg_year < 2021: return "udmurt_era"
    else: return "vas_era"
```

**Следствие:** кластеры identity-векторов называются "udmurt" / "vas", что встраивает теорию в метаданные отчёта. Журналист, читающий `persona_2 = vas_era`, делает выводы по имени, не по evidence.

**Исправление:** нейтральные имена `cluster_0`, `cluster_1`, ... + era в виде диапазона дат.

### H2. `_era_from_year` в h1_engine = то же самое

**Код:** `project/s4_verdict/h1_engine.py:21-26`

`early_scan` / `udmurt_era` / `vas_era` вшиты в код. Любая photo в 2024 будет триггерить `vas_era` правила.

### H3. `EARLY_UDMURT_MASK` (2012-2021) и `EARLY_SCAN_SILICONE` (1999-2005) — формулировки, намекающие на вывод

**Код:** `project/s1_extraction/metrics/cross_modal_rules.py:30-50`

Названия правил содержат "silicone", "mask" — то есть **до проверки данных** правило помечено как детектор маски. Научный метод требует нейтральных имён (rule_a, rule_b, ...) и описания вывода после evaluation.

### H4. `texture_silicone_prob` как имя выхода

**Код:** везде в `texture_metrics.json` ключ называется `texture_silicone_prob`, а не `texture_synthetic_prob` или `texture_anomaly_prob`.

**Следствие:** пользователь, смотрящий на JSON, видит готовый вердикт "silicone = 0.7", а не измерение + порог. Это создаёт bias подтверждения.

**Исправление:** переименовать в `texture_skin_classifier_score` или `texture_anomaly_posterior`.

---

# Блок I. Chronology конкретные баги (4 находки)

### I1. `_check_biological_impossibilities` флагит `IMPOSSIBLE_BONE_CHANGE` для каждой смежной пары фото

**Код:** `project/s4_verdict/modules/chronology.py:175-198`

```python
if is_bone_metric:
    if gap_days < 365:
        if delta > 2.0:
            per_photo_scores[p_curr.photo_id] += 1.2
            per_photo_flags[...].add(f"IMPOSSIBLE_BONE_CHANGE:...")
```

**Sanity test:** для пары фото одного человека в разных bucket (frontal vs left_threequarter_light) `bone_nasion_depth` отличается на 0.01-0.05 (из-за canonicalization к разным yaw). Если этот delta > 2.0 (нормализованный) — флаг.

Поскольку `_check_biological_impossibilities` использует `is_bone_metric = "bone_" in name`, а `bone_nasion_depth` нормализуется на `face_scale = x_span = ~0.1`, то `delta = |0.5 - 0.51| / 0.1 = 0.1` — **меньше** 2.0. OK для этого метрика. Но `bone_zygomatic_width = x_span / face_scale = 1.0` нормализован; между buckets это будет ~1.0 ± 0.05 → delta = 0.05 — OK.

**Однако:** `bone_gonial_angle` (degrees, ~120°) delta между buckets = ~5° → if `delta > 2.0` → **флаг IMPOSSIBLE_BONE_CHANGE на каждой паре**.

**Следствие:** `IMPOSSIBLE_BONE_CHANGE:bone_gonial_angle` генерируется на **всех** парах.

### I2. `RETURN_TO_BASELINE` флаг требует 5 фото deviation window

**Код:** `project/s4_verdict/modules/chronology.py:209-227`

```python
for i in range(early_end + 5, len(values)):
    window = metric_series[max(0, i-5):i]
```

На малых выборках (< 10 фото) не срабатывает. На больших — генерирует ложные срабатывания из-за шума реконструкции.

### I3. `_detect_change_points` использует ruptures.Pelt с `model="cos"`

**Код:** `project/s4_verdict/modules/chronology.py:265-310`

`cos` distance для высоко-размерного profile (~50 метрик) **известно нестабильно** при n < 100. Penalty `np.log(n) * n_metrics * 0.5` тоже подозрительно. На 1700 фото * 50 метрик = 85K-dim профиль — точно не помещается в оперативку и не интерпретируемо.

**Исправление:** либо `model="l2"`, либо per-metric univariate PELT, либо cusum.

### I4. `_detect_cross_metric_return` использует китайские символы

**Код:** `project/s4_verdict/modules/chronology.py:325` (см. A7)

```python
g_was偏离 = np.mean(np.abs(g_norm[idx-2:idx])) > 1.0
t_was偏离 = np.mean(np.abs(t_norm[idx-2:idx])) > 1.0
g_returned = abs(g_norm[idx]) < 0.3
t_returned = abs(t_norm[idx]) < 0.3
```

Использовать `g_was_deviated` и `t_was_deviated`.

---

# Блок J. Validation gaps (3 находки)

### J1. `validate_geometry_metrics` не проверяет coverage

**Код:** `project/shared/validation.py:296-350`

Проверяет только NaN/None/тип, **не** проверяет:
- минимальное число метрик (>= 50 для нормального forensic output)
- наличие `bone_nasion_depth` / `bone_orbit_*_depth` (bone-priority)
- presence of catalog-overlap (active → catalog)
- canonicalization status

**Следствие:** пустой `{}` или partial JSON проходит валидацию.

### J2. `validate_all` не запускается в `ExtractionEngine._process_one` для geometry

**Код:** `project/s1_extraction/engine.py:380-385`

```python
info_issues = validate_info_json(record.model_dump(), photo_path.stem)
```

Только `info.json` валидируется. `geometry_metrics.json` и `texture_metrics.json` НЕ валидируются.

**Следствие:** corrupted JSON на этих файлах обнаружится только в S3/S4, что ломает весь stage.

### J3. Нет integration test для atomic write / resume

`save_json` теперь атомарный (tmp + rename) — это **исправлено**. Но нет тестов:
- что происходит если `.tmp` файл уже существует (collision)
- что S4/S5 не падают на partial JSON (stale data после крэша)

---

# Рекомендованный порядок исправлений (P0 → P2)

## P0 — Блокирующие

1. **Убрать ERA_PRIOR** (D2) из classifier_v5 — это методологический блокер для публикации.
2. **Исправить `id_params` / `exp_params` доступ** (A1) — изменить `_reconstruction_to_dict` чтобы положить `id_params`, `exp_params` в `payload`.
3. **Per-bucket chronology** (D1) — переделать `_series()` чтобы фильтровать по bucket.
4. **Pose-aware calibration в S3** (B1) — `CompareEngine` должен загружать `pose_calibration_models.json` и применять pose-conditioned noise.
5. **Исключить pose_yaw/pitch/roll, face_scale, mesh_vertex_count** из identity distance (A2, A3, A4).
6. **Исправить physical_extractor** (A9) — transform landmarks в face_crop space или работать с full image.
7. **H1_evidence передавать реальные feature_flags** (F2, F1) из S1 cohort output.
8. **Перенести `bone_gonial_angle` raw degrees** (I1) — нормализовать по face_scale или исключить.

## P1 — Калибровка и прозрачность

9. **Platt fit** (D3) — либо убрать, либо fit на labelled set.
10. **Дедупликация pairs** (C4) и **передача ICP distance** (C2).
11. **Нейтральные имена** (H1, H2, F3) — cluster_0, era_2005_2012.
12. **Calibration не на калибровочном возрасте** (B5) — separate age field.
13. **H1 evidence flags** правильно (D5, F1) — реализовать `heatmap_max` и `bone_zone_violations` в S3.
14. **texture_unreliable propagation** (D7).

## P2 — Engineering

15. **S2 pre-validate geometry/texture** (J2).
16. **Bucketed alignment per pose** (A5, A6, A10).
17. **Coverage validator** (J1).
18. **All-pairs calibration noise** (B7).
19. **Test_personas/migration** — test data в `project/tests/`.
20. **Удалить engine_v2.py** (G2).
21. **CI integration** — хотя бы 1 smoke test (synthetic → run S1 → assert geometry_keys).

---

# Acceptance tests

Прежде чем новый полный запуск:

1. Synthetic mesh при 20 разных yaw/pitch/roll после canonicalization даёт одинаковые anatomy metrics в пределах tolerance.
2. `id_params` и `exp_params` присутствуют в `geometry_metrics.json` для каждой фото.
3. Active output имеет ≥ N% overlap с texture/geometry catalog.
4. Visibility mask неправильной длины даёт **structured failure**, не `{}`.
5. Missing annotation groups → `INSUFFICIENT_GEOMETRY` (не 19-metric stub).
6. Texture failure не удаляет успешно рассчитанную geometry.
7. Missing geometry → `INSUFFICIENT_EVIDENCE` (не silent skip).
8. Один malformed JSON не останавливает другие фото.
9. Rigid-transform mesh pair после S3 alignment residual ≈ 0.
10. Pair IDs уникальны.
11. Per-bucket chronology: импульс между bucket-ами НЕ генерирует флагов.
12. ERA_PRIOR не применяется: classify(texture, year=2005) и classify(same_texture, year=2024) дают одинаковый raw posterior.
13. Каждый journalist thesis ссылается на реально существующие pair_id/photo_id.
14. Платт calibrator либо обучен и сохранён, либо не упоминается.
15. Pose-aware noise discount реально уменьшает distance для pose_gap < 5°.
16. physical_extractor ROI находятся в правильных зонах (санити-тест: нарисованное лицо с известными landmarks → ear_roi покрывает ухо).
17. Run на 5 фото одного человека даёт ≥ 80% H0_SAME.
18. Run на 5 фото разных людей даёт ≥ 60% H2_DIFFERENT.

---

# Что нужно для следующей итерации

1. `pip freeze` или `conda env export` для воспроизводимости.
2. `pyproject.toml` или `requirements.txt` (текущий `audit only` режим — `python -m compileall` проходит, но `import torch` падает).
3. 1 полный `deeputin_*.log` запуска на 5-10 фото main + 5 calibration.
4. 5-10 фото с известным ground truth (PUT vs не-PUT) для sanity test новой байесовской модели.
5. Список 5-10 целевых bone-zone метрик с физическими диапазонами (после фикса A1-A6, чтобы валидировать absolute scale).
6. Запуск `s2` с `pose_calibration_models.json` после фикса B1, чтобы убедиться, что pose-aware noise реально используется в S3.

---

# TL;DR — главные 3 проблемы

1. **ERA_PRIOR** (D2) — встроенный bias. Без удаления этого prior любой вывод "post-2021 = силикон" — circular evidence.
2. **ChronologyBucketMixing** (D1) — все флаги `IMPOSSIBLE_BONE_CHANGE`, `spike:`, `age_inversion` смешивают ракурсы. Без per-bucket chronology S4 генерирует ложные H2-флаги.
3. **id_params/exp_params access bug** (A1) + **pose_yaw pollution** (A2) — geometry metrics либо пустые для identity, либо загрязнены pose-углами. S3 distances в результате не отражают реальной геометрии.

Без фикса этих трёх выводы `verdict.json`, `chronology.json`, `report.json` **не могут считаться доказательной базой**.

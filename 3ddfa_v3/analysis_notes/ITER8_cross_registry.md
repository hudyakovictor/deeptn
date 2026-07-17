# Итерация 8 — кросс-анализ идеи↔код, pose-coverage person_01..05, финальные реестры

## 1. identity_scoring_config vs identity_scoring_config_new
Путь: project/s2_metrics/modules/geometry/legacy_metrics/
- Оба файла 721 строк, schema v3, weight_profile=identity_only_v3
- **ПОБАЙТНО/СТРУКТУРНО ИДЕНТИЧНЫ**: 0 value diffs, 0 only-A/only-B keys
- Веса: id_cosine 0.35 | bone_extract_best 0.55 | zone_relations 0.10
- bone_extract_best: 49 метрик; unique named metrics: 167
- 9 pose buckets: frontal + L/R threequarter_{light,mid,deep} + L/R profile
- global_disabled включает eye_mask_silicone_prob + silicone_prob (текстура силикона ВЫКЛЮЧЕНА из identity scoring!)
- Вывод: `_new` — мёртвый дубль / не применённый diff; можно удалить один из двух

## 2. Pose policy (два поколения)
### project legacy_metrics/policy.py (9 бакетов light/mid/deep)
POSE_POLICY: frontal, left/right_threequarter_{light,mid,deep}, left/right_profile, unclassified
POSE_YAW_BILATERAL_OFF_DEG = 45
PROFILE_FRONTAL_ONLY_METRICS — большой блок bilateral/asymmetry запрещён на профиле

### newapp/backend/core/policy.py (более тонкие yaw-пороги)
POSE_YAW_OCCLUDE_DEG=18, BILATERAL_OFF=25, ZONE_HIDE=40, RECONCILE=12 (HPE vs 3DDFA)
PEER_COHORT_POSE_MAX_DEG=12

### Конфликт с current (из ITER2, lost archive)
_BUCKET_FALLBACK medium→light AND mid→light; dead profile_left/right keys
Старый pair_check: left_threequarter_light/mid/deep naming = identity_scoring, НЕ current LEFT_THREEQUARTER_MEDIUM

## 3. person_01..05 — coverage (без HPE; proxy L/R brightness asymmetry)
Все: square face crops, sequential frames (head-turn series). Нет EXIF/pose JSON.

| person | n | res | MB | proxy pose | side | notes |
|--------|---|-----|-----|------------|------|-------|
| 01 | 506 | 618² | 25.6 | profileish 87% | L70/R49 | сильный turn L↔R, pitch systematically UP (tb≈-0.17) |
| 02 | 658 | 424² | 9.4 | frontalish 73% | C81/R39 | слабый yaw range; low-res; мало profile |
| 03 | 470 | 620² | 11.0 | 3q 54% + profile 33% | R112 almost only | **односторонняя** серия (почти только RIGHT) |
| 04 | 412 | 1080² | 58.7 | frontal 64% + 3q 27% + profile 9% | L/R/C balanced | лучшее качество; наиболее «полный» yaw |
| 05 | 551 | 568² | 12.3 | 3q 67% + frontal 28% | R-dominant | mid-yaw bias right |

**Итог coverage для multi-person calibration:**
- Сумма 2597 кадров — отлично для noise floor
- Дыры: person_02 почти без profile; person_03 почти без LEFT; person_01 pitch bias up
- person_04 — якорь HQ; person_01 — якорь full yaw sweep
- **Нужен реальный HPE/3DDFA bucket assignment** перед production use (proxy ≠ forensic buckets)
- Плавность серий высокая (mean|Δlr| 0.003–0.006) → настоящие continuous head turns, не random photos

## 4. Кросс-маппинг идея → где живёт код

| Идея | project6 | newapp | old | Статус |
|------|----------|--------|-----|--------|
| Umeyama no-scale | s1/s4 alignment | pipeline/alignment | pipelinesave | РЕАЛИЗОВАНО |
| Visibility ∩ mask | multi | reconstruction/alignment | task axiom | РЕАЛИЗОВАНО |
| Expression exclusion | ? | zones/compare/scoring | — | newapp only (частично) |
| Delight UV | s1 ref | uv_module + uv_gen | uv_module | РЕАЛИЗОВАНО (user confirmed) |
| Fuzzy Bayes | stub const | full 9k LOC | — | newapp; circular priors |
| Era priors / named carriers | — | fuzzy_bayes config | — | РЕАЛИЗОВАНО-ВРЕДНО |
| Carrier ceiling / pub guard | — | posterior/decision | — | РЕАЛИЗОВАНО-ВРЕДНО |
| Peer cohort | — | peer_cohort.py | — | РЕАЛИЗОВАНО (newapp) |
| Blemish/mole UV track | — | blemish_tracking | — | РЕАЛИЗОВАНО (newapp) |
| Skin authenticity / synthetic | texture modules | skin_authenticity | biometric_texture | СЛОМАНО (0.75 floor) |
| Letterbox face_crop | s1 engine | analysis + test | — | КОД ЕСТЬ, нужен re-extract |
| Biological limits | s5_verdict | — | — | project6; unit bugs |
| Baseline return | s5 | — | — | project6 only |
| HDBSCAN alpha | s5 alpha_tracker | — | appold2/3 | project6 + old |
| Spectral Laplacian mesh | — | texture name collision | deeputin0/spectral | ПОТЕРЯНО (mesh graph) |
| TransitionValidator ±30d | — | — | v8/v9 | ПОТЕРЯНО |
| FaceMorpher sequences | — | — | deeputin | ПОТЕРЯНО (UI) |
| Mahalanobis advanced | texture_anomaly only | — | appold6 + many | ЧАСТИЧНО |
| RulesEngine modular | s5 modules (unused path) | fuzzy rules | — | ДУБЛЬ |
| engine_v2 AnchorBased | s4 unused | — | — | МЁРТВЫЙ КОД |
| NoiseModel/IdentityDisc | s3 unused | calib ref | — | МЁРТВЫЙ КОД |
| Same-date forensics | — | decision hint | plan2 artifacts | plan layer |
| Red-team aging AN-046 | — | — | — | ТОЛЬКО plan2, не код |
| identity_scoring pose core | policy+json | — | — | project6 legacy |

## 5. ФИНАЛЬНЫЙ РЕЕСТР ОШИБОК (консолидация ITER2–8)

### A. Критические (ломают научную валидность)
A1. Era-priors circular reasoning (ERA_INVESTIGATION_PRIORS = verdict map)
A2. Named carriers H_UDMURT/H_VASILICH inside Bayes vs TZ H0/H1/H2
A3. synthetic_prob hard floor 0.75 + fractal=100 miscalibration
A4. Geometry SNR ceiling 1.8571 (noise_cap 0.35)
A5. Evidence double-counting (6–10 blocks, same raw)
A6. Publication guards mutating posterior (vs post-hoc principle)
A7. H_CARRIER_SWITCH mass in rules without hypothesis in config
A8. empty texture/metrics in verdicts / texture_ratio_max null

### B. Калибровка / данные
B1. Calibration photos all date=2026-05-06 age=73
B2. far_match pose_distance up to 24° (TZ wants <5° same bucket)
B3. One cal id used 66× (skew)
B4. face_crop stretch without letterbox (code fixed, storage stale)
B5. skin resize_canonical 424×500→256×256 still non-letterbox
B6. person_01–05: uneven yaw/side coverage (proxy); need real HPE buckets

### C. Pose / buckets / naming
C1. _BUCKET_FALLBACK medium↔mid→light desync; dead profile_left/right
C2. Three naming eras: profile_left (v9) / left_profile (scoring) / left_profile schema vs LEFT_PROFILE enum
C3. light/mid/deep (9) vs medium (current 5-stage) migration incomplete
C4. POSE_YAW thresholds differ project45° vs newapp 18/25/40

### D. Dead code / dual engines
D1. engine_v2 AnchorBasedCompare unused
D2. NoiseModel/ShiftModel/IdentityDiscriminator unused
D3. BayesianEngine+RulesEngine modular path unused; hand-tuned VerdictEngine active
D4. identity_scoring_config_new == old (noop duplicate)
D5. Classifier v2/v3/v4 dead; skin_classifier_v3.pkl missing
D6. h1_engine NameError TextureRules/GeometryRules
D7. mesh_delta legacy shim

### E. Биология / хронология
E1. METRIC_ALIASES vs BONE_METRICS name mismatch → bio-checks skip
E2. mm vs ratio unit confusion biological_limits/chronology
E3. Linear fake age in service get_ageing_series
E4. apply_aging_correction multiplies z-score not threshold (appnew444)
E5. pitch correction disabled wrong direction (appnew444; verify survival)

### F. Инженерия / ops
F1. selected_metrics.json missing → fail-open
F2. core.contracts/constants stubs
F3. legacy_metrics/types.py shadows stdlib
F4. 4 env prefixes NEWWAP_/DUTIN_/DPTN_/DEEPUTIN_
F5. Silent try/except → null interpreted as no contradiction
F6. External deps missing: HPE path, METRIC_EVIDENCE_TABLE, leaderboard CSV
F7. plan2.txt self-duplicated stages 0–1 ×4–5

## 6. ФИНАЛЬНЫЙ РЕЕСТР ИДЕЙ (keep / fix / revive / drop)

### KEEP (принципы + рабочие модули)
K1. Pose-matched compare only + visibility ∩ before Umeyama no-scale
K2. Calibration = algorithm noise floor on known real skin (multi-person now)
K3. Bone-priority zones; expression/visibility gating
K4. Publications & memes = post-hoc concordance, NEVER priors
K5. Red-team alternatives mandatory (aging, weight, masking, pipeline bias, sampling)
K6. Unsupervised structure BEFORE naming carriers
K7. Provenance/hashes/run_id
K8. UV delight pipeline (alive)
K9. Peer cohort z-scores same bucket
K10. Blemish UV stability
K11. Quality/adversarial gate (blur→texture weight 0)
K12. identity_only_v3 bone-heavy weights (0.55) directionally right if metrics clean

### FIX (есть код, чинить математику/калибровку)
F-i1. Remove era priors from inference; neutral H0/H1/H2 or identical/non-identical/attack
F-i2. Likelihoods from empirical distributions, kill caps/floors that encode answer
F-i3. synthetic fusion: kill 75 floor; recalibrate fractal; letterbox all resizes; re-extract
F-i4. SNR without hard cap that forces 1.8571
F-i5. Single evidence path per raw signal
F-i6. Unify pose bucket ontology + thresholds once
F-i7. Bio-limits name/unit alignment
F-i8. Cal photo metadata dates/ages

### REVIVE from old (valuable, not in active path)
R1. TransitionValidator backbone ±30d for profiles
R2. deeputin0 spectral mesh graph (not texture FFT name collision)
R3. PerViewTrajectoryAnalyzer / rich AgingModel
R4. FaceMorpher for UI compare page
R5. AlphaStabilityTracker HDBSCAN (exists project6 — wire if alpha 199-D available)
R6. Mahalanobis advanced_anomaly
R7. Journalist AlternativeExplanations / BaselineReturn (exists project6 — ensure used)
R8. 2D-only forensic cues from biometric_texture that don't need ears/hair mesh (sclera? pores age-aware? crow's feet?) — selective, not full cascade

### DROP / DO NOT REVIVE as-is
X1. Ear/Hairline as 3DDFA mesh evidence (user: library can't)
X2. Hardcoded DOUBLE_1..N metric profiles (confirmation bias)
X3. OSINT→prior injection
X4. Meme labels as classes
X5. identity_scoring_config_new duplicate
X6. Multiple competing verdict engines in one runtime — pick one clean design

## 7. Связь person_01..05 с ТЗ калибровки
ТЗ: multi-person head-turn to measure algorithm noise across poses, not store faces.
Данные подходят: continuous turns, 5 identities, 2.6k frames.
Перед использованием: run real pose→bucket; stratify noise model by bucket×quality; exclude expression frames; person_03 needs LEFT supplement or mark right-only; person_02 low-res → lower texture weight.

## 8. Пробелы данных (ещё не в sandbox)
- Current 5-stage code archive (lost after reset) — user re-upload
- /dutin/newapp/results historical
- all_publications folder
- selected_metrics.json / evidence tables
- Real HPE on person_01..05

# Итерация 7 — корпус ТЗ / заметок / аудитов → реестр идей

## Источники (прочитаны)
1. newapp/aboutplatform.txt — каноническое продуктовое ТЗ (нейтральный H0/H1/H2)
2. newapp/plan2.txt (1421 строк) — план финального анализа AN-00x..AN-067, этапы 0–17; **текст задублирован 4–5 раз** (Этап 0 повторяется)
3. newapp/future-fix.txt — отложенные фиксы (face_crop letterbox, synthetic_prob=0.75 floor, fractal)
4. newapp/developers/audit-1..8 — 8 независимых аудитов NEWWAP/fuzzy_bayes (июнь 2026)
5. newapp/различия(гипотезы)/ — классификации масок/двойников + 1081 numbered ulik-items
6. pipeline/task.txt = old/pipelinesave/task.txt (1102) — аксиома Umeyama+visibility mask
7. old/appnew444/fix.txt — fatal errors monolit-эпохи (CranialShapeEngine, pitch, hardcoded DOUBLE)
8. old/test-new-ver/task.md — FLAME/BFM stack, Apple Silicon, no CUDA
9. old/uv_module/UV_MODULE_TZ.md — HD UV delighting (модуль жив, не потерян)
10. pair_check_report.txt — покрытие бакетов 1248 фото (старый pose-naming)
11. artifacts/00..17 — plan2 уже частично исполнялся (deliverables freeze)

## Канон ТЗ (aboutplatform) — что ДОЛЖНО быть
- Фото = археологический артефакт; 3DDFA_v3; 21 зона лица (bone priority)
- Байес: **H0 same natural / H1 synthetic / H2 different person** (НЕ named carriers)
- Калибровка на известном человеке (шум алгоритма, не «лицо»)
- Pose-dependent forensics + expression exclusion
- FFT/LBP/albedo/specular texture detector
- Chronology + biologically impossible changes (asymmetry inversion)
- Pre-extracted 3D cache, morph UI, heatmap thresholds, publication correlation as post-hoc only
- Гипотеза пользователя (в конце ТЗ): 1999–2012 original → 2012–2021 carrier A → 2021–2026 carrier B + mixing; >50% silicone possible; publications = concordance not priors; no ears/neck from 3DDFA

## plan2 — научный слой (поверх пайплайна)
Этапы: 0 audit → 1 chrono atlas → 2 A→B → 3 era B → 4 B→C → 5 era C → 6 same-date → 7 chrono-presignal → 8 intra-era → 9 publications → 10 red-team (aging/weight/masking/pipeline-bias/sampling) → 11 Bayesian matrix of MODELS → 12 semantic layer → 13 clustering/NN-graph → 14 critical anomalies → 15 deliverables → 16 refinements → 17 freeze.
Принципы: nothing proven without recheck; every hyp has alternatives; memes = weak signal only; publications ≠ classification.

## Аудиты 1–8 — консенсус (что ВСЕ нашли)
Средние баллы ~28–55/100. Общий вердикт: инженерия 70+, наука 20–35, публиковать нельзя.

### Критический консенсус (подтверждено несколькими аудитами)
C1. **Circular era-priors**: ERA_INVESTIGATION_PRIORS + ERAS boundaries → вердикты 1999–2011 = 100% H_ORIGINAL, 2012+ = UDMURT/VASILICH. Дисперсия 0.
C2. **Named carriers в Bayes** (H_UDMURT/H_VASILICH) vs ТЗ H0/H1/H2 — confirmation bias.
C3. **synthetic_prob = 0.75** у ~80% (fusion floor + fractal=100) — канал выключен (future-fix #2).
C4. **Geometry SNR ceiling 1.8571** (noise_cap 0.35) у большинства пар — геом-доказательство не дискриминирует.
C5. **texture_ratio_max = null** / empty_metrics_verdicts 1807/1807 (audit-1) — метрики не доезжают до verdicts.
C6. **Double-counting evidence** — один raw-сигнал через 6–10 EvidenceBlocks.
C7. **H_CARRIER_SWITCH в rules** при отсутствии в HYPOTHESES (= ITER6 #23 confirmed as audit finding).
C8. **R_PAIR_IDENTITY_MATCH → 80% carrier** при сильном id-match + tex_syn — логически спорно (Mask-vs-Mask по умолчанию).
C9. **Calibration date bug** — все cal photos date=2026-05-06 age=73.
C10. **Post-hoc publication guards** в decision.py меняют вердикт (нарушает own principle).
C11. **face_crop stretch** без letterbox (future-fix #1) — искажает texture input.
C12. **Silent try/except → null** интерпретируется как «нет противоречия» → завышает H0.

### Полезные предложения аудитов (сохранить как идеи, не как догму)
I1. Likelihood ratios из эмпирических распределений (LFW/CelebA), не hand weights
I2. Named carriers → layer интерпретации, не layer inference
I3. Optical compensator / PnP focal length
I4. HMM / Kalman filter aging trajectory
I5. Adversarial quality gate (blur → texture weight → 0)
I6. Confidence map per-vertex expression (не ×0.1 expression)
I7. Overnight pre-extract .npz + browser morph
I8. Sensitivity analysis / multiple priors scenarios (plan2 AN-053/054)
I9. Unsupervised clustering BEFORE naming clusters (appnew444 audit + plan2 AN-059)
I10. Noise residual FFT (не RGB) для silicone vs JPEG

## Реестр идей: реализовано / частично / потеряно / запрещено 3DDFA

### РЕАЛИЗОВАНО (в newapp и/или current project)
R1. 3DDFA_v3 reconstruction + pose buckets
R2. 21 zones + bone_priority (zones.py)
R3. Umeyama no-scale + visibility intersection (task.txt axiom → pipeline/compare)
R4. Expression exclusion (constants thresholds; quality varies)
R5. Calibration on self-photos + health buckets (concept alive)
R6. Fuzzy Bayes structure (membership/rules/posterior) — но с fatal bias
R7. Publication correlation module (post-hoc principle declared)
R8. Provenance/hashes/run_id (сильная сторона аудитов)
R9. plan2 artifacts 00–17 executed at least once
R10. UV module (user: alive, not in small archive)
R11. Blemish/mole tracking in UV (pipeline/blemish_tracking)
R12. peer_cohort, texture_live_reference, skin_authenticity stack

### ЧАСТИЧНО / СЛОМАНО
P1. Texture silicone detector — code exists, calibrated wrong (0.75 floor)
P2. Chronology impossible changes — flags exist, metrics/units broken (ITER4 #17–18)
P3. Bayesian court — structure yes, science no (priors+caps)
P4. Pose-dependent metric gating — partial; bucket name drift
P5. Pre-extracted mesh cache for UI compare — designed, often re-extract on fly
P6. Morph UI / heatmap thresholds — TZ detailed, implementation incomplete
P7. Biological aging model — linear fake age in service (audit-1)

### ПОТЕРЯНО / ВЫРЕЗАНО (но ещё в old/)
L1. Spectral graph Laplacian (deeputin0/spectral.py)
L2. PerViewTrajectoryAnalyzer / AgingModel rich (deeputin0)
L3. face_morpher sequences (deeputin) — may need UI reintegration
L4. TransitionValidator ±30d backbone for profiles (v9)
L5. Advanced Mahalanobis anomalies (appold6)
L6. UMAP+HDBSCAN alpha clusters (appold2/3, AlphaStabilityTracker)
L7. Full biometric_texture forensic cascade (v9) — many 2D detectors; ears/hair EXCLUDED by design (user correction: 3DDFA can't compare ears/hair)
L8. Modular BayesianEngine+RulesEngine (old s5) unused branch
L9. Journalist AlternativeExplanations / BaselineReturnDetector richness

### ЗАПРЕЩЕНО / ВНЕ SCOPE 3DDFA
X1. Ears, hairline, neck as primary 3D mesh evidence (user + publications note)
X2. Publications as Bayes priors (contradicts aboutplatform + plan2 + good audits)
X3. Hardcoded named DOUBLE profiles in anomaly_engine (appnew444 fatal #4)
X4. Memes as classification labels (plan2 principle 7)

## OSINT taxonomy (различия) — интерпретационный слой, НЕ priors
- Doubles 1–3 + Udmurt + Vasilich (narrative eras)
- Masks 1–4 (symmetry, gloss, static nasolabial, deep orbits, flat profile…)
- «Нулевой пациент» = calibration ideal (natural pores, dynamic forehead lines, natural asymmetry 1–3%)
- 1081 ulik-items — mostly visual forensic cues; only subset mappable to 3DDFA metrics (chin contour, cheek volume, nose shape, skin smoothness, symmetry); ears/hair/neck/sclera pupils → 2D-only or drop

## pair_check buckets (старые имена)
frontal 268, L/R profile ~300 each, 3q light/mid/deep — подтверждает историю medium/mid/deep naming vs current LEFT/RIGHT_THREEQUARTER_MEDIUM

## Новые кандидаты ошибок (ITER7)
26. plan2.txt self-duplication (этапы 0–1 повторяются 4–5×) — риск двойного выполнения / путаницы AN-id
27. aboutplatform vs fuzzy_bayes hyp naming — documented architectural split (product TZ vs investigation mode)
28. future-fix synthetic floor + fractal miscalibration = root of «все в масках» UI
29. Audits disagree on OSINT: audit-4 wants OSINT→priors; audit-2/3/5/6/8 + aboutplatform + plan2 forbid it — **canonical: post-hoc only**
30. appnew444: pitch correction disabled (wrong direction) — may still haunt vertical metrics if code path survived
31. valid_metric_count>=15 drops profiles from etalon — pose-specific baselines mandatory

## Связь с предыдущими итерациями
- ITER4/6: multiple Bayesian engines → audit consensus explains WHY current/hand-tuned and fuzzy both fail scientifically
- ITER5: lost gold list refined by user (uv alive; ears/hair not 3DDFA gold)
- ITER2 error #1 pose buckets confirmed by pair_check old names

## Итог ITER7
Корпус ТЗ изучен. Главный конфликт проекта: **богатое forensic engineering vs circular scientific inference**. Золото идей = principles (pose-matched, visibility-first Umeyama, calibration noise floor, red-team alternatives, publications post-hoc, unsupervised clusters first) + broken-but-valuable modules (texture stack, blemish UV, fuzzy structure without era priors).

# Итерация 5 — генеалогия старых версий (dutinsav/old, 30 версий)

## project-info.txt (исходное ТЗ старой эпохи)
- Цель: анализ архива 1999-2025, гипотеза двойников/масок/грима, управление через GUI в настройках
- Метод: 3DDFA-V3 + LDM134 + AdaFace embeddings + 7 forensic-модулей
- ДВА режима: DIAGNOSTICS (КАЛИБРОВКА на Викторе — один человек, поворот головы, замер техпогрешности) + MAIN (боевой архив)
- ВАЖНО: философия "не запоминаем лицо Виктора, запоминаем вариативность метрик при смене ракурса" = точно текущая calibration-философия (новые 5 датасетов расширяют это на разных людей)
- Hardcoded paths как требование (теперь — источник багов с env-префиксами)

## Масштаб (py файлов / строк)
- v2 4/5406, v4 6/3133, v5 4/4207, v6 4/5400, v7 13/9473, v8 14/9955, v9 14/10348
- app 5/4765, app33 10/10423, app43 20/3167, appnew444=appnew666 13/8879 (дубль)
- appold0 4/5400, appold1 13/5488, appold2 17/7455, appold3 17/6982, appold4 12/8035, appold5 14/10308, appold6 15/10428
- deeputin 14/4974, deeputin0 20/8035, new 4/5615, oldapp 9/12985, pipelinesave 29/5104
- sVEVER 9/13262 (самый крупный монолит), sav 3/2306, test-new-ver 16/3213, unified_v1 14/10348 (=v9), uv_module 7/900

## Две эволюционные линии
1. МОНОЛИТ (v2->v9 / app* / oldapp / sVEVER): app.py + biometric_texture.py + face_metrics.py + anomaly_engine.py + geometry_normalization.py + reporting.py. Нарастали: cluster_engine (v7+), multi_view_datamodel (v7+), transition_validator (v9/appold6/unified_v1), advanced_anomaly (только appold6). unified_v1 == v9 побайтно (10348) — финал монолитной линии.
2. МОДУЛЬНАЯ (deeputin0 -> deeputin -> pipelinesave -> current project/): __init__, разделённые extract/compare/fusion/temporal/spectral. pipelinesave (29 файлов) = прямой предок текущего staged-пайплайна (handoff/export_contract/downstream_bundle/zones.py/visibility.py/reconstruction.py/alignment.py/pose_lookup.py).

## ПОТЕРЯННОЕ ЗОЛОТО (есть в старых, нет/урезано в текущей)
- **biometric_texture.py (v9, ~1900 строк) — КОРОНА**: 13+ forensic-детекторов: EarForensics (каскад ушей — уши уникальны как отпечаток!), HairlineForensics, VeinForensics (сосуды), ForeheadForensics, EyelidForensics, CrowsFeetForensics (гусиные лапки), SpectralForensics, ScleraForensics (склера глаза), BRDFForensics (отражение кожи), AlbedoForensics + detect_grid_artifacts, ProfileForensics, MoleDetector (родинки!), detect_pores_blob (с age_years), detect_vascular_pattern, analyze_wrinkle_complexity. ПРОВЕРИТЬ что из этого выжило в текущей texture-подсистеме (V5 = 4 метрики + physical). Многое похоже ПОТЕРЯНО (уши, склера, родинки, гусиные лапки, hairline).
- **deeputin0/spectral.py**: спектральный анализ графа меша (build_laplacian, LCC, spectral_entropy, spectral_centroid) — геометрия через собственные числа Лапласиана. УНИКАЛЬНО, нет в текущей.
- **deeputin0/temporal_analysis.py**: AgingModel, TemporalDriftDetector, PerViewTrajectoryAnalyzer — модель старения по-ракурсно. Часть идей перешла в chronology, но PerView-траектории ПОТЕРЯНЫ.
- **deeputin0/fusion.py**: compute_fusion_score + compute_anomaly_verdict + _compute_confidence — ещё один (четвёртый!) движок вердикта.
- **uv_module (7 файлов)**: HD UV текстуры 2048px: delight (albedo_from_texture — убрать освещение!), UVDetailEnhancer, UVBeautyPostprocessor, UVBaker (barycentric). КРИТИЧНО для сравнения текстур без влияния света. УНИКАЛЬНО.
- **deeputin/face_morpher.py**: FaceMorpher — morph sequence между двумя 3D лицами + видео (create_morph_sequence, create_video_morph). Это морфинг из ТЗ (UI). ПОТЕРЯНО в текущей.
- **transition_validator (v9)**: профили требуют backbone-anchor в ±30 дней (анти-false-match для ненадёжных профилей). УМНАЯ идея — сравнить с anchor-логикой текущего s3.
- **appold6/advanced_anomaly.py**: _mahalanobis_scores (махаланобис к среднему) — многомерный выброс.
- **appold2/appold3 ml_pipeline.py**: UMAP + HDBSCAN(min_cluster_size=20) кластеризация embeddings — прародитель alpha_tracker HDBSCAN. В текущем alpha_tracker есть, но UMAP-шаг ПОТЕРЯН.

## Кандидаты в ошибки (ITER5)
20. appnew444 == appnew666 и v9 == unified_v1 — точные дубли-копии (можно удалить для экономии).
21. profile_left/profile_right в transition_validator (старое имя) vs left_profile/right_profile в текущей schemas — подтверждает кандидат #1 (рассинхрон имён bucket тянется из старых версий).
22. 4 независимые реализации вердикта в истории: anomaly_engine (монолит), deeputin0/fusion, старый s5 hand-tuned, модульный BayesianEngine — ни одна не калибрована на данных, все на ручных весах.

## ПОПРАВКИ ОТ ПОЛЬЗОВАТЕЛЯ (после ITER5)
- 3DDFA-V3 НЕ умеет сравнивать уши и волосы — поэтому Ear/Hairline детекторы вырезаны ОСОЗНАННО. Нюанс: старые детекторы работали каскадами по 2D-фото (не по мешу), так что теоретически возродимы как независимый 2D-канал, но это решение пользователя, не ошибка.
- uv_module ИСПОЛЬЗУЕТСЯ в текущей версии — просто не был отправлен в архиве. Убрать из "потерянного". Значит в архиве кода не хватает и других рабочих модулей — уточнять перед выводами "потеряно".

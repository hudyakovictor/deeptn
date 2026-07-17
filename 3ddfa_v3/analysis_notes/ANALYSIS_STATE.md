# DEEPUTIN — журнал анализа (восстановлен после сброса песочницы)

## ВАЖНО: состояние песочницы
- Песочница сбрасывалась 13.07. Утеряны: /data/current_small (код текущей версии!), старый журнал.
- Восстановлено: /data/current_big (5 датасетов person_01..05, из вложения 115MB), /data/dutinsav_light, /data/archive_extract.
- НУЖНО ОТ ПОЛЬЗОВАТЕЛЯ: заново прислать маленький архив с кодом текущей версии (~1.3MB).

## Статус итераций
- [x] ИТЕР 1: обзор текущей версии, aboutplatform.txt, pipeline.yaml, run.py
- [x] ИТЕР 2: S1 Extraction (2049 метрик, policy, texture V5, quality_gate)
- [x] ИТЕР 3: S2 Identity + S3 Compare (текущая версия)
- [x] ИТЕР 4: Verdict + Report — по СТАРОЙ версии (dutinsav/project/s5_verdict, s6_report — прямой предок текущих s4/s5). Diff с текущей версией — после повторной загрузки кода.
- [ ] ИТЕР 5: генеалогия старых версий (v2..v9, appold0-6, app33, sVEVER, unified_v1, deeputin0)
- [ ] ИТЕР 6: newapp backend (fuzzy_bayes, zones.py 152K, pipeline_runner)
- [ ] ИТЕР 7: корпус ТЗ/заметок -> реестр идей (реализовано/потеряно)
- [ ] ИТЕР 8: кросс-анализ идеи-код, pose-coverage датасетов person_01..05
- [ ] ИТЕР 9: финальный отчет исследования

## ИТЕР 4 — Verdict/Report (старая версия как прокси)

### Как работает VerdictEngine (s5_verdict/engine.py, 625 строк)
- _verdict_for_photo: агрегирует пары фото -> avg geometry/texture/same/diff/syn suspicion.
- Байес: priors H0=0.52 H1=0.18 H2=0.20 HU=0.10; likelihood-формулы вручную (exp(-...) для H0).
- silicone_physical_boost = sss*0.30 + specular_sharpness*0.20 + seam*0.25 + norm_spectral*0.15 + lbp*0.10 + wrinkle_aniso*0.10.
- Множители с капом 1.35x/0.65x (FIX #9): chronology flags, baseline_return (H1 x1.25), alpha_anomaly (H1 x1.20).
- Много FIX-комментариев: FIX #2 (пустые пары -> нейтральные значения, а не 0 в пользу H0), FIX #3 (инвертированный quality_penalty), FIX #5/#6 (ненормализованный spectral_slope, отрицательный sss), FIX #8 (identity_hint по умолчанию UNKNOWN, не PUT), FIX #10 (убран x2 у cross-modal).

### Сильные находки (идеи, которые стоит сохранить)
- AlphaStabilityTracker: HDBSCAN-кластеризация 199-D alpha identity векторов во времени; прыжки между кластерами = аномалия. СИЛЬНАЯ идея.
- BaselineReturnDetector: метрика отклонилась и вернулась к baseline 5+ летней давности = невозможно при старении = маркер маски.
- BiologicalConstraintChecker: лимиты хирургии (ринопластика 180д, фейслифт 90д, блефаро 60д, импланты 365д, кость <=0.5мм/год, инверсия асимметрии 5 лет).
- H1SyntheticDetector: texture + geometry (маска на похожем черепе) + cross-modal + prosthetic zones (скуловые дуги, переносица, угол челюсти — зоны натяжения маски).
- RulesEngine: 9 читаемых правил (R_SYNTHETIC_GEOMETRY_STABLE 0.9, R_GEOMETRY_AGE_MISMATCH 0.85 и т.д.) + warning_level.
- journalist_engine AlternativeExplanations: для каждой аномалии проверяет mundane-объяснения (ракурс <5, возраст <2 лет, качество, грим/освещение) — форензик-грамотно.
- ChronologyAnalyzer: BIOLOGICAL_LIMITS + BONE_METRICS набор костно-стабильных метрик.

### Кандидаты в ошибки (ITER4)
15. ДВА байесовских движка в verdict: engine.py считает Байес ин}

## ИТЕР 9 — итоговый отчёт
- Завершена консолидация всего доступного корпуса без переписывания кода.
- Итог: `/data/analysis_notes/ITER9_final_study_report.md`.
- Зафиксировано ограничение: после reset отсутствует `/data/current_small`; current S4/S5 требуют повторной сверки после загрузки малого архива.
- Статус: изучение доступных материалов завершено; переход к реализации не авторизован.

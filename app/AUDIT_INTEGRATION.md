# Интеграция рекомендаций итогового аудита

## Реализовано в app6 v1.5

- Основа Stage 1 app5 сохранена: single inference, identity-only, renderer/combined visibility, observed/original/confidence UV, hashes, atomic commit и validator.
- Dense mesh, point-to-plane, anatomical zone support, shape proxies и mesh calibration уже присутствовали в app6.
- Baseline-return, multiple-testing, quality integration, evidence packets, private Stage 2B corroboration и HTML Stage 3 уже присутствовали.
- Texture-канал расширен до LBP, GLCM и frequency descriptors с optional scikit-image и NumPy/OpenCV fallback.
- Добавлен iteratively trimmed Kabsch для landmarks, identity-only landmarks, local descriptors и dense mesh. Сохраняются fit/trim counts и residual before/after.
- Добавлена независимая cross-bin corroboration: она работает только после фиксации same-bin результатов и не меняет первичные residuals или thresholds.
- Добавлена агрегация сравнений по событиям/датам с явным учётом известных source groups. Плоская папка не выдаётся за независимые источники и получает `source_unknown`.
- Нейтрализована терминология темпа: canonical статусы `rapid_change_candidate` и `persistent_rapid_change_candidate`; старые `biological_rate_*` оставлены только как compatibility aliases.
- Evidence/report artifacts дополнены `cross_bin_corroboration.json` и `event_aggregation.csv`.

## Не перенесено намеренно

- persona priors и заранее назначенные эпохи;
- silicone probability;
- H0/H1/H2 с ручными likelihood;
- identity labels и автоматические выводы о подмене;
- prior-driven weighting первичных измерений;
- сравнение невидимых зон.

## Требует реальных production-данных

- cluster-level significance dense mesh относительно per-vertex calibration null;
- camera/codec strata (нужны надёжные metadata/source labels);
- crop-jitter ensemble с настоящим 3DDFA inference;
- ridge/skeleton/Skan temporal branch matching морщин;
- клинический biological reference layer.

Эти пункты нельзя честно считать завершёнными на синтетике или трёх примерах: для них необходимы веса модели, основной набор, полная same-person calibration и независимые source/event metadata.

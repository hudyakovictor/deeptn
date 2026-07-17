# Итерация 6 — newapp/backend (fuzzy_bayes + core + pipeline)

## fuzzy_bayes (9019 строк, 24 файла) — САМЫЙ ЗРЕЛЫЙ вердикт-движок в истории проекта (5-й по счёту)
- Гипотезы ИМЕНОВАННЫЕ по носителям: H_ORIGINAL / H_UDMURT / H_VASILICH / H_UNCERTAIN (богаче текущего H0/H1/H2!)
- Приоры ПО ЭРАМ (ERA_1_BASELINE: original 0.68 -> ERA_3_UDMURT: original 0.40, udmurt 0.28...), профили default/investigation, versioned deterministic
- Нечёткая логика: triangular/trapezoidal membership; лингвистические переменные geom_ratio (acceptable/unlikely/suspicious/impossible), silicone (natural/elevated/synthetic с деградацией по шуму), snr, reliability, aging, id_match; коридоры смягчаются от quality_factor
- rules.py: RuleHit -> распределение массы по гипотезам + русские объяснения (i18n); правила: identity match/mismatch по 3DDFA id-векторам, geom impossible, texture synthetic (с разветвлением: на геом-расхождении = протез-модификатор, на совпадении = surface confound), Mask-vs-Mask (сильное совпадение геометрии + синтетическая текстура!), aging envelope, low reliability -> uncertain (но geometry_decisive override)
- posterior.py: log-space байес, carrier_mass, apply_carrier_ceiling, eliminate_impossible
- decision.py: carrier publication guard, cross-bucket consensus, взвешенный консенсус бакетов
- inference_store, schema, registry, i18n — продакшн-обвязка (версионирование, git_commit в runner)

## Эпистемическая гигиена (ЦЕННО, сохранить принципы)
- publication_correlation: "Publications are post-hoc alignment only — must never modify Bayes priors"
- investigation_eras: эры = EXPLORATORY метки, НЕ приоры; границы валидирует СЛЕПОЙ DBSCAN/persona-анализ
- cascade.py: симметричная агрегация текстурных флагов "избавляемся от токсичного max()" [BUGFIX в комментарии]
- peer_cohort: z-score относительно pose-matched соседей в том же бакете
- texture_live_reference: эталон "живая кожа" = calibration HQ + взвешенный main 1999-2002 (с guard от загрязнения синтетикой)

## pipeline/ (предок текущего s1) — модули, КОТОРЫХ НЕТ в присланном архиве текущей версии
- blemish_tracking.py: трекинг родинок в UV-пространстве + compare_blemish_stability (идея MoleDetector ВЫЖИЛА здесь!)
- periocular_metrics.py: метрики контура глаз (arc/chord/poly-residual/tangent)
- skin_authenticity/ (7 файлов: patches, fusion, scorer, calibration)
- mask_anomaly/ (scorer + baseline_cache + calibration_stats.json)
- person_baseline.py, zones.py 760 (bone_priority_core/supporting, expression exclusion mask)
- СПРОСИТЬ ПОЛЬЗОВАТЕЛЯ: живы ли эти модули в текущей версии (как uv_module) или действительно отброшены

## Кандидаты в ошибки (ITER6)
23. В rules.py масса правил распределяется в "H_CARRIER_SWITCH", но HYPOTHESES = (H_ORIGINAL, H_UDMURT, H_VASILICH, H_UNCERTAIN) — гипотезы "H_CARRIER_SWITCH" НЕТ в конфиге. Проверить merge_rule_hits/normalize: куда уходит эта масса — возможно молча теряется/нормализуется в ноль.
24. Текущая версия ОТКАЗАЛАСЬ от fuzzy_bayes в пользу более примитивного hand-tuned байеса — возможно главная регрессия проекта: потеряны эрные приоры, carrier-гипотезы, Mask-vs-Mask логика, русские объяснения, carrier ceiling, eliminate_impossible.
25. mesh_delta.py — legacy shim (указывает на pipeline/compare.compute_forensic_vertex_deltas) — ещё один след миграций.

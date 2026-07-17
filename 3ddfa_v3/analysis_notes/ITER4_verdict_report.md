# Итерация 4 — Verdict + Report + shared (по старой 6-стадийной ветке dutinsav_light)

## ВАЖНО: сброс песочницы
- current_small (КОД текущей версии) и /data/analysis_notes были стёрты.
- В вложениях остался только большой архив датасетов (→ /data/current_big) + dutinsav_light + archive_extract.
- Статус итераций 1-3 (S1/S2/S3 текущей версии) уже в памяти; чтобы читать КОД текущей версии дальше — нужно заново залить маленький zip с кодом.

## Старый s5_verdict (= предок текущего s4_verdict), читал как прокси
### VerdictEngine._verdict_for_photo (активный путь)
- priors H0/H1/H2/HU = 0.52/0.18/0.20/0.10
- likelihoods = ручные экспоненты/clip по avg_geom, avg_tex, chronology, silicone_boost, silicone_physical_boost, h1_result
- Много FIX-комментариев (след ручной отладки):
  - FIX#2 empty pairs -> neutral (1.0/0.5/0.5), а не 0.0 (раньше был bias в H0)
  - FIX#3 quality_penalty был ИНВЕРТИРОВАН (низкое качество -> высокий penalty -> рост H1)
  - FIX#5 spectral_slope был НЕНОРМАЛИЗОВАН (2-4 * 0.15)
  - FIX#6 sss_index мог быть ОТРИЦАТЕЛЬНЫМ -> снижал H1
  - FIX#8 identity_hint default "PUT"->"UNKNOWN" (раньше bias в H0)
  - FIX#9 кап cumulative boost 1.35x / reduce 0.65x (раньше 1.18*1.25*1.20=1.77x разгон H1)
  - FIX#10 (h1_engine) убран *2 множитель cross-modal

### Кандидаты в ошибки (ITER4)
15. ДВОЙНОЙ БАЙЕС: BayesianEngine+ProbLogic+RulesEngine (modules/) — полноценный структурированный байес (priors от evidence, log-space, rule_bias, warning_level) НЕ вызывается VerdictEngine. Движок вместо этого считает priors/likelihoods вручную в _verdict_for_photo. Две параллельные байес-реализации, активна худшая (магические числа). rules_engine содержит 9 читаемых правил — потерянный ценный актив.
16. h1_engine.py: import CrossModalTextureRules из s2_metrics, но использует TextureRules()/GeometryRules() без import -> NameError при вызове. Класс H1SyntheticDetector, вероятно, нерабочий/недописан.
17. chronology.py METRIC_ALIASES: старые имена (bone_nasion_depth...) -> zone_* из актуального catalog. BONE_METRICS в chronology СТАРЫЕ имена (bone_orbit_L_depth, mesh_symmetry_x, id_norm) — если не прогоняются через _resolve_metric, био-проверки молча не срабатывают (ключей нет в данных).
18. biological_limits max_change_ratio коммент "30% relative (metrics are ratios, not mm)", но chronology BIOLOGICAL_LIMITS в мм (max_bone_change_mm_per_year 0.5, ..._short_gap 2.0) и описания "отклонилась на X мм" — путаница единиц мм vs ratio между модулями.
19. alpha_tracker HDBSCAN(min_cluster_size=5) на 199-D alpha — ключевая идея (один человек = стабильный кластер), но требует alpha-векторов из 3DDFA — проверить, передаются ли они реально (в текущей версии могло отвалиться).

## Ценные идеи (собрать в реестр)
- baseline_return: метрика отклонилась и вернулась к baseline 5+ лет = невозможно при старении -> маркер маски (сильная идея)
- alpha-кластеризация во времени (прыжки между кластерами)
- био-лимиты хирургии (рино/фейслифт/блефаро с min_gap заживления)
- journalist_engine.AlternativeExplanations: авто-проверка mundane-объяснений (ракурс/возраст/качество/грим) — отличный анти-false-positive слой

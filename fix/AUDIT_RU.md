# Первичный аудит DEEPUTIN

Дата аудита: 2026-07-12  
Репозиторий: `hudyakovictor/deeptn`, commit `0181e46`

## Краткий вывод

В текущем состоянии пайплайн нельзя считать корректно работающим или пригодным для интерпретации результатов расследования.

Проблема не сводится к одной ошибке. Одновременно присутствуют:

1. невоспроизводимое окружение и отсутствующая часть 3DDFA-V3;
2. ошибки оркестрации и конфигурации;
3. разрывы контрактов данных между S1–S5;
4. фактически неработающая pose-aware калибровка;
5. дублирование сравнений и потеря quality/expression metadata;
6. статистически некалиброванные вероятности;
7. генерация отчётом неподтверждённых дат, пар и единиц измерения;
8. методологический bias: априорная вероятность силикона зависит от эпохи, которую система должна исследовать объективно.

**Существующие `report.json`, `report.md` и `journalist_report.json` пока нельзя использовать как доказательную основу.**

---

## Что было проверено

- структура репозитория и корневое ТЗ `aboutplatform.txt`;
- CLI и конфигурация;
- импорты и компиляция Python;
- схемы JSON и переходы S1 → S2 → S3 → S4 → S5;
- калибровка ракурса и качества;
- попарное сравнение;
- хронологический анализ;
- Bayesian/Platt-слой;
- генерация журналистского отчёта;
- pytest collection;
- санити-тест S2–S5 на синтетическом архиве с известным ground truth.

### Фактические результаты запуска

| Проверка | Результат |
|---|---|
| `python -m compileall -q project core` | Проходит: синтаксических ошибок нет |
| `python -m project.run --help` | Падает на импорте: `ModuleNotFoundError: torch` |
| Валидация `project/config/pipeline.yaml` | Собственная конфигурация отклоняется: stage `s2` объявлен invalid |
| `pytest --collect-only` | 3 ошибки collection, тесты не собираются |
| Наличие requirements/pyproject/environment | Отсутствуют |
| Наличие `core/3ddfa_v3` | Отсутствует и исключено через `.gitignore` |

---

# P0 — блокирующие ошибки

## P0.1. Репозиторий невозможно развернуть с нуля

В репозитории нет:

- `pyproject.toml`;
- `requirements.txt` или lock-файла;
- `environment.yml`;
- инструкции по установке;
- кода и assets `core/3ddfa_v3`.

При этом `project/s1_extraction/modules/reconstruction.py` импортирует `torch` и ожидает:

- `demo`;
- `face_box`;
- `model.recon`;
- `assets/face_model.npy`.

Даже после установки PyTorch S1 не запустится без внешней папки 3DDFA-V3 и весов.

Корневой `run_pipeline.py` дополнительно жёстко привязан к одной машине:

- `/opt/homebrew/Caskroom/miniconda/base/envs/deeputin/bin/python`;
- `/Users/victorkhudyakov/deeputin`;
- `/Volumes/SDCARD/...`.

### Что исправить

- добавить `pyproject.toml` и lock-файл;
- оформить 3DDFA-V3 как submodule, отдельный setup-скрипт или документированную внешнюю зависимость;
- добавить preflight-проверку assets/весов до запуска обработки;
- убрать абсолютные пути из кода;
- разделить зависимости на base и `s1/ml`, чтобы S2–S5 и `--help` могли работать без загрузки PyTorch.

---

## P0.2. Корневой runner использует не тот calibration input

`run_pipeline.py` передаёт:

```python
--input-calibration str(main_output)
```

То есть S1-калибровка читает результаты основного датасета, а не `/Volumes/SDCARD/photo/calibration` из ТЗ.

Это делает calibration reference пустым или загрязнённым основным набором. Ground-truth «один известный человек» теряется.

### Что исправить

Передавать реальный каталог калибровочных фото отдельным аргументом/настройкой. Никогда не использовать output основного набора как calibration input.

---

## P0.3. Собственный YAML не проходит собственную валидацию

`project/config/pipeline.yaml` содержит stages:

```yaml
[s1, s2, s3, s4, s5]
```

Но `project/shared/config_validation.py:36` разрешает:

```python
{"s1", "s3", "s4", "s5", "s6"}
```

В результате `s2` отклоняется, а несуществующий в CLI `s6` считается допустимым.

Дополнительно:

- `paths` и `stages` из YAML runner фактически не применяет;
- `${DPTN_DATA_ROOT:-data}` — shell-синтаксис, который `yaml.safe_load` не раскрывает;
- параметры в validator сдвинуты по старой нумерации: настройки compare проверяются как `s4`, хотя runner передаёт их в `s3`.

### Что исправить

Создать единую typed-конфигурацию и один список stages. Валидировать уже раскрытые пути. CLI override должен иметь явный приоритет над YAML.

---

## P0.4. S2 теряет реальные углы калибровочных фото

В `project/s2_identity/engine.py:37-44` код создаёт calibration data из `Stage2Record`, но читает:

```python
record.pose.yaw
```

У `Stage2Record` поля `pose` нет. Из-за `hasattr` ошибка маскируется, и для **каждой** фотографии сохраняются:

```python
yaw = pitch = roll = 0
```

Именно pose-aware компенсация является центральным требованием ТЗ, но сейчас модель строится без реального pose gap.

Более того, `pose_calibration_models.json` сохраняется, однако основной `CompareEngine` его вообще не загружает. Он использует только агрегированный `pairwise_noise`, не зависящий от фактической разницы углов конкретной пары.

### Что исправить

- соединять `Stage2Record` с соответствующим `Stage1Record` и брать `stage1.pose`;
- хранить pose/quality provenance в едином artifact schema;
- загружать pose model в S3;
- вычислять ожидаемый шум отдельно для каждой метрики и конкретных `yaw/pitch/roll/quality` пары;
- при отсутствии здоровой calibration bucket выдавать `UNCERTAIN`, а не нулевой шум.

---

## P0.5. Между этапами теряется значительная часть Stage2

S1 формирует богатый `Stage2Record` и записывает `stage2_manifest.json`, включая:

- `metric_notes`;
- `texture_assessability`;
- `quality_summary`;
- geometry/skin hints;
- anomaly metadata.

Но S2, S3, S4 и S5 игнорируют manifest и каждый раз заново создают урезанный `Stage2Record` только из:

- `info.json`;
- `geometry_metrics.json`;
- `texture_metrics.json`.

Поля возвращаются к default-значениям. Следствия:

- `texture_unreliable` в S4 практически всегда считается `False`;
- physical features ищутся в `metric_notes`, который уже потерян;
- geometry identity hint теряется;
- texture assessability теряется;
- downstream не знает, была ли метрика реально измерена, gated или восстановлена fallback-ом.

### Что исправить

Использовать единый versioned artifact reader и `stage2_manifest.json` как источник Stage2. Отдельные JSON метрик должны быть производными/просмотровыми файлами, а не альтернативной схемой.

---

## P0.6. Ошибки S1 маскируются как успешное завершение

`ExtractionEngine.run()` ловит исключение для каждого фото, логирует его и продолжает, но не возвращает runner-у число ошибок и не выбрасывает итоговое исключение.

`PipelineRunner` после этого пишет:

```text
S1 complete
```

даже если не обработано ни одного фото. Пустой calibration reference также только warning. Поэтому процесс может завершиться с exit code 0 при фактически неработающем анализе.

### Что исправить

Ввести `StageResult`:

```text
processed / skipped / failed / warnings / fatal
```

и строгие критерии:

- 0 успешно обработанных main photos → fatal;
- 0 calibration photos → fatal для S2–S4;
- missing reference → fatal для calibrated mode;
- доля failed выше порога → non-zero exit code.

---

## P0.7. Неверная интеграция TextureSkinClassifierV5

`TextureSkinClassifierV5.__init__` принимает boolean `quality_compensation`, но код передаёт туда путь `texture_leaderboard`:

```python
TextureSkinClassifier(texture_leaderboard)
```

Путь становится truthy boolean. Никакая модель или leaderboard при этом не загружается.

При вызове classifier код не передаёт `pose` и `year`, поэтому заявленные:

- yaw compensation;
- era handling

не используют данные конкретного снимка.

Одновременно V5 содержит априорные вероятности:

```python
pre_2012 = 0.05
2012_2021 = 0.40
post_2021 = 0.60
```

с комментариями, что ранний период — «original», поздний — «silicone likely». Это встраивает расследуемую гипотезу в сам классификатор и создаёт круговое доказательство. Для объективного анализа такой prior недопустим без внешнего, заранее зарегистрированного и независимо валидированного основания.

### Что исправить

- исправить constructor contract;
- убрать hypothesis-dependent era prior из основного результата;
- использовать эпоху только для domain/quality correction, обученной на независимых данных;
- выдавать отдельно `raw texture evidence` и `prior-adjusted scenario`, не смешивая их;
- обязательно сохранять версию модели, training dataset hash и calibration metrics.

---

## P0.8. Отчёт выдумывает единицы, даты и evidence links

`project/s5_report/engine.py:367-374` преобразует нормализованный geometry score в миллиметры:

```python
"delta_mm": evidence["geometry"] * 100
```

Но `geometry` — агрегированный normalized distance/z-like score, не физическая длина. Такое преобразование математически недопустимо.

Также отчёт создаёт:

- `date = "N/A"`, хотя дата есть в `info.json`;
- искусственный pair id `photo_id__anchor`, которого обычно нет в `pairs.json`;
- для chronology `date_a == date_b` и `gap_days = 0`;
- фразу «биологически невозможно» без ссылки на реальную пару и измерение.

В санити-тесте это породило формулировки о различиях порядка **86 000 мм**.

### Что исправить

- никогда не подписывать normalized score как mm;
- физические единицы использовать только при доказанной scale calibration;
- каждый тезис должен содержать существующие `photo_id`, `pair_id`, обе даты, raw values, normalization, threshold, quality и uncertainty;
- если evidence link не разрешается, тезис не публиковать;
- заменить категоричные утверждения на уровни поддержки гипотезы.

---

# P1 — критические ошибки логики анализа

## P1.1. S3 создаёт дубли пар и повторно учитывает доказательство

Одна и та же пара может попасть как adjacent и как anchor comparison. Дедупликации нет, pair ID одинаковый.

В санити-тесте:

- записано 64 pair rows;
- уникальных pair IDs — 58;
- 6 пар посчитаны дважды.

Эти повторы попадают в `pair_index` и смещают средние evidence в S4.

Также расчёт total progress неверен: было заявлено 68, обработано 64, progress остановился на 94%.

### Что исправить

Строить множество канонических пар `(min(photo_a, photo_b), max(...))`, хранить reason list (`adjacent`, `anchor`) и считать пару ровно один раз.

---

## P1.2. Хронологический анализ смешивает разные ракурсы

`ChronologyAnalyzer.build()` сортирует все фото в одну временную последовательность. `_series()` строит один ряд метрик, не разделяя девять pose buckets.

В итоге соседними точками ряда могут быть frontal и profile, хотя ТЗ прямо требует сравнивать группы ракурсов отдельно.

Дополнительные проблемы:

- список метрик берётся только из первой записи;
- пропуски превращаются в NaN и могут незаметно отключить детекцию;
- `rate = delta / gap_days` затем делится на baseline из **не нормированных по времени delta**, то есть величины имеют разные размерности;
- thresholds `0.75`, `0.35`, `2 mm`, `3 mm` применяются к разным нормализованным ratio без единой системы единиц.

### Что исправить

Строить chronology отдельно по bucket, а внутри bucket учитывать непрерывный pose residual. Использовать metric-specific robust rate models и calibration intervals.

---

## P1.3. «Platt calibration» фактически не обучена

`VerdictEngine` создаёт `FittedPlattCalibrator`, но не вызывает `fit()` и не загружает параметры. Используется heuristic fallback.

После предсказания `calibration_report.json` строит `y_true` из **тех же предсказанных labels**:

```python
y_true = 1 if predicted hypothesis == H0 else 0
```

Это не ground truth и не измеряет calibration quality.

При этом итоговый report заявляет, что вероятности калиброваны через Platt scaling.

### Что исправить

- до появления размеченного validation set не называть scores вероятностями/posterior;
- хранить их как `support_score`;
- Platt/isotonic fit выполнять только на независимом ground truth;
- ECE/MCE считать только по реальным labels.

---

## P1.4. «Bayesian» likelihoods и priors не подтверждены данными

Priors H0/H1/H2/HU и большинство коэффициентов заданы вручную. Likelihood expressions не являются оценёнными распределениями `P(E|H)`.

Такие значения можно использовать как экспериментальный rules score, но нельзя интерпретировать как судебно значимые posterior probabilities.

### Что исправить

Переименовать текущий слой в heuristic evidence fusion. Для Bayesian модели нужны независимые H0/H1/H2 datasets, likelihood distributions, cross-validation и sensitivity analysis по priors.

---

## P1.5. Geometry и texture distances загрязнены служебными метриками

В geometry comparison могут участвовать:

- `pose_yaw/pitch/roll`;
- `mesh_vertex_count`;
- абсолютные bbox dimensions;
- expression magnitude;
- visibility ratio.

То есть различие позы может одновременно:

1. считаться identity difference;
2. затем частично компенсироваться общим calibration noise.

В texture distance участвуют quality и classifier output fields (`overall_quality`, `noise_level`, `texture_*_prob`), смешивая физическую текстуру, качество изображения и уже готовый вердикт classifier.

### Что исправить

Ввести versioned allowlists:

- identity geometry;
- soft-tissue geometry;
- pose nuisance;
- image quality;
- raw texture;
- classifier outputs.

Нельзя повторно подавать classifier posterior как независимую raw texture feature.

---

## P1.6. Expression-aware исключение зон фактически не работает

S1 записывает только три boolean:

```text
smile_excluded / jaw_excluded / neutralized
```

S3 ожидает структуру с `intensities` и `excluded_zones`. Поля `exp_vector` в `Stage1Record` нет. Поэтому `_get_expression_flags()` обычно возвращает `None`.

GeometryExtractor также считает все зоны до применения smile/jaw exclusions.

### Что исправить

Сохранять нормализованный expression vector, confidence и явный список исключённых зон. Применять mask до aggregation каждой geometry metric и записывать effective metric coverage.

---

## P1.7. Координаты и metadata reconstruction несогласованы

- `_reconstruction_to_dict()` кладёт `id_params` и `exp_params` в top-level;
- `GeometryExtractor` ищет их внутри `reconstruction["payload"]`, поэтому id/exp metrics не извлекаются;
- `image_shape` строится из shape массива вершин, а не из размеров изображения;
- bbox fallback использует 512×512, потому что у `ReconstructionResult` нет `image_size`;
- physical feature extractor получает landmarks исходного изображения вместе с resized crop 424×500, то есть системы координат не совпадают;
- normal path сохраняет/читает BGRA через OpenCV, а часть downstream кода называет массив RGBA/RGB.

### Что исправить

Определить явные coordinate spaces и transform chain:

```text
original pixels → detector crop → model 224 → canonical 3D → output crop 424×500 → UV
```

Каждый массив landmarks/vertices должен иметь поле `space` и transform matrix.

---

## P1.8. Calibration age profile построен на неверном возрасте

S1 всегда рассчитывает возраст от даты рождения Владимира Путина, в том числе для пользовательского calibration dataset.

Таким образом, личные calibration photos получают не возраст реального calibration subject. Age profile из такого набора не имеет заявленного смысла.

### Что исправить

- хранить birthdate/age только для main subject;
- calibration noise model отделить от biological ageing model;
- ageing model строить на независимых longitudinal datasets, а не на личных calibration photos пользователя.

---

## P1.9. Неправильные семантики внутри H1 detector

S4 передаёт:

- `geometry_distance` как `excess_distance`;
- `texture_distance` как `heatmap_mean`;
- максимальный geometry distance как `heatmap_max`;
- количество пар с geometry distance > 1 как `bone_zone_violations`.

Это не heatmap и не число нарушенных анатомических зон. На этих подменённых полях H1 detector формирует тезисы о «натяжении протеза».

### Что исправить

H1 должен получать реальные per-zone mesh residuals с единицами/scale и coverage. Если heatmap не рассчитан — соответствующее evidence должно быть `missing`, а не заполнено другим score.

---

# P2 — инженерные риски

## P2.1. Тесты не автоматизированы

Файлы с `test_*.py` являются преимущественно debug scripts и требуют локальных моделей/путей. Нет unit/integration tests для основных контрактов S1–S5.

Symlink `deeputin -> project` приводит к двойному pytest collection одних и тех же файлов.

Нужны тесты минимум для:

- config load/override;
- schema round-trip;
- missing artifact handling;
- calibration pose propagation;
- pair deduplication;
- bucket-separated chronology;
- no-evidence → UNCERTAIN;
- no conversion normalized score → mm;
- deterministic report links;
- non-zero exit on stage failure.

## P2.2. Дублирующиеся реализации

Есть параллельные версии:

- `engine.py` и `engine_v2.py`;
- два `InlineMetricsExtractor`;
- несколько texture classifiers V1/V3/V4/V5;
- два texture extractors;
- несколько calibration/noise modules;
- старая нумерация stages S3/S4/S5/S6 в логах.

Неясно, какая версия authoritative. Это уже привело к несовместимым constructor contracts и старой валидации config.

## P2.3. Слишком широкое подавление исключений

Во многих местах используются `except Exception`, `pass` и silent fallback. Научно важная метрика может исчезнуть, но пайплайн продолжит работу с `0.0`, `{}` или neutral defaults без явного снижения доверия.

Нужно различать:

- unavailable;
- not assessable;
- numerical failure;
- out of domain;
- measured zero.

---

# Результат санити-теста

Был создан небольшой воспроизводимый набор:

- 8 calibration records одного условного человека;
- 14 main records того же условного человека;
- только в 2 main records искусственно внесена сильная аномалия;
- известные yaw, даты, quality, geometry и texture features.

Из-за eager-import S1 тест S2–S5 пришлось запускать с audit-only обходом импорта `torch`; production-код не менялся.

Результат:

- identities: 14;
- pair rows: 64;
- unique pair IDs: 58;
- verdicts: 9 × `H2_DIFFERENT`, 5 × `H_UNCERTAIN`, 0 × `H0_SAME`;
- report объявил 13 critical findings;
- normalized geometry score был представлен как сотни/тысячи/десятки тысяч миллиметров;
- альтернативные объяснения содержали нераскрытые placeholders `{pose_gap}`, `{age_gap}`, `{quality}`.

Это не benchmark качества модели, но достаточный regression/sanity failure: заведомо same-person baseline не должен превращаться в 0 H0 и физически невозможные единицы.

---

# Рекомендуемый порядок исправления

## Этап 1. Сделать запуск воспроизводимым

1. `pyproject.toml` + lock;
2. documented 3DDFA setup;
3. portable CLI без абсолютных путей;
4. preflight;
5. lazy imports по stages;
6. корректный calibration input;
7. строгие exit codes.

## Этап 2. Зафиксировать контракты данных

1. schema version во всех artifacts;
2. один `Stage2Record` reader;
3. pose, quality, assessability, expression и provenance не должны теряться;
4. explicit missing/error states;
5. JSON Schema tests между stages.

## Этап 3. Починить calibration и compare

1. реальные pose из Stage1;
2. bucket + continuous pose residual;
3. per-metric noise functions;
4. quality-conditioned intervals;
5. metric allowlists;
6. pair deduplication;
7. expression-zone exclusion;
8. unknown/insufficient calibration → UNCERTAIN.

## Этап 4. Переделать chronology

1. отдельный ряд для каждого bucket;
2. корректная размерность rates;
3. metric-specific biological limits;
4. сравнение только при достаточном overlap/quality;
5. реальная ссылка на обе фотографии для каждого flag.

## Этап 5. Отделить evidence от hypothesis bias

1. удалить era prior, основанный на предполагаемой истории двойников;
2. raw evidence хранить отдельно от scenario assumptions;
3. текущие posterior переименовать в support scores;
4. настоящую calibration делать только по независимому ground truth.

## Этап 6. Сделать отчёт проверяемым

Каждый тезис должен включать:

- реальные photo/pair IDs;
- даты A/B;
- bucket и pose gap;
- quality/assessability;
- metric name;
- raw A/B;
- unit;
- calibration expectation/interval;
- residual и threshold;
- альтернативные объяснения;
- уровень неопределённости;
- путь к исходному artifact.

---

# Что нужно для полноценного исправления и проверки S1

Репозиторий не содержит исходных фото и `core/3ddfa_v3`. Для следующего этапа нужны:

1. полный лог одного запуска, где видны первые ошибки;
2. точная версия/источник `core/3ddfa_v3` и список весов;
3. 5–20 main photos разных лет/ракурсов;
4. 5–20 calibration photos с известными yaw/pitch/roll;
5. соответствующие текущие output-папки для этих фото;
6. версия Python и вывод `pip freeze`/`conda env export` с рабочей машины.

До получения фото можно исправить P0-инфраструктуру, схемы, pair deduplication, config, fail-fast и тестовый каркас. Алгоритмические thresholds и научные выводы без реальных данных перенастраивать нельзя.

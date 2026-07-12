# DEEPUTIN — 50 ключевых анализов кода по методу 20/80

Дата: 2026-07-12  
Ревизия: `0181e46`  
Область: S1–S5, 3D reconstruction/canonicalization, geometry coverage, visibility, artifacts, compare, verdict, chronology, report.

## О границе «98% ошибок»

Без исходных фотографий, реальных output-папок, внешнего `core/3ddfa_v3`, весов и рабочего environment невозможно математически подтвердить покрытие 98% всех дефектов. Эти 50 проверок спроектированы по принципу 20/80: они охватывают пути, через которые проходят почти все данные и где один дефект способен испортить сотни/тысячи фото.

Проверены:

- 3D pose → canonical mesh → metrics;
- фактическое покрытие geometry catalog;
- все основные причины пустого `geometry_metrics.json`;
- условия создания/пропуска `verdict.json`;
- scalar compare и mesh compare;
- calibration, chronology, texture и report;
- статический анализ всего `project` и `core`.

## Краткий итог

Обнаружены **системные дефекты**, полностью объясняющие наблюдения пользователя:

### Почему не рассчитывается большая часть геометрии

1. Активный extractor и geometry catalog используют разные пространства имён.
2. На полной синтетической реконструкции активный extractor выдал 145 ключей, но пересечение с каталогом составило **0 из 234**.
3. Legacy-движок на 2199 метрик вообще не подключён к S1.
4. Для bucket `*_threequarter_medium` legacy-движок возвращает **0 метрик**, поскольку он знает только имя `*_threequarter_mid`.
5. При отсутствии annotation groups остаётся 19 преимущественно служебных метрик.
6. При плохой длине visibility mask весь geometry extractor падает и S1 заменяет результат на `{}`.

### Почему `verdict.json` создаётся не для всех фото

Динамическая матрица показала:

- отсутствует `info.json` → фото молча пропускается;
- отсутствует `geometry_metrics.json` → фото молча пропускается;
- пустой `{}` geometry → verdict создаётся как будто запись допустима;
- отсутствует texture → verdict создаётся;
- один повреждённый geometry/texture/identity/pair JSON → может упасть весь S4 до создания общего `verdicts.json`;
- per-photo исключения внутри verdict loop проглатываются, поэтому часть файлов создаётся, часть нет;
- записи JSON неатомарные, поэтому прерванный запуск оставляет повреждённые artifacts.

### Что с 3D-нормализацией

- Для известных функции bucket names математика `canonicalize_vertices_for_bucket()` прошла synthetic rotation test с ошибкой порядка `1e-5` model units.
- Но реальные enum/schema используют `medium`, а normalization map — `mid`.
- Поэтому medium-модели доворачиваются к yaw=0°, а не к ±45°; ошибка относительно ожидаемого положения составила около **58.8 model units** на тестовом лице.
- S3 использует другой alignment, и в нём матрица Procrustes транспонирована неверно. Чистое rigid-преобразование не совмещается обратно.
- Результат mesh/ICP практически не попадает в итоговый evidence score.

---

# 50 анализов

## Блок A. 3D-реконструкция и канонизация

### Анализ 1. Какой путь нормализации реально активен

**Код:** `project/s1_extraction/metrics/modules/geometry_extractor.py:298-331`  
**Критичность:** информационно/высокая

Активный S1 не вызывает `AlignmentEngine.align()`. Вместо этого `GeometryExtractor.extract()` напрямую вызывает `canonicalize_vertices_for_bucket()` перед вычислением метрик.

**Вывод:** сама идея «сначала довернуть, затем считать» в активном extractor присутствует. Но вокруг неё есть дефекты, перечисленные ниже. Исправление только класса `AlignmentEngine` не изменит текущий S1.

---

### Анализ 2. Математический synthetic test канонизации известных bucket names

**Код:** `modules/alignment.py:236-287`  
**Критичность:** положительная проверка

Создано невырожденное 3D-лицо, применены известные pitch/yaw/roll, translation и фактическая rotation matrix. Затем выполнено обратное вращение к target pose.

Для bucket names, которые есть в `_CANONICAL_YAW_BY_VIEW_GROUP`, максимальная ошибка составила примерно `4e-6 … 1.2e-5` model units.

**Вывод:** формула работает при условии, что:

- bucket name найден;
- `rotation_matrix` имеет именно ожидаемую 3DDFA row-vector convention;
- translation корректна;
- вход — `vertices_world`, а не camera projection.

---

### Анализ 3. Критический разрыв `medium` против `mid`

**Код:**

- `shared/schemas.py:19-20` — `left_threequarter_medium`;
- `modules/alignment.py:16-17` — `left_threequarter_mid`;
- legacy policy/config — тоже `*_mid`.

**Критичность:** CRITICAL

Когда active extractor получает `left_threequarter_medium`, lookup target yaw не находит ключ и использует default `0.0`.

Результат synthetic test:

- ожидаемый target: −45° / +45°;
- фактический target: 0°;
- ошибка положения: ~58.8 model units.

Legacy metric runner для `left_threequarter_medium` выдал **0 метрик без ошибок**; для `left_threequarter_mid` — 324.

**Исправление:** один enum vocabulary во всех schemas/configs/catalogs. Нужна migration map `mid ↔ medium` для старых artifacts.

---

### Анализ 4. Несогласованные границы девяти ракурсов

**Код:** `shared/utils.py:242-272`, `docs/JSON_FORMATS.md`, visibility thresholds.

**Критичность:** HIGH

В коде классификация использует:

- frontal `<10°`;
- light `<33°`;
- medium `<56°`;
- deep `<78°`;
- profile `≥78°`.

Документация описывает другие пороги: 15/30/45/60°. Visibility имеет отдельные переходы 45/60°.

**Следствие:** одно фото получает разные режимы в pose bucket, visibility policy и документации. Calibration bucket может не соответствовать main bucket.

---

### Анализ 5. `AlignmentEngine.align()` является заглушкой

**Код:** `modules/alignment.py:357-390`  
**Критичность:** HIGH

Класс вычисляет `target_angles`, но возвращает исходную reconstruction без изменений.

**Следствие:** любой новый код, который начнёт использовать публичный `AlignmentEngine`, ошибочно решит, что mesh нормализован. Сейчас active extractor обходит эту заглушку, но архитектурно она опасна.

---

### Анализ 6. Ошибка канонизации скрывается и используются raw vertices

**Код:** `geometry_extractor.py:322-331`  
**Критичность:** CRITICAL

Любая ошибка rotation matrix, translation, shape или bucket поглощается:

```python
except Exception:
    pass
```

После этого метрики считаются на raw pose, но artifact не содержит `canonicalization_status=failed`.

**Следствие:** часть фото выглядит обработанной, хотя метрики находятся в другом coordinate space.

---

### Анализ 7. Нельзя подтвердить convention внешнего `compute_rotation`

**Код:** `modules/reconstruction.py:241-248`, `modules/alignment.py:246-278`  
**Критичность:** HIGH

Synthetic test подтверждает внутреннюю формулу при принятой row-vector convention. Но исходник внешнего `core/3ddfa_v3` отсутствует, поэтому невозможно проверить, является ли сохранённая `rotation_matrix` именно applied matrix или требует transpose.

**Исправление:** unit test непосредственно против установленной версии 3DDFA: `base → model.transform → canonicalize → target`.

---

### Анализ 8. Translation сохраняется в «канонических» вершинах и загрязняет depth/centroid

**Код:**

- `alignment.py:280-285`;
- `geometry_extractor.py:109-120, 177-220`.

**Критичность:** CRITICAL

Canonicalization сохраняет camera/model translation. Затем extractor вычисляет:

- `centroid_x/y/z / face_scale`;
- `bone_nasion_depth = mean(z) / face_scale`;
- orbital depth.

Абсолютный `z` содержит camera distance/translation, а не только анатомическую глубину.

**Следствие:** положение/crop может интерпретироваться как изменение костей.

**Исправление:** анатомические depth metrics считать относительно стабильной face plane/origin после удаления rigid translation.

---

### Анализ 9. `vertices_canonical` в reconstruction.pkl не являются canonical

**Код:** `s1_extraction/engine.py:699-731`  
**Критичность:** CRITICAL

Сохраняется:

```python
"vertices_canonical": recon.vertices_camera
```

Это camera-space mesh в исходной позе, а не результат `canonicalize_vertices_for_bucket()`.

**Следствие:** S3 верит неверному имени и сравнивает не тот coordinate space. Geometry JSON и mesh compare используют разные версии mesh.

---

### Анализ 10. Procrustes в активном S3 использует транспонированную rotation matrix

**Код:** `s3_compare/engine.py:19-69`  
**Критичность:** CRITICAL

Для row-vector points при `H = source.T @ target` нужно согласованное `R`. Текущий код возвращает `Q.T` вместо известного `Q`.

Synthetic rigid test:

- target был создан только rotation+translation;
- после alignment mean residual остался `0.852` model units;
- returned `R` совпал с `Q.T`, а не `Q`.

**Следствие:** даже идентичные meshes после известного поворота выглядят различными.

---

### Анализ 11. Рассчитанный shared-visible alignment выбрасывается

**Код:** `s3_compare/engine.py:413-425`  
**Критичность:** CRITICAL

Сначала Procrustes считается на `shared_vis`, затем его `R/t` не применяются к full mesh. Вместо этого вызывается новый Procrustes на всех вершинах:

```python
full_aligned = procrustes_align(verts_a, verts_b)[0]
```

**Следствие:** hidden/expression vertices снова участвуют в alignment; visibility filtering не выполняет заявленную функцию.

---

### Анализ 12. Zone fallback содержит гарантированный `NameError`

**Код:** `s3_compare/zone_mapper.py:155-190`  
**Критичность:** CRITICAL

Функция получает `face_center`, но использует неопределённую переменную `skin_center`.

Ruff подтвердил `F821 Undefined name skin_center`.

Если хотя бы одна forensic zone пуста, ICP path падает; активный compare ловит ошибку и тихо возвращает `None`.

---

### Анализ 13. ICP/heatmap почти не влияет на итоговый verdict

**Код:** `s3_compare/engine.py:299-348, 363-386`  
**Критичность:** CRITICAL

`icp_distance` и `heatmap` вычисляются, но:

- heatmap не записывается в `PairEvidence`;
- ICP distance не добавляется к geometry distance;
- сохраняется только строковый flag `icp_dist=...`;
- report не может восстановить настоящий per-zone mesh evidence.

Ruff также показывает, что local `heatmap` назначен, но не используется.

---

### Анализ 14. Expression не нейтрализуется по умолчанию

**Код:**

- `ExtractionEngine`: `neutral_expression=False`, `identity_only=False`;
- `reconstruction.py:234-241`;
- expression exclusion S3 не получает ожидаемую структуру.

**Критичность:** HIGH

По умолчанию base shape содержит полный expression vector. Smile/jaw flags не применяются к geometry extractor до вычисления метрик.

**Следствие:** губы, подбородок, jaw и часть щёк меняются из-за мимики и идут в identity distance.

---

### Анализ 15. `id_params` и `exp_params` сохраняются не там, где ищет active extractor

**Код:**

- `_reconstruction_to_dict()` кладёт параметры top-level;
- `GeometryExtractor` ищет `reconstruction["payload"]`.

**Критичность:** HIGH

В реальном artifact `id_norm/id_mean/id_std` и expression metrics не рассчитываются, хотя на in-memory `ReconstructionResult.payload` данные существовали.

---

## Блок B. Geometry coverage, zones и visibility

### Анализ 16. Active geometry output не пересекается с geometry catalog

**Код:** `geometry_extractor.py` против `geometry/catalog.py`  
**Критичность:** CRITICAL

Dynamic result на полной reconstruction:

- catalog list: 234 entries, 233 unique;
- active output: 145 metrics;
- catalog overlap: **0**;
- catalog missing: фактически весь каталог.

Active names: `zone_left_eye_span_x`, `zone_skin_centroid_z`, ...  
Catalog names: `zone_orbit_L_span_lateral_ratio`, `zone_cheekbone_L_*`, ...

**Это основной ответ на жалобу о нерассчитанной геометрии.**

---

### Анализ 17. Geometry resolver не может классифицировать active output

**Код:** `geometry/resolver.py:31-74, 154-163`  
**Критичность:** CRITICAL

Resolver rules ожидают catalog/legacy keys. На полном наборе active keys:

- hits: 0;
- selected metric keys: 0;
- scores PUT/UDMURT/VAS: 0;
- result: `UNCERTAIN`, confidence 0.

Позднее S2 подменяет это distance-to-calibration heuristic, поэтому проблема маскируется.

---

### Анализ 18. Legacy catalog на 2199 метрик — мёртвая ветка

**Код:** `geometry/legacy_metrics/*`  
**Критичность:** CRITICAL

В репозитории есть registry с 2199 specs и runner, но `ExtractionEngine` их не вызывает. Поиск показал, что `compute_single_photo_metrics`, `build_metric_context` и `project_geometry_aliases` не используются production pipeline.

**Следствие:** наличие файлов и coverage report не означает, что метрики реально извлекаются.

---

### Анализ 19. Legacy runner возвращает 0 для реального medium enum

**Критичность:** CRITICAL

Dynamic result:

- frontal: 513 values;
- left light: 350;
- left `mid`: 324;
- left `medium`: **0**;
- left profile: 325.

Ошибок при `medium` нет — система молча возвращает пустой список.

---

### Анализ 20. Legacy `vertices_canon` равны raw vertices

**Код:** `legacy_metrics/context.py:151-158`  
**Критичность:** CRITICAL

Context делает:

```python
vertices_canon = vertices_raw.copy()
```

Никакой canonical rotation не применяется, хотя source space у сотен результатов записывается как `canon_bucket`.

Dynamic test подтвердил `vertices_raw == vertices_canon` для всех bucket.

---

### Анализ 21. Legacy visibility получает segmentation image вместо per-vertex mask

**Код:** `legacy_metrics/context.py:10-16, 159, 207-208`  
**Критичность:** CRITICAL

`visibility` map направлен на `seg_visible`, который имеет форму примерно `224×224×8`. Ожидается mask длины N vertices.

Dynamic context получил visibility shape `[224, 224, 8]`.

**Следствие:** если legacy modules начнут реально использовать visibility, индексация/coverage будет неверной.

---

### Анализ 22. Alias generator не используется и генерирует псевдометрики

**Код:** `geometry/aliases.py`  
**Критичность:** HIGH

`project_geometry_aliases()` нигде не вызывается. При этом он создаёт сотни анатомически названных полей из небольшого числа общих proxies, например side bias ±0.03 и общих face width/height.

Ruff обнаружил восемь повторяющихся dict keys; более поздние значения незаметно перезаписывают ранние.

**Вывод:** его нельзя просто подключить как «исправление coverage» — это создаст заполненные, но не измеренные метрики.

---

### Анализ 23. Полное отсутствие annotation groups оставляет только 19 метрик

**Критичность:** HIGH

Dynamic active extractor:

- complete groups: 145;
- no groups: 19.

Остаются mesh bbox, pose, scale, symmetry и часть служебных полей. Большинство зональных/костных значений исчезает.

---

### Анализ 24. Частичные annotation groups дают частичный JSON без coverage status

**Критичность:** HIGH

При 3 группах из 8 active extractor вернул 66 metrics. Artifact не говорит, что отсутствуют nose/lips/skin и связанные зоны.

**Следствие:** два JSON разной полноты сравниваются по случайному intersection без minimum required coverage.

---

### Анализ 25. Полностью невидимый mesh всё равно считается частично пригодным

**Критичность:** HIGH

С `visible_idx_renderer = all False` extractor вернул 23 metrics. `readiness.geometry` всё равно записывается как `available`.

**Следствие:** «ничего не видно» не отделяется от корректного нулевого измерения.

---

### Анализ 26. Неверная длина visibility mask обнуляет всю геометрию

**Код:** `compute_asymmetry()` и zone filtering  
**Критичность:** CRITICAL

При 800 vertices и visibility длины 10 получен:

```text
IndexError: boolean index did not match indexed array
```

Inline extractor ловит эту ошибку снаружи и устанавливает `geometry = {}`.

**Это прямой механизм появления пустых geometry files.**

---

### Анализ 27. Renderer mask может быть неверно истолкован как список индексов

**Код:** `reconstruction.py:358-373`  
**Критичность:** HIGH

Любой 1D integer array трактуется как vertex indices. Если renderer возвращает integer 0/1 mask длины N, код пометит видимыми преимущественно только vertices 0 и 1.

Нужна явная проверка:

- dtype bool или unique `{0,1}` + length N → mask;
- иначе → index list.

---

### Анализ 28. Порядок восьми annotation groups не валидируется

**Код:** `geometry_extractor.py:11-16`, `zone_mapper.py:1-7`  
**Критичность:** HIGH

Весь анализ предполагает фиксированный порядок `[right_eye, left_eye, ... skin]`, но asset metadata/version не проверяются.

Если другая версия face model использует другой order, все анатомические названия становятся неверными без исключения.

---

### Анализ 29. Несколько разных анатомических зон используют одну и ту же полную skin group

**Код:** `geometry_extractor.py:18-35, 188-220`  
**Критичность:** HIGH

В mapping cheekbone L/R, chin, jaw L/R, forehead и nose wings указывают на `skin`. В active macro metrics обе cheek переменные получают одну полную skin zone.

**Следствие:** поля с разными анатомическими названиями не являются независимыми измерениями соответствующих зон.

---

### Анализ 30. PCA normal имеет произвольный знак

**Код:** `geometry_extractor.py:92-119`  
**Критичность:** HIGH

SVD/PCA normal `vh[-1]` может быть `n` или `-n` между запусками/фото. Код не ориентирует normal относительно camera/outward reference.

**Следствие:** `normal_mean_x/y/z` способен инвертировать знак без изменения лица.

---

### Анализ 31. Asymmetry algorithm не сопоставляет левые и правые анатомические vertices

**Код:** `geometry_extractor.py:225-270`  
**Критичность:** HIGH

Алгоритм отражает каждый vertex по x, затем сравнивает с vertex того же индекса после global rotation. Это не является correspondence «левая точка ↔ правая парная точка». Mid-sagittal plane явно не оценивается.

**Следствие:** `bone_asymmetry_x` нельзя интерпретировать как физическую костную асимметрию.

---

### Анализ 32. Geometry validation не проверяет catalog coverage

**Код:** `shared/validation.py:296-350`  
**Критичность:** CRITICAL

Validator проверяет только:

- пуст ли dict;
- NaN/None;
- numeric type.

Он не проверяет:

- required metrics для bucket;
- overlap с catalog;
- min zone coverage;
- canonicalization status;
- visibility ratio;
- units/source space.

Поэтому 19 служебных metrics могут пройти как успешная geometry extraction.

---

### Анализ 33. Geometry exception превращается в `{}` без структурированной причины

**Код:** `s1_extraction/engine.py:138-143`  
**Критичность:** HIGH

Любая ошибка geometry даёт `{}`. Нет полей:

- error type;
- failed module;
- input shapes;
- canonicalization status;
- retryability.

Later stages не отличают failed extraction от «у лица реально нет метрик».

---

### Анализ 34. Geometry JSON сохраняется только после успешной texture/classifier цепочки

**Код:** `engine.py:138-272`  
**Критичность:** CRITICAL

Geometry вычисляется в начале, но `geometry_metrics.json` записывается только в самом конце общего метода. Между ними выполняются texture extraction, physical features, classifier, cohort logic и Stage2 validation.

Если texture/classifier падает, уже рассчитанная geometry **не сохраняется вообще**.

Это объясняет фото, где реконструкция есть, но metrics/verdict отсутствуют.

---

### Анализ 35. Face assets и coordinate spaces могут обнулить texture/physical extraction

**Код:** `engine.py:496-505, 510-600`; physical extraction: `165-188`  
**Критичность:** HIGH

Проблемы:

- return value `cv2.imwrite()` не проверяется;
- normal file читается OpenCV как BGRA, но часть кода называет первые каналы RGB;
- landmarks находятся в original/model coordinates;
- physical extractor получает resized crop 424×500 без transform landmarks в crop space;
- если mask не читается, создаётся нулевой RGBA и processing продолжается.

**Следствие:** некоторые фото дают пустые/ложные texture и physical metrics без fatal status.

---

## Блок C. Artifacts и нестабильное создание verdict-файлов

### Анализ 36. Отсутствующий `info.json` молча исключает фото из S4

**Dynamic result:** returned verdicts 0, per-photo verdict отсутствует, общий `verdicts.json` создаётся пустым.

**Критичность:** HIGH

Нет quarantine/error manifest с причиной пропуска.

---

### Анализ 37. Отсутствующий `geometry_metrics.json` молча исключает фото

**Код:** `s4_verdict/engine.py:175-193`  
**Dynamic result:** verdict не создаётся.

**Критичность:** CRITICAL

Это главный непосредственный механизм «для некоторых фото verdict-файла нет».

---

### Анализ 38. Пустой geometry `{}` считается допустимым и получает verdict

**Dynamic result:** verdict создан.

**Критичность:** CRITICAL

Таким образом:

- missing file → фото исчезает;
- empty dict → фото получает Bayesian/heuristic verdict.

Это непоследовательная семантика. При отсутствии geometry должен быть `UNCERTAIN / insufficient evidence`, а не обычный verdict.

---

### Анализ 39. Missing texture допускается, missing geometry — нет

**Dynamic result:** без `texture_metrics.json` verdict создан с `{}` texture.

**Критичность:** HIGH

Такая асимметрия не описана schema/readiness policy и ведёт к разным судьбам частичных artifacts.

---

### Анализ 40. Один повреждённый metrics JSON останавливает весь S4

**Dynamic result:** malformed geometry или texture вызвал `JSONDecodeError`; не созданы ни per-photo verdict, ни общий `verdicts.json`.

**Критичность:** CRITICAL

`load_json()` не имеет safe/quarantine mode, а loaders S4 не изолируют ошибки по фото.

---

### Анализ 41. Повреждённый identity или pair index также останавливает весь S4

**Dynamic result:**

- malformed `identity.json` → весь stage падает;
- schema-incomplete identity → Pydantic ValidationError;
- malformed `pair_index.json` → весь stage падает;
- одна incomplete PairEvidence → весь stage падает.

**Критичность:** CRITICAL

Ошибки должны быть локализованы по artifact/photo/pair, а не останавливать весь архив.

---

### Анализ 42. JSON writes неатомарные, resume/cleanup отсутствуют

**Код:** `shared/utils.py:40-45`  
**Критичность:** HIGH

Файл пишется напрямую в final path. При остановке процесса остаётся truncated JSON. На следующем запуске он вызывает описанный stage-wide crash.

Также photo output directory создаётся до завершения reconstruction; stale artifacts предыдущего запуска не очищаются и не проверяются по version/hash.

**Исправление:** write temp → fsync → atomic rename; artifact manifest с status/hash/version.

---

### Анализ 43. `photo_id = filename stem` создаёт collisions

**Код:** `shared/utils.py:109-110`, recursive `list_images()`  
**Критичность:** HIGH

Два файла:

```text
folderA/2005_01_01.jpg
folderB/2005_01_01.png
```

получат один output directory. Второй перезапишет artifacts первого; manifest может содержать дубли ID; verdict останется один.

**Исправление:** stable ID должен включать relative path и content hash.

---

### Анализ 44. Ошибки отдельных фото и S1 могут завершиться «успешно»

**Код:** `ExtractionEngine.run():387-411`, `PipelineRunner`  
**Критичность:** CRITICAL

Per-photo exceptions только логируются. Runner не получает failed count и пишет `S1 complete`. Missing calibration reference тоже warning.

В S4 per-photo exception приводит к `continue`, поэтому часть `verdict.json` создаётся, часть нет, но stage может дойти до конца.

**Исправление:** обязательный StageResult + `failed_photo_manifest.json` + non-zero exit по формальным критериям.

---

## Блок D. Остальные 20% кода, дающие 80% искажений

### Анализ 45. Pair comparison создаёт дубли и смещает verdict

**Код:** `s3_compare/engine.py:148-182`  
**Критичность:** HIGH

Одна пара может быть adjacent и anchor с одинаковым pair ID. На smoke test:

- 64 rows;
- 58 unique IDs;
- 6 duplicated pairs.

Они повторно входят в средние S4.

---

### Анализ 46. Pose-aware calibration фактически отключена

**Код:** `s2_identity/engine.py:37-47`; CompareEngine  
**Критичность:** CRITICAL

S2 читает `record.pose`, которого нет в Stage2Record, и записывает yaw/pitch/roll=0. Созданный `pose_calibration_models.json` CompareEngine не загружает.

То есть заявленное вычитание шума конкретной разницы ракурсов не работает.

---

### Анализ 47. Chronology смешивает разные buckets и некорректные размерности

**Код:** `s4_verdict/modules/chronology.py`  
**Критичность:** CRITICAL

Все ракурсы сортируются в один timeline. Metric series не строятся отдельно по bucket. `rate = delta / days` делится на baseline, рассчитанный из raw delta, то есть сравниваются разные размерности.

**Следствие:** смена frontal→profile может стать «биологическим скачком», а настоящий скачок может исчезнуть.

---

### Анализ 48. Texture classifier подключён с неправильным contract и hypothesis bias

**Код:**

- `engine.py:123`;
- `classifier_v5.py:86, 91-97, 58-64`.

**Критичность:** CRITICAL

Путь leaderboard передаётся в boolean `quality_compensation`; модель не загружается. При classify не передаются pose/year. В classifier встроен prior, повышающий silicone probability для поздних лет исходя из расследуемой гипотезы.

Дополнительно downstream теряет `texture_assessability`, `quality_summary` и physical `metric_notes`, потому что не читает Stage2 manifest.

---

### Анализ 49. Report и «калиброванные вероятности» не соответствуют evidence

**Код:** `s5_report/engine.py`, `s4_verdict/calibration_analysis.py`  
**Критичность:** CRITICAL

- normalized geometry score умножается на 100 и называется mm;
- dates заменяются `N/A`;
- создаются несуществующие `photo__anchor` pair IDs;
- chronology получает одинаковые date A/B и gap=0;
- Platt calibrator не fit;
- calibration report использует собственные predictions как `y_true`.

Итоговые posterior/report нельзя трактовать как вероятности или физические измерения.

---

### Анализ 50. Оркестрация, environment и тесты не защищают ни один из этих контрактов

**Критичность:** CRITICAL

Одновременно:

- `run_pipeline.py` передаёт `main_output` как calibration input;
- абсолютные пути привязаны к одному Mac;
- отсутствуют requirements/pyproject;
- `core/3ddfa_v3` отсутствует;
- собственный YAML отвергается собственным validator из-за `s2`;
- `python -m project.run --help` требует torch;
- pytest collection падает;
- production contract tests для S1–S5 отсутствуют;
- Ruff нашёл 311 diagnostics, включая 3 undefined names и 11 duplicate dict keys.

**Следствие:** дефекты не могли быть автоматически обнаружены до полного запуска архива.

---

# Дополнительные динамические результаты

## Geometry degradation matrix

| Case | Active metrics | Catalog overlap | Результат |
|---|---:|---:|---|
| 8 annotation groups, visibility valid | 145 | 0 | Формально заполнено, catalog не покрыт |
| annotation groups отсутствуют | 19 | 0 | Почти только служебные metrics |
| только 3 groups | 66 | 0 | Partial без coverage status |
| все vertices invisible | 23 | 0 | Readiness всё равно available |
| visibility length mismatch | — | — | `IndexError`, geometry становится `{}` |

## Legacy runner matrix

| Bucket | Values |
|---|---:|
| frontal | 513 |
| left_threequarter_light | 350 |
| left_threequarter_mid | 324 |
| left_threequarter_medium | **0** |
| left_profile | 325 |

## Verdict artifact matrix

| Artifact condition | Per-photo verdict | Общий verdicts.json |
|---|---:|---:|
| valid | да | да |
| missing info | нет | пустой |
| missing geometry | нет | пустой |
| empty geometry `{}` | да | да |
| missing texture | да | да |
| malformed geometry/texture | нет | нет, stage exception |
| malformed identity | нет | нет, stage exception |
| invalid pair schema | нет | нет, stage exception |

---

# Первопричины по симптомам пользователя

## Симптом: «геометрические метрики наполовину не рассчитались»

Наиболее вероятная цепочка:

1. Geometry catalog и active extractor несовместимы по названиям.
2. Legacy 2199-metric engine не вызывается.
3. `medium` buckets несовместимы с `mid`.
4. Annotation/visibility частично отсутствуют или имеют неверную форму.
5. Validator не сообщает coverage относительно required bucket set.
6. Ошибка texture после geometry мешает сохранить уже рассчитанный geometry JSON.

## Симптом: «verdict-файлы то есть, то нет»

Наиболее вероятная цепочка:

1. S1 сохранил `info.json`, но inline metrics упал до записи geometry.
2. S4 молча пропустил photo directory без geometry file.
3. На другом фото geometry был `{}`, поэтому verdict неожиданно создался.
4. Повреждённый JSON от прерванного запуска остановил весь S4.
5. Per-photo verdict loop продолжил работу после локальной ошибки, оставив дырки.
6. Повторный запуск смешал новые и stale artifacts.

## Симптом: «для некоторых фото вообще ничего не извлекает»

Вероятные точки:

- face detector не нашёл лицо;
- model/assets unavailable;
- bad visibility shape;
- invalid annotation indices/order;
- face mask не записалась/не прочиталась;
- texture extractor/classifier упал до сохранения geometry;
- output ID collision перезаписал другое фото;
- exception был подавлен и runner завершился с ложным success.

---

# Рекомендуемый порядок исправлений

## P0 — сначала, до повторной обработки архива

1. Унифицировать `medium`/`mid` во всех enums, configs и artifacts.
2. Ввести `artifact_version` и migration старых bucket names.
3. Выбрать один geometry engine: active или legacy; удалить иллюзию параллельной production-готовности.
4. Создать настоящий catalog contract: expected/required/optional metrics по bucket.
5. Сохранять geometry сразу после geometry extraction, независимо от texture.
6. Ввести `geometry_status.json` с canonicalization, visibility, groups, coverage и errors.
7. Исправить `vertices_canonical` и сохранять реально canonical mesh.
8. Удалить translation из anatomy-relative depth metrics.
9. Исправить S3 Procrustes и применять shared-visible R/t ко всему mesh.
10. Исправить `skin_center` и сохранять настоящий per-zone ICP evidence.
11. Missing/empty geometry → `INSUFFICIENT_EVIDENCE`, а не skip/обычный verdict.
12. Изолировать malformed artifacts по фото; не падать всем stage.
13. Atomic JSON writes + temp files + manifest completion marker.
14. Unique photo ID через relative path + content hash.
15. Возвращать non-zero exit при failed photos/calibration.

## P1 — затем

16. Expression-neutral identity mesh или zone exclusion до метрик.
17. Per-vertex visibility contract и shape validation.
18. Correct annotation metadata/version validation.
19. Реальная pose-aware calibration, используемая CompareEngine.
20. Bucket-separated chronology.
21. Удалить era hypothesis prior из texture classifier.
22. Перестать называть heuristic scores вероятностями/mm.

---

# Минимальные acceptance tests перед новым полным запуском

1. Один synthetic mesh при 20 разных yaw/pitch/roll после canonicalization даёт одинаковые anatomy metrics в пределах tolerance.
2. Для каждого из 9 bucket names target yaw совпадает с policy.
3. Medium bucket извлекает ненулевой required metric set.
4. Geometry output имеет ≥ заданного overlap с catalog.
5. Visibility mask неправильной длины даёт structured failure, не `{}`.
6. Missing annotation groups даёт `INSUFFICIENT_GEOMETRY`.
7. Texture failure не удаляет успешно рассчитанную geometry.
8. Missing geometry создаёт explicit uncertain verdict с причиной.
9. Один malformed JSON не останавливает другие фото.
10. Rigid-transform mesh pair после S3 alignment имеет residual около нуля.
11. Pair IDs уникальны.
12. Ни один normalized score не получает unit `mm` без scale calibration.
13. Каждый journalist thesis разрешается в реальные photo/pair artifacts.
14. Pipeline exit code != 0 при необработанных фото.
15. Повторный запуск на тех же inputs детерминирован и не читает stale incomplete artifacts.

---

# Что нужно от реального запуска для следующего этапа

Чтобы перейти от code audit к точечному исправлению и проверить реальные 3DDFA conventions, нужны:

- один полный `deeputin_*.log`;
- 10–20 проблемных photo output directories целиком;
- соответствующие исходные фото;
- рабочая папка/версия `core/3ddfa_v3`;
- `conda env export` или `pip freeze`;
- перечень ожидаемых geometry metrics по каждому из девяти ракурсов.

Без этого можно исправить архитектурные P0-дефекты, но нельзя честно валидировать физические thresholds и внешнюю rotation convention модели.

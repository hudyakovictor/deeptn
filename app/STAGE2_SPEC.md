# ТЗ этапа 2 — калиброванный хронологический анализ

## 1. Цель

Этап 2 анализирует уже извлечённые данные без повторного запуска 3DDFA_V3. Он сравнивает фотографии внутри однородных ракурсов и отделяет наблюдаемые изменения лица от шума реконструкции, используя семь калибровочных наборов, в каждом из которых гарантированно изображён один человек.

Этап 2 не подтверждает заранее выбранную гипотезу и не выдаёт медицинских, биологических или идентификационных утверждений. Его результат — измерения, неопределённость, аномалии временного ряда и пригодность доказательств.

## 2. Входы

### 2.1 Основной набор

- `stage1_manifest.json` со статусом complete;
- `main_index.csv` и `main_timeline.csv`;
- каталоги фотографий schema `deeputin-photo-v2.0`;
- только записи с `validation.status=complete`;
- отдельные статусы пригодности geometry, segmentation и UV.

### 2.2 Калибровка

Семь наборов `person_01`…`person_07`, 943 валидированные записи schema v7. Внутри каждого набора все кадры принадлежат одному человеку. Калибровка используется для оценки repeatability/noise 3DDFA, но не как эталон внешности основного человека.

### 2.3 Запреты

- повторный inference 3DDFA;
- сравнение разных pose bins как эквивалентных наблюдений;
- использование невидимых landmarks/вершин;
- применение одного глобального порога ко всем ракурсам;
- выводы по одной паре фотографий;
- использование beauty UV для анализа;
- Bayesian verdict, synthetic probability, identity persona и заранее заданные «биологические пределы».

## 3. Единица анализа

Основная временная серия строится отдельно для каждого `pose_bin`:

- left_profile;
- left_deep;
- left_mid;
- left_light;
- frontal;
- right_light;
- right_mid;
- right_deep;
- right_profile.

Внутри bin записи сортируются по `(date, same_date_sequence, photo_id)`. Сравнения выполняются:

1. соседняя пара по хронологии;
2. устойчивый baseline — медиоид первых качественных наблюдений bin;
3. rolling reference — медиоид последних K пригодных наблюдений;
4. long-range anchors для проверки устойчивости обнаруженного сдвига.

Никакая пара из разных bins не участвует в основном количественном выводе. Межракурсные связи допускаются только как независимое подтверждение направления изменения.

## 4. Подготовка данных

### 4.1 Geometry representation

Главное представление — `object_normalized`. `bin_canonical` используется для визуализации и предварительной инициализации, но не заменяет pair alignment.

Анализировать отдельно:

- identity+expression mesh/landmarks;
- identity-only mesh/landmarks;
- alpha_id;
- alpha_exp.

Это позволяет различать изменение реконструированной identity-формы, expression и общий шум модели.

### 4.2 Visibility

Для пары A/B валидны только точки:

`visible_A AND visible_B`.

При наличии отдельных масок дополнительно сохраняются доли потерь из-за front-facing и renderer occlusion. Минимальное покрытие определяется калибровкой отдельно для каждого bin и зоны.

### 4.3 Pair alignment

- начальная система: normalized object space;
- similarity/rigid alignment по стабильным inner-face anchors;
- robust ICP только по общей видимой области;
- trimming верхнего хвоста residuals;
- запрет scale drift сверх калибровочного диапазона;
- обязательное сохранение residual before/after, scale, rotation, translation, overlap и convergence status.

Список стабильных anchors выбирается по repeatability семи calibration datasets, а не вручную.

## 5. Модель шума по калибровке

### 5.1 Подбор калибровочных кадров

Для каждой основной записи из каждого из семи наборов выбрать top-K кандидатов по расстоянию:

- yaw, pitch, roll;
- combined visibility pattern/coverage;
- face scale или projected landmark scale;
- geometry QA;
- при texture-анализе — UV observed coverage.

Выбор не использует геометрическое сходство лица, иначе calibration noise будет смещён в сторону основного объекта.

### 5.2 Calibration pairs

Внутри каждого человека строятся пары кадров с близкими условиями. Так как identity фиксирована, остаточные различия оценивают:

- reconstruction noise;
- pose leakage;
- expression leakage;
- visibility/occlusion instability;
- sensitivity к качеству изображения.

### 5.3 Нормирование

Для каждой метрики, pose bin, зоны и quality stratum оценить:

- median noise;
- MAD/IQR;
- p90/p95/p99;
- bootstrap confidence interval;
- долю нестабильных точек;
- зависимость от pose distance и visibility overlap.

Основной эффект выражать как robust z-score или percentile относительно calibration distribution. Сохранять и raw effect, и normalized effect.

### 5.4 Семь независимых оценок

Не объединять все calibration datasets сразу. Сначала получить семь независимых оценок, затем агрегировать медианой. Сохранять dispersion между людьми и leave-one-calibration-set-out sensitivity.

## 6. Геометрический анализ

### 6.1 Соответствие целям aboutplatform

Старое ТЗ используется как перечень исследовательских направлений, а не как источник заранее истинных выводов:

- bilateral asymmetry и взаимное положение орбит/челюсти оцениваются только во frontal bin;
- профильные depth/contour markers оцениваются отдельно для left/right profile и только по видимой стороне;
- zygomatic/orbital и другие zone markers требуют достаточного renderer-visible overlap;
- volume/width proxies допускаются только после calibration repeatability и coverage gate;
- хронологические инверсии и скачки считаются кандидатами аномалии, но не доказательством маски или подмены;
- любой кандидат обязан пройти calibration noise, expression, quality, persistence и cross-bin checks.

### 6.2 Уровни

1. alpha level: расстояние и направление изменения alpha_id/alpha_exp;
2. landmark level: LDM106 и LDM134 residual vectors;
3. dense mesh level: robust point-to-point/point-to-plane residuals;
4. zone level: агрегаты анатомических зон;
5. temporal level: устойчивость изменения по следующим датам.

### 6.3 Метрики

Минимальный набор:

- median and p95 Euclidean residual;
- signed depth/lateral/vertical displacement;
- trimmed RMSE;
- point-to-plane residual;
- local surface normal change;
- zone coverage;
- alpha_id cosine/L2 distance;
- alpha_exp distance;
- identity-only versus identity+expression delta.

Не рассчитывать десятки коррелирующих индексов до проверки repeatability.

### 6.4 Зоны

Versioned zone map строится по vertex indices. Для каждой зоны сохраняются coverage, raw effect, calibration percentile, CI и status. Зона без достаточной общей видимости получает `insufficient_visibility`, а не нулевое изменение.

## 7. Хронологический анализ

Для каждой серии внутри pose bin:

- robust baseline;
- adjacent deltas;
- cumulative displacement from baseline;
- change-point candidates;
- persistence: сохраняется ли эффект в последующих кадрах;
- reversibility: возвращается ли ряд в calibration noise band;
- cross-bin corroboration: виден ли согласованный эффект в другом ракурсе.

Сильной аномалией считается не большой residual одной пары, а изменение, которое:

1. выше calibration p95/p99;
2. локализовано в зоне с достаточной видимостью;
3. устойчиво к top-K calibration matches;
4. повторяется в последующих датах;
5. желательно подтверждается независимым pose bin;
6. не объясняется alpha_exp или низким geometry QA.

## 8. Texture/UV-анализ

Источник — только `uv_texture_analysis` в пересечении:

`observed AND is_original AND confidence >= threshold`.

Beauty texture запрещена для измерений. Анализ выполняется patch-based, с erosion и без пар через границу ROI. Масштаб patch/distance нормируется по projected face scale. Минимальные семейства: GLCM, LBP histogram, gradients, frequency/residual and color-neutral local contrast. Все признаки проходят calibration repeatability; нестабильные удаляются.

Texture и geometry имеют независимые статусы. Texture никогда не изменяет geometry measurement, а служит отдельным каналом подтверждения/опровержения.

## 9. Negative controls

Обязательны:

- calibration same-person pairs как null distribution;
- пары основной фотографии с самой собой после сериализации — ожидаемый zero residual;
- перестановка хронологии — проверка ложных change points;
- deliberately mismatched pose pairs — должны быть отклонены gate;
- identity+expression против identity-only — контроль expression leakage.

## 10. Выходы

- `analysis_manifest.json` с hash входного stage-1 manifest, config/code hash;
- `calibration_noise_model.npz/json`;
- `pair_metrics.parquet` или CSV;
- `zone_metrics.parquet` или CSV;
- `timeline_metrics.parquet` или CSV;
- `change_points.json`;
- `photo_analysis/{photo_id}.json`;
- данные heatmap без финального рендера;
- `analysis_validation.json`.

Каждое поле имеет units, representation, metric version, calibration stratum и uncertainty.

## 11. Статусы вместо verdict

Допустимые статусы:

- within_calibration_noise;
- elevated_but_uncertain;
- persistent_geometric_change;
- expression_dominated;
- insufficient_visibility;
- insufficient_quality;
- insufficient_temporal_support.

Запрещено автоматически объявлять подмену личности, маску, операцию или «биологически невозможное» изменение. Такие интерпретации не следуют непосредственно из 3DDFA residuals.

## 12. Критерии приёмки этапа 2

- все результаты воспроизводимы без 3DDFA inference;
- ни одна основная метрика не использует пары разных pose bins;
- каждая метрика имеет calibration distribution;
- все зоны используют intersection visibility;
- zero-point tests близки к машинному нулю;
- calibration false-positive rate измерен и записан;
- изменение одной threshold/config создаёт новый config hash;
- interrupted run безопасно продолжается;
- не менее 95% валидных stage-1 записей получают корректный анализ либо объяснимый статус недостаточности;
- никакой verdict не строится без temporal persistence и calibration uncertainty.

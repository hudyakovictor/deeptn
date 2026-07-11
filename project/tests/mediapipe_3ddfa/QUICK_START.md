# Quick Start — Eye Test Mask Detection

## Быстрый старт

### 1. Подготовка (один раз)

```bash
# Клонировать 3DDFA-V3
git clone https://github.com/wang-zidu/3DDFA-V3
cd 3DDFA-V3

# Установить зависимости (см. README 3DDFA-V3)
conda create -n TDDFAV3 python=3.8
conda activate TDDFAV3
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip install -r requirements.txt
pip install mediapipe scipy scikit-image

# Скопировать скрипты
cp /path/to/eyetest.py .
cp /path/to/compare_eyetest.py .
```

### 2. Тестирование

```bash
# Прогнать на реальных фото
python eyetest.py -i /path/to/real_photos -o eyetest_real.json

# Прогнать на фото в маске
python eyetest.py -i /path/to/mask_photos -o eyetest_mask.json

# Сравнить
python compare_eyetest.py eyetest_real.json eyetest_mask.json
```

### 3. Результат

Скрипт покажет таблицу:
```
Rank Method                          AUC Cohen_d  Real_μ  Mask_μ
----------------------------------------------------------------
1    iris_aperture_ratio            0.92    2.10   0.150   0.720
2    aperture_shrinkage             0.89    1.85   0.120   0.650
3    eyelid_sss                     0.87    1.72   0.180   0.580
...
```

**AUC > 0.9** = метод отлично разделяет реальные лица и маски.

---

## Структура вывода

Каждое фото получает **25 оценок** (0=реальное лицо, 1=маска):

### Eye-focused (15 методов)
- `aperture_shrinkage` — сужение глазной щели
- `iris_aperture_ratio` — зрачок кажется больше
- `eyelid_thickness` — утолщение века
- `lid_smoothness` — гладкость века
- `lash_invisibility` — отсутствие ресниц
- `periocular_discontinuity` — скачок текстуры
- `eyelid_edge_sharpness` — резкость края века
- `eye_depth_anomaly` — глубина глаз (Z из 3DDFA)
- `eye_symmetry_anomaly` — асимметрия L/R
- `brow_lid_gap` — зазор бровь-веко
- `sclera_iris_boundary` — граница склера-радужка
- `pupil_apparent_size` — видимый размер зрачка
- `orbit_area_ratio` — площадь орбит (segmentation)
- `periocular_lbp_entropy` — энтропия LBP
- `eyelid_sss` — R-B diff (subsurface scattering)

### Cross-system (10 методов)
- `landmark_discrepancy` — расхождение landmarks 3DDFA vs MP
- `specular_brdf` — характер бликов
- `eye_contour_divergence` — расхождение контура глаз
- `bbox_expansion` — расширение bbox
- `subsurface_violation` — нарушение SSS
- `nasolabial_suppression` — подавление складок
- `ear_texture_cliff` — скачок у ушей
- `gaze_inconsistency` — несоответствие взгляда
- `mesh_curvature_anomaly` — гладкость mesh
- `skin_tone_mismatch` — расхождение цвета

---

## Пример вывода для одного фото

```json
{
  "filename": "photo_001.jpg",
  "eye_scores": {
    "aperture_shrinkage": 0.65,
    "iris_aperture_ratio": 0.72,
    "eyelid_thickness": 0.40,
    "lid_smoothness": 0.55,
    "lash_invisibility": 0.30,
    "periocular_discontinuity": 0.45,
    "eyelid_edge_sharpness": 0.50,
    "eye_depth_anomaly": 0.60,
    "eye_symmetry_anomaly": 0.25,
    "brow_lid_gap": 0.70,
    "sclera_iris_boundary": 0.35,
    "pupil_apparent_size": 0.68,
    "orbit_area_ratio": 0.55,
    "periocular_lbp_entropy": 0.40,
    "eyelid_sss": 0.75
  },
  "cross_scores": {
    "landmark_discrepancy": 0.55,
    "specular_brdf": 0.30,
    ...
  },
  "combined_mean": 0.52,
  "combined_max": 0.75,
  "top5_methods": [
    ["eyelid_sss", 0.75],
    ["iris_aperture_ratio", 0.72],
    ["brow_lid_gap", 0.70],
    ["pupil_apparent_size", 0.68],
    ["aperture_shrinkage", 0.65]
  ]
}
```

---

## Что делать если...

### MediaPipe не находит лицо
- Проверьте что фото достаточно качественное
- Увеличьте `min_detection_confidence` в MediaPipeAnalyzer (сейчас 0.3)
- Для полупрофиля MediaPipe работает хуже — используйте больше eye методов

### 3DDFA-V3 падает на некоторых фото
- Проверьте что retinaface установлен
- Попробуйте `--detector mtcnn`
- Для CPU: `--device cpu` (медленно, но работает)

### Хочу добавить свой метод
- Добавьте метод в класс `EyeMaskDetector` или `CrossSystemDetector`
- Верните `(score, details)` где score в [0, 1]
- Метод автоматически попадёт в JSON и сравнение

---

## Ожидаемые результаты

На основе данных из CSV (PUT vs UDMURT vs VAS):

| Метод | Ожидаемый AUC | Физическая основа |
|-------|---------------|-------------------|
| `iris_aperture_ratio` | **0.90+** | Зрачок кажется больше при маске |
| `aperture_shrinkage` | **0.88+** | Глазная щель сужена на 30-40% |
| `eyelid_sss` | **0.85+** | R-B diff: кожа >15, силикон <5 |
| `brow_lid_gap` | **0.85+** | Зазор бровь-веко +52% при маске |
| `orbit_area_ratio` | **0.82+** | Орбиты меньше на 33% |
| `eye_depth_anomaly` | **0.80+** | Глаза утоплены глубже |
| `landmark_discrepancy` | **0.80+** | 3DDFA и MP расходятся в зоне глаз |

**Combined AUC (все 25 методов):** ~0.95

---

## Файлы проекта

```
eyetest.py                    — главный скрипт (запуск из 3DDFA-V3/)
compare_eyetest.py            — сравнение двух JSON
QUICK_START.md                — этот файл
EYETEST_README.md             — полное описание
MASK_DETECTION_PLAN.md        — план доработок (35 методов, документы 1-3)
MASK_DETECTION_3DDFA_MEDIAPIPE.md  — 12 cross-system методов
MASK_DETECTION_15_MORE_METHODS.md  — 15 расширенных методов
```

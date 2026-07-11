# Eye Test — Silicon Mask Detection Suite

Тестовый набор из **25 методов** детекции силиконовых масок на основе **3DDFA-V3 + MediaPipe**.  
Особый упор на зону **глаз, век и периорбитальной области**.

---

## Что делает

Для каждого фото из папки:
1. **3DDFA-V3** — 3D реконструкция лица (35709 vertices, 68/106/134 landmarks, segmentation, UV texture)
2. **MediaPipe** — 478 landmarks + iris tracking (10 точек радужки) + face detection
3. **25 методов** анализируют данные обеих систем
4. Результат сохраняется в `eyetest.json`

Потом два JSON (реальные фото vs маска) сравниваются скриптом `compare_eyetest.py` — и видно, какие методы лучше разделяют.

---

## 25 методов

### Eye-focused (15 методов)

| # | Метод | Что измеряет |
|---|-------|-------------|
| 1 | `aperture_shrinkage` | Сужение глазной щели (aperture/IOD) |
| 2 | `iris_aperture_ratio` | Зрачок/апертура (кажется увеличенным) |
| 3 | `eyelid_thickness` | Толщина века (двойной слой силикон+кожа) |
| 4 | `lid_smoothness` | Гладкость века (Laplacian variance) |
| 5 | `lash_invisibility` | Отсутствие ресниц (Gabor energy) |
| 6 | `periocular_discontinuity` | Скачок текстуры вокруг глаз (LBP hist diff) |
| 7 | `eyelid_edge_sharpness` | Резкость края века (gradient at edge) |
| 8 | `eye_depth_anomaly` | Глубина глаз из 3DDFA Z (маска утопляет) |
| 9 | `eye_symmetry_anomaly` | Асимметрия апертур L/R |
| 10 | `brow_lid_gap` | Зазор бровь-веко (UDMURT +52%) |
| 11 | `sclera_iris_boundary` | Резкость границы склера-радужка |
| 12 | `pupil_apparent_size` | Видимый размер зрачка/площадь глаза |
| 13 | `orbit_area_ratio` | Площадь орбит из 3DDFA segmentation |
| 14 | `periocular_lbp_entropy` | Энтропия LBP вокруг глаз (маска=гладко) |
| 15 | `eyelid_sss` | R-B diff на веках (SSS) |

### Cross-system (10 методов)

| # | Метод | Что измеряет |
|---|-------|-------------|
| 16 | `landmark_discrepancy` | Расхождение 3DDFA vs MP landmarks (eye zone) |
| 17 | `specular_brdf` | Характер бликов (sharp + elongated = силикон) |
| 18 | `eye_contour_divergence` | Расхождение контура глаз между системами |
| 19 | `bbox_expansion` | Маска расширяет лицо (3DDFA bbox > MP bbox) |
| 20 | `subsurface_violation` | R/B на тонких зонах + R-G корреляция |
| 21 | `nasolabial_suppression` | Подавление носогубных складок |
| 22 | `ear_texture_cliff` | Скачок текстуры у ушей |
| 23 | `gaze_inconsistency` | Несоответствие направления взгляда |
| 24 | `mesh_curvature_anomaly` | Гладкость mesh (маска=low curvature variance) |
| 25 | `skin_tone_mismatch` | Расхождение цвета UV vs image-space |

---

## Установка

### 1. Клонировать 3DDFA-V3 и установить зависимости
```bash
git clone https://github.com/wang-zidu/3DDFA-V3
cd 3DDFA-V3

# Установить по README 3DDFA-V3
conda create -n TDDFAV3 python=3.8
conda activate TDDFAV3
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
pip install -r requirements.txt
```

### 2. Установить MediaPipe и SciPy
```bash
pip install mediapipe scipy
```

### 3. Скопировать скрипты в директорию 3DDFA-V3
```bash
cp eyetest.py /path/to/3DDFA-V3/
cp compare_eyetest.py /path/to/3DDFA-V3/
```

---

## Запуск

### Шаг 1: Прогнать на папке с реальными фото
```bash
python eyetest.py --inputpath /path/to/real_photos --output eyetest_real.json
```

### Шаг 2: Прогнать на папке с фото в маске
```bash
python eyetest.py --inputpath /path/to/mask_photos --output eyetest_mask.json
```

### Шаг 3: Сравнить результаты
```bash
python compare_eyetest.py eyetest_real.json eyetest_mask.json
```

### Параметры
```bash
python eyetest.py \
    --inputpath ./my_photos/ \
    --output results.json \
    --device cuda \          # или cpu
    --backbone resnet50 \    # или mbnetv3
    --detector retinaface
```

---

## Формат вывода

### eyetest.json
```json
{
  "input_path": "/path/to/photos",
  "n_images": 50,
  "method_statistics": {
    "aperture_shrinkage": {"mean": 0.45, "std": 0.12, ...},
    ...
  },
  "results": [
    {
      "filename": "photo1.jpg",
      "image_size": [1920, 1080],
      "eye_scores": {
        "aperture_shrinkage": 0.65,
        "iris_aperture_ratio": 0.72,
        "eyelid_thickness": 0.40,
        ...
      },
      "cross_scores": {
        "landmark_discrepancy": 0.55,
        "specular_brdf": 0.30,
        ...
      },
      "combined_mean": 0.48,
      "combined_max": 0.72,
      "eye_mean": 0.52,
      "cross_mean": 0.42,
      "top5_methods": [["iris_aperture_ratio", 0.72], ...]
    }
  ]
}
```

### eyetest_comparison.json (после сравнения)
```json
{
  "method_comparisons": [
    {
      "method": "iris_aperture_ratio",
      "auc": 0.92,
      "cohens_d": 2.1,
      "real_mean": 0.15,
      "mask_mean": 0.72,
      "p_value": 1.2e-8,
      ...
    }
  ]
}
```

---

## Как интерпретировать

| Метрика | Значение |
|---------|----------|
| **AUC = 0.5** | Метод не разделяет (случайно) |
| **AUC = 0.7** | Умеренное разделение |
| **AUC = 0.8** | Хорошее разделение |
| **AUC > 0.9** | Отличное разделение |
| **Cohen's d > 1.0** | Сильный эффект |
| **p < 0.01** | Статистически значимо |

---

## Особенности

- **Работает с полупрофильными фото** — все методы используют нормализацию на IOD
- **Один видимый глаз** — каждый метод считает для L и R отдельно, берёт доступный
- **Не требует GPU для MediaPipe** — работает на CPU
- **3DDFA-V3** — нужен GPU для быстрого инференса (CPU тоже работает, но медленно)

---

## Файлы

| Файл | Описание |
|------|----------|
| `eyetest.py` | Главный тестовый скрипт (запускать из 3DDFA-V3/) |
| `compare_eyetest.py` | Сравнение двух JSON (real vs mask) |
| `MASK_DETECTION_PLAN.md` | План доработок (документ 1) |
| `MASK_DETECTION_3DDFA_MEDIAPIPE.md` | 12 cross-system методов (документ 2) |
| `MASK_DETECTION_15_MORE_METHODS.md` | 15 расширенных методов (документ 3) |

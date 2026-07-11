# Полный гайд по обновлению texture-анализа кожи/силикона без обучения

Цель: сделать выявление настоящей кожи vs высокореалистичной силиконовой маски точнее и стабильнее **без обучения модели**, используя набор физически интерпретируемых texture metrics из `face_mask.png` и итоговый rule-based verdict.

Гайд написан под текущую структуру проекта `deeptn` и опирается на аудит 40 тестовых прогонов:

- `project/data/output_real30/*/face_mask.png` — real skin
- `project/data/output_silicone_30/*/face_mask.png` — silicone mask

---

## 1. Главная архитектурная идея

Не надо делать verdict по одной метрике и не надо обязательно обучать RandomForest. Правильнее сделать экспертный ансамбль правил:

```text
face_mask.png
  ↓
TextureExtractor: FFT / spectral / pores / GLCM / GLRLM / GLSZM / WLD / LBP / quality
  ↓
RuleBasedTextureSkinClassifier
  ↓
texture_silicone_prob, texture_real_prob, texture_skin_confidence, top_rules
  ↓
stage3/stage5 verdict
```

Каждая метрика даёт signed evidence:

```text
+1  = сильный признак силикона
 0  = нейтрально
-1  = сильный признак настоящей кожи
```

Итоговый логит:

```python
logit = sum(weight_i * normalized_evidence_i)
texture_silicone_prob = sigmoid(logit)
```

Вердикт лучше делать не бинарным по `0.5`, а с зоной неопределённости:

```text
p <= 0.35      → real likely
0.35 < p < 0.62 → unknown / mixed evidence
p >= 0.62      → silicone suspected
```

---

## 2. Что показал аудит `face_mask.png`

Я пересчитал метрики напрямую из `face_mask.png`, не из готовых `texture_metrics.json`.

Скрипт:

```bash
python tools/audit_face_mask_metrics.py
```

Создаёт:

```text
FACE_MASK_METRICS_AUDIT.md
face_mask_recomputed_texture_metrics.csv
face_mask_recomputed_texture_metrics_auc.csv
face_mask_recomputed_texture_metrics_thresholds.csv
```

### Самые сильные метрики

| Метрика | AUC abs | Направление силикона | Смысл |
|---|---:|---|---|
| `fft_high_low_ratio` | 0.943 | ниже | у силикона меньше хаотической микровысокочастотной энергии кожи |
| `spectral_slope_beta` | 0.748 | выше | материал визуально глаже, спектр круче |
| `tv_residual_sparsity` | 0.740 | ниже | меньше естественного sparse microrelief |
| `pore_density_r2_mpx` | 0.733 | ниже | меньше настоящей мелкой пористости |
| `wld_joint_entropy` | 0.723 | ниже | меньше хаоса локальных градиентов |
| `glrlm_sre` | 0.718 | ниже | длиннее гладкие run-зоны |
| `glszm_small_area_emphasis` | 0.708 | ниже | меньше мелких gray-level зон |
| `glcm_diss_d3_aniso` | 0.700 | выше | выше направленная текстурная анизотропия |
| `bimodality_ashman_D` | 0.695 | выше | сильнее двухмодальность яркости/материала |
| `lacunarity` | 0.688 | выше | менее естественная фрактальная неоднородность |

Лучший одиночный диагностический порог:

```text
fft_high_low_ratio <= 0.0814788 → silicone
accuracy на текущей выборке: 0.875
```

Но в продакшене нельзя полагаться на один порог. Нужен кворум нескольких семейств.

---

## 3. Обновить `TextureSkinClassifierV2`: rule-based режим по умолчанию

Файл:

```text
project/s2_metrics/modules/texture/classifier.py
```

### 3.1. Убрать зависимость от `.pkl` как обязательную

Вверху файла добавить:

```python
MODEL_PATH = Path(__file__).parent / "skin_classifier_v2.pkl"

# По умолчанию работаем без обучения и без загружаемой ML-модели.
RULES_ONLY_DEFAULT = True
```

Конструктор должен выглядеть так:

```python
def __init__(self, model_path: str | Path | None = None, rules_only: bool = RULES_ONLY_DEFAULT) -> None:
    path = Path(model_path) if model_path else MODEL_PATH
    self.rules_only = bool(rules_only)
    self._pipeline = None
    self._feature_names: List[str] = []
    if (not self.rules_only) and path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._pipeline = data["pipeline"]
            self._feature_names = data.get("feature_names", [])
        except Exception:
            self._pipeline = None
            self._feature_names = []
```

Так обычный вызов:

```python
TextureSkinClassifier()
```

будет работать без обучения. Если когда-нибудь захочешь сравнить с обученной моделью:

```python
TextureSkinClassifier(rules_only=False)
```

---

## 4. Добавить мягкий quality gate

Не надо сразу выбрасывать фото в `unknown`, если JPEG/blockiness слегка плохие. Сильный texture evidence должен выживать.

Добавить два уровня gate:

```python
QUALITY_GATE_THRESHOLDS = {
    "overall_quality_min": 0.28,
    "sharpness_score_min": 25.0,
    "noise_level_max": 8.0,
    "jpeg_blockiness_max": 2.0,
}

EXTREME_QUALITY_GATE = {
    "overall_quality_min": 0.18,
    "sharpness_score_min": 10.0,
    "noise_level_max": 12.0,
    "jpeg_blockiness_max": 2.6,
}
```

Логика:

```text
extreme bad quality → hard unknown
moderately bad quality → logit *= 0.85 и повышаем threshold
normal quality → обычный rule score
```

Функция:

```python
def _quality_state(self, q: Dict[str, float]) -> Tuple[bool, bool, str]:
    overall_q = float(q.get("overall_quality", 1.0) or 0.0)
    sharpness = float(q.get("sharpness_score", q.get("q_laplacian_var", 1000.0)) or 0.0)
    noise = float(q.get("noise_level", q.get("q_noise_sigma", 0.0)) or 0.0)
    blockiness = float(q.get("jpeg_blockiness", q.get("q_jpeg_blockiness", 1.0)) or 0.0)

    hard = (
        overall_q < EXTREME_QUALITY_GATE["overall_quality_min"] or
        sharpness < EXTREME_QUALITY_GATE["sharpness_score_min"] or
        noise > EXTREME_QUALITY_GATE["noise_level_max"] or
        blockiness > EXTREME_QUALITY_GATE["jpeg_blockiness_max"]
    )
    soft = (
        overall_q < QUALITY_GATE_THRESHOLDS["overall_quality_min"] or
        sharpness < QUALITY_GATE_THRESHOLDS["sharpness_score_min"] or
        noise > QUALITY_GATE_THRESHOLDS["noise_level_max"] or
        blockiness > QUALITY_GATE_THRESHOLDS["jpeg_blockiness_max"]
    )
    reason = f"q={overall_q:.2f}, sharp={sharpness:.0f}, noise={noise:.1f}, jpeg={blockiness:.2f}"
    return hard, soft, reason
```

---

## 5. Rule-based scoring: основной код

В `classifier.py` добавить метод `_heuristic_classify()`.

Рекомендуемый стартовый rule set:

```python
def _heuristic_classify(self, metrics: Dict[str, float], q: Dict[str, float], soft_gated: bool, quality_reason: str) -> Dict[str, Any]:
    rules: List[Dict[str, float | str]] = []
    logit = 0.0

    def m(name: str, default: float) -> float:
        try:
            v = float(metrics.get(name, default))
            return v if np.isfinite(v) else default
        except Exception:
            return default

    logit += self._add_rule(
        rules, "fft_high_low_ratio",
        (0.095 - m("fft_high_low_ratio", 0.095)) / 0.055,
        2.20,
        "silicone: depressed chaotic micro high-frequency energy",
    )
    logit += self._add_rule(
        rules, "spectral_slope_beta",
        (m("spectral_slope_beta", 2.75) - 2.75) / 0.55,
        1.00,
        "silicone: steeper 1/f^beta spectrum / smoother material",
    )
    logit += self._add_rule(
        rules, "wld_joint_entropy",
        (6.15 - m("wld_joint_entropy", 6.15)) / 0.35,
        0.55,
        "silicone: lower local gradient-orientation entropy",
    )
    logit += self._add_rule(
        rules, "pore_density_r2_mpx",
        (28000.0 - m("pore_density_r2_mpx", 28000.0)) / 9000.0,
        0.55,
        "silicone: reduced small pore/blob density",
    )
    logit += self._add_rule(
        rules, "glrlm_sre",
        (0.42 - m("glrlm_sre", 0.42)) / 0.12,
        0.45,
        "silicone: fewer short runs / longer smooth runs",
    )
    logit += self._add_rule(
        rules, "glszm_small_area_emphasis",
        (0.44 - m("glszm_small_area_emphasis", 0.44)) / 0.08,
        0.45,
        "silicone: fewer small gray-level zones",
    )
    logit += self._add_rule(
        rules, "glcm_diss_d3_aniso",
        (m("glcm_diss_d3_aniso", 0.07) - 0.07) / 0.055,
        0.45,
        "silicone: stronger directional GLCM dissimilarity anisotropy",
    )
    logit += self._add_rule(
        rules, "lacunarity",
        (m("lacunarity", 2.0) - 2.0) / 0.28,
        0.35,
        "silicone: larger gaps / less fractal pore randomness",
    )
    logit += self._add_rule(
        rules, "seam_score",
        (m("seam_score", 0.03) - 0.03) / 0.07,
        0.35,
        "silicone: boundary/seam discontinuity",
    )
    logit += self._add_rule(
        rules, "tv_residual_sparsity",
        (0.79 - m("tv_residual_sparsity", 0.79)) / 0.035,
        0.35,
        "silicone: less sparse natural micro residual after TV denoise",
    )

    if soft_gated:
        logit *= 0.85

    prob_silicone = float(1.0 / (1.0 + np.exp(-logit)))
    prob_real = float(1.0 - prob_silicone)
    confidence = float(max(prob_real, prob_silicone))
    adaptive_thresh = max(self._adaptive_threshold(q, soft_gated), 0.62)

    if confidence < adaptive_thresh:
        hint = "unknown"
    else:
        hint = "silicone" if prob_silicone >= 0.5 else "real"

    strongest = sorted(rules, key=lambda r: abs(float(r["signed_evidence"])), reverse=True)[:5]

    return {
        "texture_skin_hint": hint,
        "texture_skin_confidence": confidence,
        "posterior": {"real": prob_real, "silicone": prob_silicone},
        "used_metrics": [str(r["metric"]) for r in rules],
        "model_loaded": False,
        "heuristic_fallback": True,
        "rule_based": True,
        "quality_gated": False,
        "quality_soft_gated": soft_gated,
        "quality_reason": quality_reason if soft_gated else "ok",
        "quality_threshold": adaptive_thresh,
        "heuristic_logit": float(logit),
        "heuristic_top_rules": strongest,
    }
```

Нужны helpers:

```python
@staticmethod
def _clip_unit(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, -1.0, 1.0))


def _add_rule(self, rules: List[Dict[str, float | str]], name: str, raw: float, weight: float, why: str) -> float:
    value = self._clip_unit(raw)
    contribution = float(weight * value)
    rules.append({
        "metric": name,
        "signed_evidence": contribution,
        "normalized": value,
        "why": why,
    })
    return contribution
```

---

## 6. Исправить `stage2`: сначала physical aux, потом классификация

Файл:

```text
project/s2_metrics/engine.py
```

Проблема была в порядке:

```text
texture_hint = classify(texture)
physical_features = extract(...)
texture.update(physical_features)
```

То есть `seam_score`, `sss_index`, `specular_sharpness` не участвовали.

Должно быть так:

```python
# Merge Tier 3 physical aux into texture BEFORE classification.
texture.update(physical_features)

texture_hint = self.texture_classifier.classify(texture, info.quality)
posterior = texture_hint.get("posterior", {}) if isinstance(texture_hint, dict) else {}
try:
    texture["texture_silicone_prob"] = float(posterior.get("silicone", 0.5))
    texture["texture_real_prob"] = float(posterior.get("real", 0.5))
    texture["texture_skin_confidence"] = float(texture_hint.get("texture_skin_confidence", 0.0))
except Exception:
    texture["texture_silicone_prob"] = 0.5
    texture["texture_real_prob"] = 0.5
    texture["texture_skin_confidence"] = 0.0
```

В `metric_notes` добавить debug:

```python
metric_notes["texture_classifier_model_loaded"] = str(texture_hint.get("model_loaded", False)).lower()
metric_notes["texture_classifier_heuristic_fallback"] = str(texture_hint.get("heuristic_fallback", False)).lower()
metric_notes["texture_silicone_prob"] = str(texture.get("texture_silicone_prob", 0.5))
metric_notes["texture_real_prob"] = str(texture.get("texture_real_prob", 0.5))
metric_notes["texture_quality_reason"] = str(texture_hint.get("quality_reason", "ok"))
if texture_hint.get("heuristic_top_rules"):
    metric_notes["texture_heuristic_top_rules"] = str(texture_hint.get("heuristic_top_rules"))
```

---

## 7. Исправить `stage3`: не терять probability

Файл:

```text
project/s3_identity/engine.py
```

При загрузке `texture_metrics.json` надо восстановить hint из `texture_silicone_prob`:

```python
tex = tex or {}
silicone_prob = float(tex.get("texture_silicone_prob", 0.5)) if isinstance(tex, dict) else 0.5
if silicone_prob >= 0.65:
    texture_skin_hint = "silicone"
elif silicone_prob <= 0.35:
    texture_skin_hint = "real"
else:
    texture_skin_hint = "unknown"

records.append(Stage2Record(
    ...
    texture=tex,
    texture_skin_hint=texture_skin_hint,
    texture_skin_confidence=float(abs(silicone_prob - 0.5) * 2.0),
))
```

И в `_texture_suspicion()` первым делом использовать готовый probability:

```python
def _texture_suspicion(self, record: Stage2Record, reference: CalibrationReference, stage1: Stage1Record | None = None) -> float:
    if "texture_silicone_prob" in record.texture:
        try:
            return float(np.clip(float(record.texture["texture_silicone_prob"]), 0.0, 1.0))
        except Exception:
            pass

    # fallback старой логики ниже
```

Почему это важно: старая логика искала только ключи `texture_*`, но основные metrics называются без префикса, поэтому texture suspicion часто становился нулём.

---

## 8. Исправить `stage5`: continuous probability в итоговый H1

Файл:

```text
project/s5_verdict/engine.py
```

Если `stage3` отсутствует, восстановить skin hint:

```python
if stage3:
    skin_hint = stage3.skin_hint
else:
    silicone_prob = float(record.texture.get("texture_silicone_prob", 0.5)) if record.texture else 0.5
    skin_hint = "silicone" if silicone_prob >= 0.65 else ("real" if silicone_prob <= 0.35 else "unknown")
```

Добавить continuous boost:

```python
texture_silicone_prob = float(record.texture.get("texture_silicone_prob", 0.5)) if record.texture else 0.5
silicone_boost = max(0.0, (texture_silicone_prob - 0.5) * 0.4)
if skin_hint == "silicone":
    silicone_boost = max(silicone_boost, 0.2)
```

Добавить в reasoning/evidence:

```python
f"texture_silicone_prob={texture_silicone_prob:.3f}"
```

и:

```python
"texture_silicone_prob": texture_silicone_prob
```

---

## 9. Добавить audit tools

### 9.1. Проверка rule-based классификатора

Файл:

```text
tools/evaluate_texture_classifier.py
```

Запуск:

```bash
python tools/evaluate_texture_classifier.py
```

Опционально записать probability в существующие `texture_metrics.json`:

```bash
python tools/evaluate_texture_classifier.py --write-prob
```

Ожидаемый текущий результат на 40 тестах:

```text
ROC AUC: 0.9225
Hint coverage: 33/40 classified, 7 unknown
Hint precision on classified subset: 0.909
Accuracy if unknown resolved by p>=0.5: 0.85
```

### 9.2. Пересчёт метрик напрямую из `face_mask.png`

Файл:

```text
tools/audit_face_mask_metrics.py
```

Запуск:

```bash
python tools/audit_face_mask_metrics.py
```

Создаёт:

```text
FACE_MASK_METRICS_AUDIT.md
face_mask_recomputed_texture_metrics.csv
face_mask_recomputed_texture_metrics_auc.csv
face_mask_recomputed_texture_metrics_thresholds.csv
```

---

## 10. Рекомендуемая финальная логика verdict

### 10.1. Не использовать один порог

Плохо:

```python
if fft_high_low_ratio < 0.081:
    silicone
```

Хорошо:

```text
silicone suspected, если совпали 3+ независимых семейства:
- FFT micro-frequency низкий
- spectral slope высокий
- pore/blob density низкий
- WLD entropy низкий
- GLRLM/GLSZM smoothness
- GLCM anisotropy высокий
- seam/boundary высокий, если доступен
```

### 10.2. Вердикт по зонам

```python
if texture_silicone_prob >= 0.70:
    verdict = "silicone_suspected_high_confidence"
elif texture_silicone_prob >= 0.62:
    verdict = "silicone_suspected"
elif texture_silicone_prob <= 0.30:
    verdict = "real_skin_likely_high_confidence"
elif texture_silicone_prob <= 0.35:
    verdict = "real_skin_likely"
else:
    verdict = "unknown_mixed_texture_evidence"
```

### 10.3. Объяснение verdict

В отчёте всегда показывать top rules:

```text
texture_silicone_prob = 0.91
Вердикт: silicone suspected
Главные причины:
- fft_high_low_ratio низкий: reduced chaotic high-frequency skin energy
- spectral_slope_beta высокий: smoother 1/f spectrum
- pore_density_r2_mpx низкий: reduced pore density
- wld_joint_entropy низкий: reduced gradient entropy
- glrlm_sre низкий: longer smooth runs
```

---

## 11. Что улучшить в `TextureExtractor` следующим этапом

Это не обязательно для первого обновления, но даст прирост качества.

### 11.1. Patch-based FFT вместо одного центрального патча

Сейчас часть FFT/спектральных метрик может зависеть от центрального crop и границ маски. Лучше считать по многим патчам внутри маски:

```text
найти 40x40 / 56x56 патчи, полностью внутри skin mask
для каждого посчитать FFT features
взять median + IQR
```

Добавить метрики:

```text
pfft_high_low_median
pfft_high_low_iqr
pfft_beta_median
pfft_beta_iqr
```

### 11.2. Улучшить pore detector

Текущий `white_tophat disk(2)` полезен, но лучше добавить multi-scale:

```text
disk(1), disk(2), disk(3)
blob area filter
eccentricity filter
density per Mpx
median area
IQR density по патчам
```

Новые признаки:

```text
pore_density_ms_mpx
pore_density_patch_iqr
pore_area_median
pore_eccentricity_iqr
```

### 11.3. Отдельно учитывать границу маски

Для силиконовой маски важны seam/boundary. Если нет landmarks, можно сделать fallback по alpha-boundary:

```text
skin_mask distance transform
boundary band: dt 2..8 px
inner band: dt 15..30 px
сравнить GLCM contrast / color / gradient
```

Новая метрика:

```text
mask_boundary_texture_jump
```

---

## 12. Важное предупреждение про cohort anomaly

`CohortTextureAnomalyDetectorV2` нельзя обучать на той же пачке, где есть подозрительные silicone cases.

Проблема:

```text
если VAS-era содержит в основном силиконовые маски,
то silicone становится baseline,
и detector считает силикон нормой
```

Правильно:

```text
real-only baseline per era/quality
или отключить cohort anomaly для H1 synthetic,
или использовать только rule-based texture_silicone_prob
```

---

## 13. Проверка после обновления

### 13.1. Компиляция

```bash
python -m py_compile \
  project/s2_metrics/modules/texture/classifier.py \
  project/s2_metrics/engine.py \
  project/s3_identity/engine.py \
  project/s5_verdict/engine.py \
  tools/evaluate_texture_classifier.py \
  tools/audit_face_mask_metrics.py
```

### 13.2. Аудит классификатора

```bash
python tools/evaluate_texture_classifier.py
```

### 13.3. Пересчёт метрик из PNG

```bash
python tools/audit_face_mask_metrics.py
```

### 13.4. Запись вероятностей в тестовые JSON

```bash
python tools/evaluate_texture_classifier.py --write-prob
```

После этого в `texture_metrics.json` должны появиться:

```json
{
  "texture_silicone_prob": 0.91,
  "texture_real_prob": 0.09,
  "texture_skin_confidence": 0.91
}
```

---

## 14. Итоговая схема обновления

Минимальный набор изменений:

```text
1. classifier.py
   - RULES_ONLY_DEFAULT = True
   - soft/hard quality gate
   - _heuristic_classify()
   - texture_silicone_prob/posterior/top_rules

2. s2_metrics/engine.py
   - classify после physical_features
   - сохранить texture_silicone_prob в texture_metrics.json
   - сохранить debug в metric_notes

3. s3_identity/engine.py
   - восстановить texture_skin_hint из texture_silicone_prob
   - _texture_suspicion() использует texture_silicone_prob

4. s5_verdict/engine.py
   - H1 synthetic получает continuous silicone boost
   - evidence/reasoning содержит texture_silicone_prob

5. tools/
   - evaluate_texture_classifier.py
   - audit_face_mask_metrics.py
```

Главная идея: система должна выносить решение не потому, что обученная модель так сказала, а потому что несколько независимых texture-семейств согласованно указывают на материал без естественной кожи.

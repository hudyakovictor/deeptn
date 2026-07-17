# QA report — lead-aware pipeline v1.4

## Реализовано

- Отдельные identity-only LDM106/LDM134.
- Per-landmark point-motion calibration по pose bin.
- 13 локальных shape families для каждой LDM134.
- Per-landmark × family calibration median/MAD/p95/count.
- Импорт архива зацепок как coverage targets, а не ground truth.
- Автоматическое соответствие legacy metric → new measurement family.
- Хронологический pace-анализ с учётом количества дней между кадрами.
- Статусы `same_day_structural_conflict`, `biologically_improbable_rate_candidate`, `persistent_biologically_improbable_change`.
- Расследовательский HTML с разделом перепроверки архива, timelines, point-motion map и pace-aware change table.
- Texture-image канал усилен до LBP/GLCM/frequency descriptors: `lbp_chi2_delta`, GLCM contrast/homogeneity/energy/correlation deltas и high-frequency ratio.
- Добавлен безопасный fallback без обязательной установки `scikit-image`; если `scikit-image` доступен в production, используется его `local_binary_pattern`/`graycomatrix`.
- Evidence packets теперь сохраняют расширенные texture-поля, чтобы Stage 2/3 не теряли независимый кожно-текстурный канал.

## Проверки

- `compileall`: pass.
- 23 unit tests: pass.
- Texture descriptor unit test: pass.
- Synthetic end-to-end: pass.
- Намеренно внесённое локальное изменение найдено как coherent/persistent change.
- Темп изменений на коротком интервале маркируется chronology-rate detector.
- Stage 2 validation: complete.
- Stage 3 validation: complete.
- Browser QA desktop/mobile: no console errors; на mobile `scrollWidth == clientWidth`.

## Ограничение

Реальный Stage 1 inference и полный batch 1900+ не выполнялись из-за отсутствия model weights и основного датасета в sandbox. Перед production run обязательны gates 2/10/100/full.

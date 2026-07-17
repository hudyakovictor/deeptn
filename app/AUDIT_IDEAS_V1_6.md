# Дополнительные идеи аудита — внедрение v1.6

## Добавлено

- Масштабно-зависимая эрозия границ texture ROI: края warp, волос и бровей меньше влияют на результат.
- Сетка 4×3 патчей внутри ROI: один блик или локальный артефакт не определяет метрику всей зоны.
- Локальная энтропия кожи и patch-level median/MAD.
- Gabor bank: два масштаба × четыре ориентации; сохраняется ориентационный профиль и междатный RMSE.
- Расширенные texture evidence fields: entropy, Gabor, patch reproducibility и erosion metadata.
- Pose-leakage diagnostic: для основных geometry/mesh metrics проверяется ранговая зависимость residual от pose distance после нормализации.
- `pose_leakage_diagnostic.json` входит в обязательные артефакты и manifest.

## Уже покрывалось до v1.6

- canonical/object-normalized geometry;
- intersection visibility;
- identity-only mesh и alpha channels;
- pose bins;
- robust trimmed alignment;
- point-to-plane dense mesh residual;
- LBP, GLCM и frequency analysis;
- baseline return;
- cross-bin corroboration;
- source/event aggregation;
- multiple-testing correction;
- private prior corroboration без обратного влияния на blind analysis.

## Следующая очередь, требующая реальных данных или дополнительных зависимостей

1. Crop-jitter ensemble с 5–10 повторными 3DDFA-реконструкциями.
2. UV subpixel registration и registration-error map.
3. Frangi/Sato/Meijering wrinkle probability maps.
4. Skeleton/Skan graph extraction и branch matching.
5. Within-date median ROI template, leave-one-image-out и bootstrap CI.
6. Dense connected-cluster significance относительно per-vertex calibration null.
7. Camera/codec/domain strata по достоверным metadata.
8. Synthetic pose rerender/reconstruction calibration.

Эти пункты нельзя честно объявлять готовыми без весов модели, полного набора фотографий, связанных event/source metadata и полноценной калибровки.

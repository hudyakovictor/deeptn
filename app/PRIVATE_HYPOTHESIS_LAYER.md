# Приватный слой перепроверки гипотез

## Назначение

Этот раздел изолирован от публичного Stage 3. Он импортирует старые гипотезы как цели повторной проверки, но не использует старые posterior, thresholds и диапазоны как доказательство или текущую калибровку.

## Покрытие приложенного add.zip

- Доступных источников гипотез: 16 из 16.
- Найдено записей гипотез/событий/проверок: 6 223.
- Импортировано без потери исходного payload: 6 223.
- Покрытие импорта: 100%.
- Минимальное требование: 95%.
- Семейств гипотез в новом реестре: 18.

В приватный ledger входят photo-level hypotheses, полные forensic verdict payloads, chronology events, top breaks, identity stress records, evidence packets, canonical anomaly entries, cross-bucket consensus, H0 contradictions, morphing candidates, era breakpoints, bucket consistency, natural-texture baseline, mesh-noise report, calibration health и metric coverage.

## Смена выравнивания

Старый manifest содержит `reference_use_mesh_alignment=false`. Новая версия использует `iteratively_trimmed_kabsch_v1_no_scale`. Поэтому:

- старые числовые диапазоны не переносятся;
- старые posterior не считаются актуальными;
- старые anomaly thresholds не используются для новых статусов;
- каждый диапазон должен быть пересчитан на current calibration после нового alignment;
- старые результаты сохраняются только как provenance и retest targets.

## Статусы приватного ретеста

- `retested_with_current_alignment` — найдено соответствие в новых Stage-2 парах.
- `pending_missing_current_data` — гипотеза сохранена, но текущих Stage-1/Stage-2 данных для честного ретеста нет.
- `technical_anomaly_candidate` — новый технический канал снова дал candidate-сигнал.
- `within_current_noise_or_no_strong_change` — сильный повторный сигнал не получен.
- `inconclusive` — качество, калибровка или конфликт каналов не позволяют интерпретацию.

## Текущее состояние seed-раздела

Все 6 223 записи импортированы, но 6 223 имеют `pending_missing_current_data`, потому что add.zip содержит результаты старой реализации, а не новый полный Stage-1 extraction и Stage-2 output с новым alignment. Это преднамеренная защита от ложного заявления, что старые числа уже перепроверены новой геометрией.

После выполнения нового Stage 1 и Stage 2 запустить:

```bash
python3 -m app6.run_private_hypotheses CURRENT_STAGE2_OUTPUT UNPACKED_ADD_ZIP PRIVATE_OUTPUT
```

Модуль автоматически сопоставит photo/date targets с новыми парами, пересчитанными диапазонами и текущим metric catalog.

## Граница вывода

Приватный слой проверяет технические гипотезы согласованности. Он не превращает аномалию в автоматическое утверждение о личности, маске, операции, материале или медицинской причине.

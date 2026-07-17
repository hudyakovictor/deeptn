# DEEPUTIN app6

Полностью переписанный этап 1 для 3DDFA_V3. Главная цель — один раз извлечь и атомарно сохранить все дорогие данные, после чего этапы анализа и визуализации не запускают нейросеть повторно.

## Что исправлено

- один вызов `net_recon` на фотографию;
- корректные `alpha_alb` и `alpha_sh`;
- identity+expression и identity-only mesh;
- object/normalized/canonical/camera/image-224 representations;
- LDM106/LDM134 и официальные vertex indices;
- front-facing, renderer и combined visibility полного mesh;
- исходные восемь semantic channels;
- skin+nose mask с явной policy и без пространственно ложного resize fallback;
- UV analysis, beauty, observed mask, confidence, original mask и triangle visibility;
- hash-based photo ID;
- строгие даты только из `YYYY_MM_DD[_N]`;
- atomic directory commit;
- validator и hash-aware resume;
- технический QA без verdict и искусственного overall score;
- correspondence-based reprojection checks.

## Обязательные assets

Фактическая структура проекта на Mac должна быть такой:

```text
/Users/victorkhudyakov/myproject/
├── app/                    # этот архив
├── 3ddfa_v3/
│   ├── assets/
│   ├── model/
│   └── uv_module/
├── dataset/
└── old/
```

Обязательные веса находятся не внутри `app`, а в соседней библиотеке `3ddfa_v3`:

- `3ddfa_v3/assets/face_model.npy`;
- `3ddfa_v3/assets/net_recon.pth` либо `3ddfa_v3/assets/net_recon_mbnet.pth`;
- `3ddfa_v3/assets/large_base_net.pth`.

В приложенном пользовательском архиве этих весов нет; app2 завершит работу до batch run с понятной ошибкой, а не создаст частичные результаты.

## Имена фотографий

Допустимы только:

```text
1999_01_11.jpg
1999_01_11_2.jpg
1999_01_11_3.png
```

EXIF не читается.

## MacBook M1

Используйте CPU. Bundled renderer 3DDFA_V3 не поддерживает MPS как полноценный backend: MPS попадает в ветку nvdiffrast. `--device auto` на macOS автоматически выбирает CPU. Conda не обязательна.

## Smoke run

```bash
cd /Users/victorkhudyakov/myproject
python3 app/run_stage1.py \
  --input dataset/main \
  --output results/app2_smoke \
  --device cpu \
  --limit 2 \
  --fail-fast
```

Повторите ту же команду: обе записи должны быть пропущены валидным resume.

## Batch gates

```bash
# 10 фото
python3 app/run_stage1.py --input dataset/main --output results/stage1_v2 --device cpu --limit 10 --fail-fast

# 100 фото — убрать fail-fast, чтобы оценить error rate
python3 app/run_stage1.py --input dataset/main --output results/stage1_v2 --device cpu --limit 100

# полный набор
python3 app/run_stage1.py --input dataset/main --output results/stage1_v2 --device cpu
```

Не запускайте полный набор, пока 100-photo gate не завершился без structural validation errors.

## Тесты без весов

```bash
python3 -m unittest discover -s app/tests -v
```

## Этап 2

Полное ТЗ находится в `app/STAGE2_SPEC.md`.


## Stage 2 Workbench (integrated)

Исполняемое reference-ядро настроечного Stage 2 находится в `app/stage2/workbench/`:

- `core/` — immutable contracts, DAG и Preview impact;
- `recommendation/` — Recommended/Balanced/Strict/Sensitive presets;
- `pipeline/` — Quality/Calibration, Evidence/Fusion/Chronology и Release/Private Retest.

Оно не заменяет существующий production `app/stage2/engine.py`: интеграция добавлена отдельным namespace без разрушения текущего pipeline. Тесты находятся в `app/tests/workbench/`.

Полная проверка:

```bash
./app/scripts/run_all_tests.sh
```

## UV/morphing contract v2.2

- `morphing/mesh.obj` always retains the complete fixed topology for future morphing.
- `reconstruction.npz` contains explicit visibility-cut `analysis_mesh_*` arrays for scientific analysis.
- UV artifacts live in `uv_module/` inside each photo result.
- `uv_confidence.png` is a continuous colour confidence preview, not a binary mask.
- `uv_texture.png` is hole-free for display/morphing; raw observed texture and synthetic completion mask remain in `uv.npz` so filled pixels never count as evidence.

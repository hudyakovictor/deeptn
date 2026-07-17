# Project path contract v2.3

Expected macOS layout:

```text
/Users/victorkhudyakov/myproject/
├── app/
├── 3ddfa_v3/
│   ├── assets/
│   ├── model/
│   └── uv_module/
├── dataset/
└── old/
```

`app/run_stage1.py` resolves `myproject` automatically. It adds both `myproject` and `myproject/3ddfa_v3` to Python's import path. No files are searched under `app/3ddfa_v3`, and no absolute user path is embedded in code.

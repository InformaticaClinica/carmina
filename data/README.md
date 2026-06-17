# Data Directory

This directory is intentionally committed only as an empty skeleton. Put local datasets and generated artifacts here when running Carmina.

```text
data/
├── inputs/
│   ├── CARMEN1_mappings.tsv
│   └── txt/
│       ├── ann/
│       ├── raw/
│       ├── masked/
│       └── identify/
└── outputs/
    ├── debug/
    │   └── logs/
    ├── logs/
    └── metrics/
```

Do not commit clinical data, generated model outputs, logs, metrics, credentials, or zipped datasets. See `docs/data.md` for details.

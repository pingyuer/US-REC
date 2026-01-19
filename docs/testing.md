# Testing

## Run

```bash
pytest -q
```

## Layout
- `tests/` contains unit tests for losses/metrics/models and light integration for `main.py`.
- External-service tests are opt-in and skipped by default.

## Opt-in integration

```bash
RUN_MLFLOW_INTEGRATION=1 pytest -q tests/test_mlflow_integration.py
RUN_TUI_REAL_DATA=1 pytest -q tests/test_tui_real_data.py
```

## Notes
- Metrics/losses are covered by hand-checkable fake-data tests to reduce silent correctness bugs.
- If you run inside a restricted sandbox, network calls (e.g. albumentations version check) may warn but do not affect correctness.


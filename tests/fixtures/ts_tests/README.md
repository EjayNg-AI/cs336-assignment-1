# tests/fixtures/ts_tests

This folder contains transformer model fixtures used by tests in
[`../../test_model.py`](../../test_model.py).

## Contents

- `model_config.json`: model hyperparameters for the fixture.
- `model.pt`: serialized reference weights.

These files are test inputs, not training outputs. Keep them in sync with the
snapshot expectations in [`../../_snapshots/`](../../_snapshots/) when reference
model behavior intentionally changes.

# Tests

From the repository root, install the test dependencies and run the suite:

```sh
python -m pip install -e '.[test,crypto]'
python -m pytest -q
```

`unit/` groups protocol, arithmetic, GPU, inference, and proof checks.
`integration/` covers multi-party execution, transports, model inference, and
golden transcripts. `benchmarks/` checks benchmark fixtures and accounting.
Tests that require unavailable GPU or native proof support skip those cases.

Run one area with `python -m pytest tests/unit/arithmetic -q`. The
[`fixtures/mpspdz/`](fixtures/mpspdz/) directory holds reference programs,
transcripts, and native execution evidence; its test code lives in
`integration/test_rep4_oracle.py` and `integration/test_runtime_golden.py`.
Golden generators are under [`tools/validation/`](../tools/validation/).

# Repository tools

Run tools from the repository root after the editable installation described in
[the main README](../README.md#run-locally).

`validation/` contains the independent Rep4 oracle and MP-SPDZ golden-fixture
generators. For their commands and reference assumptions, see
[MP-SPDZ validation](../README.md#validate-against-mp-spdz).

`plots/plot_performance.py` reads retained inference results and writes the
performance figures under `docs/assets/`:

```sh
python tools/plots/plot_performance.py
```

Plot generation requires `matplotlib` in addition to the project dependencies.

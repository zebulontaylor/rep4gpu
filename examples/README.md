# Synthetic next-token example

From the repository root, run:

```sh
python -m examples.toy_next_token
```

The example trains a CPU MLP to predict `(a + b) % 8` from two tokens. It uses
one-hot inputs, 64 ReLU hidden units, and eight output logits. All 64 contexts
are used for both training and evaluation; the reported accuracy is not held-out
accuracy. This model is a synthetic training example, not a transformer.

The four-party evaluation gives the context inputs to party 0 and the model to
party 1. It shares both through the protocol, evaluates Q12 fixed-point layers,
and opens only the final logits. Every party's logits must match the integer
reference. Public fixture checks establish that accumulators fit the selected
ring before sharing.

The command writes public test artifacts to `results/toy_next_token/`:
`model.pt` contains the PyTorch state dictionary, and `metadata.json` records
training settings, ring bounds, accuracy, and logit/log-probability errors.
Use `--output PATH` to select another directory. Optional `--device cuda --fused`
runs the same checked evaluation on CUDA. This example reports correctness;
use `benchmarks.primitives` and `benchmarks.inference` for timing.

## Share a PyTorch Sequential model

`rep4gpu.nn` accepts a flat `torch.nn.Sequential` containing `Linear`, `ReLU`,
`GELU`, `SiLU`, and `Identity`. Build the public architecture on the model owner:

```python
from rep4gpu.nn import sequential_spec, share_sequential

spec = sequential_spec(module)
```

Distribute this JSON-compatible `spec` to all four parties. It contains layer
sizes and activation names, with no parameters. Each party then runs:

```python
shared_model = share_sequential(
    party, spec, module=module if party.party == 1 else None,
    owner=1, fractional_bits=12,
)
secret_logits = shared_model.forward(secret_inputs)
```

Only party 1 supplies the PyTorch model. Parameter sharing runs once; reuse
`shared_model` with new shared inputs. `forward` returns a `Share` and never
opens it. The caller chooses whether to open the result through `party.open`.
All parties must use the same validated spec and fixed-point settings and enter
collectives in the same order. The owner validates every layer and encoded
parameter before its first parameter-sharing collective; abort the transport
if any participant fails, as the runnable example does.

Adjacent `Linear` and `ReLU` layers use fused fixed-point matmul/ReLU. GELU and
SiLU use the existing [fixed-point approximations](../docs/activations.md), so
their logits can differ from PyTorch floating-point activations. GELU requires
`approximate="none"`. Establish accumulator bounds for the chosen inputs,
weights, and arithmetic ring before inference. On CUDA, set
`weight_storage="limbs"` to prepare static weights, or `"words32"` for compact
ring32 weight storage. Preparation requires CUDA and occurs before `forward`.

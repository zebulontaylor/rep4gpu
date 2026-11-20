"""A public integer model with a content-derived identity for network audits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .layer_commitments import Layer


class AuditModel:
    """Load bounded square affine/ReLU layers without executable serialization.

    Every boundary contains ``x`` with shape [batch, width]. Arithmetic uses
    signed int64 words with wrapping matrix products and addition, followed by
    ReLU. Both roles hold these public weights; this is not a private model.
    """

    def __init__(self, spec):
        if (not isinstance(spec, dict) or set(spec) != {'format', 'width', 'layers'}
                or spec['format'] != 'siv-int64-relu-v1'
                or type(spec['width']) is not int or not 1 <= spec['width'] <= 128
                or not isinstance(spec['layers'], list) or not 1 <= len(spec['layers']) <= 64):
            raise ValueError('invalid model format, width, or layer count')
        self.width = spec['width']
        layers = []
        parameters = []
        for index, item in enumerate(spec['layers']):
            if not isinstance(item, dict) or set(item) != {'weight', 'bias'}:
                raise ValueError('each layer needs weight and bias')
            weight, bias = item['weight'], item['bias']
            if (not isinstance(weight, list) or len(weight) != self.width
                    or any(not isinstance(row, list) or len(row) != self.width for row in weight)
                    or not isinstance(bias, list) or len(bias) != self.width):
                raise ValueError('weights and biases must match model width')
            if any(type(v) is not int or not -(1 << 63) <= v < (1 << 63)
                   for row in [*weight, bias] for v in row):
                raise ValueError('model values must be signed int64 integers')
            w = torch.tensor(weight, dtype=torch.int64)
            b = torch.tensor(bias, dtype=torch.int64)

            parameters.append((w, b))

            def forward(state, w=w, b=b):
                return {'x': torch.clamp_min(state['x'] @ w + b, 0)}

            layers.append(Layer(f'affine_relu_{index}', forward, preserves_inputs=True))
        canonical = json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
        self.model_id = 'sha256:' + hashlib.sha256(b'siv.audit-model.v1\0' + canonical).hexdigest()
        self.layers = tuple(layers)
        # Parameters and callbacks share these tensors; callers must not mutate them.
        self.parameters = tuple(parameters)

    @classmethod
    def load(cls, path):
        with Path(path).open('rb') as stream:
            payload = stream.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError('model file exceeds 32 MiB')
        return cls(json.loads(payload))

    def validate_state(self, state):
        if not isinstance(state, dict) or set(state) != {'x'}:
            raise ValueError('model state must contain only x')
        x = state['x']
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.int64
                or x.device.type != 'cpu' or x.ndim != 2
                or not 1 <= x.shape[0] <= 64 or x.shape[1] != self.width):
            raise ValueError('x must be a CPU int64 tensor of shape [1..64, model width]')


def example_spec():
    """Return a small model whose parameters are explicit in the setup files."""
    return {
        'format': 'siv-int64-relu-v1', 'width': 4,
        'layers': [
            {'weight': [[2, 0, -1, 0], [0, 1, 0, 1], [1, 0, 2, 0], [0, -1, 0, 1]],
             'bias': [i, -1, 1, 0]}
            for i in range(4)
        ],
    }

"""Share small public Sequential architectures without exposing their weights."""
from dataclasses import dataclass

import torch

from .arithmetic.activations import FixedPoint
from .protocol.runtime import Share


_ACTIVATIONS = {torch.nn.ReLU: 'ReLU', torch.nn.GELU: 'GELU',
                torch.nn.SiLU: 'SiLU', torch.nn.Identity: 'Identity'}


def sequential_spec(module):
    """Return a JSON-compatible public architecture, containing no parameters.

    Only a flat ``torch.nn.Sequential`` with Linear, ReLU, GELU, SiLU and
    Identity is supported. GELU uses the default ``approximate='none'`` mode.
    """
    if type(module) is not torch.nn.Sequential:
        raise TypeError('expected a flat torch.nn.Sequential')
    layers = []
    for layer in module:
        if type(layer) is torch.nn.Linear:
            if tuple(layer.weight.shape) != (layer.out_features, layer.in_features) or (
                    layer.bias is not None and tuple(layer.bias.shape) != (layer.out_features,)):
                raise ValueError('Linear parameter shapes do not match its public dimensions')
            layers.append(dict(type='Linear', in_features=layer.in_features,
                               out_features=layer.out_features, bias=layer.bias is not None))
        elif type(layer) in _ACTIVATIONS:
            if type(layer) is torch.nn.GELU and layer.approximate != 'none':
                raise ValueError("GELU requires approximate='none'")
            layers.append(dict(type=_ACTIVATIONS[type(layer)]))
        else:
            raise TypeError(f'unsupported Sequential module: {type(layer).__name__}')
    return _validate_spec(dict(version=1, layers=layers))


def _validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {'version', 'layers'} or type(spec['version']) is not int or spec['version'] != 1:
        raise ValueError('spec requires version=1 and layers')
    if not isinstance(spec['layers'], list):
        raise ValueError('spec layers must be a list')
    layers, width = [], None
    for layer in spec['layers']:
        if not isinstance(layer, dict):
            raise ValueError('each spec layer must be a dictionary')
        name = layer.get('type')
        if name == 'Linear':
            if set(layer) != {'type', 'in_features', 'out_features', 'bias'}:
                raise ValueError('Linear spec requires in_features, out_features and bias')
            if any(type(layer[key]) is not int or layer[key] < 1 for key in ('in_features', 'out_features')):
                raise ValueError('Linear dimensions must be positive integers')
            if type(layer['bias']) is not bool:
                raise ValueError('Linear bias must be a boolean')
            if width is not None and width != layer['in_features']:
                raise ValueError('adjacent Linear dimensions disagree')
            width = layer['out_features']
        elif name in _ACTIVATIONS.values():
            if set(layer) != {'type'}:
                raise ValueError(f'{name} spec only accepts type')
        else:
            raise ValueError(f'unsupported Sequential spec layer: {name!r}')
        layers.append(dict(layer))
    return dict(version=1, layers=layers)


@dataclass(frozen=True)
class _Layer:
    kind: str
    weight: object = None
    bias: object = None


class SharedSequential:
    """A shared model whose forward method never opens its inputs or outputs."""
    def __init__(self, runtime, layers, fractional_bits):
        self.runtime = runtime
        self.layers = tuple(layers)
        self.fixed_point = FixedPoint(runtime, fractional_bits)

    def forward(self, value):
        if not isinstance(value, Share) or value.binary:
            raise TypeError('forward expects an arithmetic Share')
        fp, index = self.fixed_point, 0
        while index < len(self.layers):
            layer = self.layers[index]
            if layer.kind == 'Linear':
                if index+1 < len(self.layers) and self.layers[index+1].kind == 'ReLU':
                    value = fp.matmul_relu(value, layer.weight, layer.bias)
                    index += 2
                    continue
                value = fp.matmul(value, layer.weight)
                if layer.bias is not None:
                    value = fp.add(value, layer.bias)
            elif layer.kind == 'ReLU':
                value = fp.relu(value)
            elif layer.kind == 'GELU':
                value = fp.gelu(value)
            elif layer.kind == 'SiLU':
                value = fp.silu(value)
            index += 1
        return value

    __call__ = forward


def share_sequential(rt, spec, module=None, owner=1, fractional_bits=12, weight_storage=None):
    """Input model parameters once and return a reusable shared Sequential.

    All parties must call with the same validated public ``sequential_spec``.
    Only owner supplies module; other parties supply None. Parameters are
    nearest-even encoded, transposing Linear weights for shared matmul.
    ``weight_storage='limbs'`` or ``'words32'`` prepares static CUDA weights;
    words32 requires ring32. Preparation and parameter sharing precede forward.
    Callers must establish input/intermediate ring bounds. GELU and SiLU use
    the approximations documented in ``docs/ring-protocol.md``.
    """
    spec = _validate_spec(spec)
    if type(owner) is not int or owner not in range(4):
        raise ValueError('owner must be in [0,3]')
    fp = FixedPoint(rt, fractional_bits)
    if fractional_bits >= rt.bits:
        raise ValueError('fractional_bits must be smaller than the arithmetic ring')
    if weight_storage not in (None, 'limbs', 'words32'):
        raise ValueError('weight_storage must be None, limbs or words32')
    if weight_storage is not None and torch.device(rt.device).type != 'cuda':
        raise ValueError('prepared weight storage requires CUDA')
    if weight_storage == 'words32' and rt.bits != 32:
        raise ValueError('words32 requires ring32')
    encoded = []
    if rt.party == owner:
        if module is None or sequential_spec(module) != spec:
            raise ValueError('owner module does not match the public Sequential spec')
        # Validate and encode every parameter before the first collective input.
        for layer in module:
            if type(layer) is torch.nn.Linear:
                parameters = [layer.weight.detach().T]
                if layer.bias is not None:
                    parameters.append(layer.bias.detach())
                for parameter in parameters:
                    value = fp.encode(parameter)
                    if rt.bits == 32 and not bool(((value >= -(1 << 31)) & (value < (1 << 31))).all()):
                        raise ValueError('encoded model parameter exceeds signed arithmetic ring')
                    encoded.append(value)
    elif module is not None:
        raise ValueError('only the model owner may supply module')
    layers, cursor = [], 0
    for layer in spec['layers']:
        if layer['type'] != 'Linear':
            layers.append(_Layer(layer['type']))
            continue
        weight = rt.input(owner, encoded[cursor] if rt.party == owner else None,
                          (layer['in_features'], layer['out_features']))
        cursor += 1
        bias = None
        if layer['bias']:
            bias = rt.input(owner, encoded[cursor] if rt.party == owner else None,
                            (layer['out_features'],))
            cursor += 1
        if weight_storage is not None:
            weight = rt.prepare_weight_share(weight, storage=weight_storage)
        layers.append(_Layer('Linear', weight, bias))
    return SharedSequential(rt, layers, fractional_bits)

"""Untrusted witnesses are checked privately before shared outputs are released."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.proofs.witness_activations import (
    InvalidWitness, WitnessProgram, WitnessSession, numerical_program,
)


SCALE = 1 << 24


def run_four(callback):
    torch.set_num_threads(1)
    network = IdealNetwork(timeout=30)
    seeds = make_seeds()
    parties = [Party(i, network.endpoint(i), {j: s for j, s in seeds.items() if j != i})
               for i in range(4)]

    def guarded(party):
        try:
            return callback(party)
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(guarded, parties))


def secret(party, encoded, owner=0, binary=False):
    encoded = torch.as_tensor(encoded, dtype=torch.int64)
    return party.input(owner, encoded if party.party == owner else None,
                       encoded.shape, binary=binary)


def test_reciprocal_honest_and_only_acceptance_opened():
    encoded = torch.tensor([SCALE, SCALE * 3 // 2, SCALE * 2])
    witness = torch.round(SCALE * SCALE / encoded.double()).long()

    def work(p):
        x, w = secret(p, encoded), secret(p, witness, owner=1)
        opened = []
        original_open = p.open

        def observe(value):
            result = original_open(value)
            opened.append(result.clone())
            return result

        p.open = observe
        session = WitnessSession(p)
        assert session.evaluate(numerical_program('reciprocal'), {'x': x},
                                {'reciprocal': w}) is None
        assert opened == []
        outputs = session.finish()
        assert len(outputs) == 1
        assert len(opened) == 1 and opened[0].numel() == 1
        assert int(opened[0]) in (0, 1)
        return original_open(outputs[0])

    for result in run_four(work):
        assert torch.equal(result, witness)


@pytest.mark.parametrize('bad', [-2**63, -SCALE, 0, SCALE * 2, 2**63 - 1])
def test_reciprocal_rejects_malformed_and_wrong_witness(bad):
    def work(p):
        x = secret(p, [SCALE])
        w = secret(p, [bad], owner=1)
        session = WitnessSession(p)
        session.evaluate(numerical_program('reciprocal'), {'x': x}, {'reciprocal': w})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_reciprocal_rejects_in_range_wrong_value():
    def work(p):
        session = WitnessSession(p)
        session.evaluate(numerical_program('reciprocal'),
                         {'x': secret(p, [SCALE])},
                         {'reciprocal': secret(p, [SCALE // 2], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


@pytest.mark.parametrize('bad', [-2**63, -1, 0, SCALE * 3, 2**63 - 1])
def test_reciprocal_rejects_out_of_domain_input(bad):
    def work(p):
        session = WitnessSession(p)
        session.evaluate(numerical_program('reciprocal'),
                         {'x': secret(p, [bad])},
                         {'reciprocal': secret(p, [SCALE], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_custom_program_multiple_witnesses_and_sticky_failure():
    def recompute(ctx, inputs, witnesses):
        x = ctx.bounded(inputs['x'], 0, 2)
        ctx.residual(ctx.runtime.sub(witnesses['a'], x), 0)
        ctx.residual(ctx.runtime.sub(witnesses['b'], x), 0)
        return ctx.runtime.add(witnesses['a'], witnesses['b'])

    program = WitnessProgram('double', {'a': (0, 2), 'b': (0, 2)}, recompute)

    def work(p):
        x = secret(p, [SCALE])
        good = secret(p, [SCALE], owner=1)
        bad = secret(p, [SCALE + 1], owner=1)
        session = WitnessSession(p)
        session.evaluate(program, {'x': x}, {'a': good, 'b': good})
        session.evaluate(program, {'x': x}, {'a': good, 'b': bad})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


@pytest.mark.parametrize('kind', ['silu', 'gelu', 'swiglu'])
def test_numerical_activation_honest(kind):
    from rep4gpu.proofs.witness_prover import numerical_witnesses
    x = torch.tensor([-4, -1, 0, 1, 4], dtype=torch.int64) * SCALE
    value = torch.tensor([1, 2, 0, -2, -1], dtype=torch.int64) * SCALE
    plain_inputs = {'x': x, 'value': value} if kind == 'swiglu' else {'x': x}
    claims = numerical_witnesses(kind, plain_inputs)

    def work(p):
        inputs = {'x': secret(p, x)}
        if kind == 'swiglu':
            inputs['value'] = secret(p, value)
        session = WitnessSession(p)
        session.evaluate(numerical_program(kind), inputs,
                         {name: secret(p, value, owner=1) for name, value in claims.items()})
        return p.open(session.finish()[0])

    truth = (torch.nn.functional.gelu(x.double()/SCALE) if kind == 'gelu'
             else torch.nn.functional.silu(x.double()/SCALE))
    if kind == 'swiglu':
        truth *= value.double()/SCALE
    for actual in run_four(work):
        assert torch.max(torch.abs(actual.double()/SCALE - truth)) < 5e-6


def test_builtin_rejects_binary_witness_and_shape_mismatch():
    def work(p):
        x = secret(p, [SCALE, SCALE])
        binary = secret(p, [1, 1], owner=1, binary=True)
        scalar = secret(p, [SCALE], owner=1)
        for witness in (binary, scalar):
            session = WitnessSession(p)
            with pytest.raises((ValueError, TypeError)):
                session.evaluate(numerical_program('reciprocal'), {'x': x},
                                 {'reciprocal': witness})
    run_four(work)


def test_unknown_witness_name_rejected():
    def work(p):
        x = secret(p, [SCALE])
        session = WitnessSession(p)
        with pytest.raises((ValueError, KeyError)):
            session.evaluate(numerical_program('reciprocal'), {'x': x}, {'extra': x})
    run_four(work)


def test_custom_differently_shaped_witnesses_and_multiple_outputs():
    def recompute(ctx, inputs, witnesses):
        x = ctx.bounded(inputs['x'], 0, 2)
        ctx.residual(ctx.runtime.sub(witnesses['vector'], x), 0)
        scalar_expected = ctx.runtime.public(SCALE, like=witnesses['scalar'])
        ctx.residual(ctx.runtime.sub(witnesses['scalar'], scalar_expected), 0)
        return ctx.runtime.add(witnesses['vector'], witnesses['scalar'])

    program = WitnessProgram('custom', {'scalar': (0, 2), 'vector': (0, 2)}, recompute)

    def work(p):
        x = secret(p, [SCALE, 2 * SCALE])
        scalar = secret(p, SCALE, owner=1)
        vector = secret(p, [SCALE, 2 * SCALE], owner=1)
        session = WitnessSession(p)
        for _ in range(2):
            assert session.evaluate(program, {'x': x},
                                    {'scalar': scalar, 'vector': vector}) is None
        outputs = session.finish()
        assert len(outputs) == 2
        for output in outputs:
            assert torch.equal(p.open(output), torch.tensor([2 * SCALE, 3 * SCALE]))
        with pytest.raises((ValueError, RuntimeError)):
            session.finish()
        with pytest.raises((ValueError, RuntimeError)):
            session.evaluate(program, {'x': x}, {'scalar': scalar, 'vector': vector})
    run_four(work)


def test_rmsnorm_honest_and_negative_root_rejected():
    from rep4gpu.proofs.witness_activations import rmsnorm_program
    x = torch.tensor([[SCALE, -SCALE, SCALE, -SCALE], [0, 0, 0, 0]])
    inverse = torch.round(torch.rsqrt((x.double()/SCALE).square().mean(-1, keepdim=True)
                                     + 1/16) * SCALE).long()

    def work(p):
        shared_x = secret(p, x)
        root = secret(p, inverse, owner=1)
        session = WitnessSession(p)
        session.evaluate(rmsnorm_program(epsilon=1/16), {'x': shared_x}, {'inverse_sqrt': root})
        result = p.open(session.finish()[0])
        invalid = WitnessSession(p)
        invalid.evaluate(rmsnorm_program(epsilon=1/16), {'x': shared_x},
                         {'inverse_sqrt': p.mul_public(root, -1)})
        with pytest.raises(InvalidWitness):
            invalid.finish()
        return result

    expected = (x * inverse) >> 24
    for result in run_four(work):
        assert torch.equal(result, expected)


@pytest.mark.parametrize('bad', [-2**63, 0, 2**63 - 1])
def test_rmsnorm_extreme_witness_rejected(bad):
    from rep4gpu.proofs.witness_activations import rmsnorm_program

    def work(p):
        session = WitnessSession(p)
        session.evaluate(rmsnorm_program(), {'x': secret(p, [[SCALE, -SCALE]])},
                         {'inverse_sqrt': secret(p, [[bad]], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_softmax_honest_maximum_and_shift_attacks():
    from experiments.benchmarks.activations.numerical_primitives import IntegerOracle
    from rep4gpu.arithmetic.numerical import NumericalFixedPoint
    from rep4gpu.proofs.witness_activations import softmax_program
    x = torch.tensor([[0, SCALE, -SCALE, 2*SCALE], [0, 0, 0, 0]])
    shift = x.max(-1, keepdim=True).values
    oracle = IntegerOracle()
    exp = NumericalFixedPoint(oracle).exp(oracle.public(x-shift)).lo
    inverse = torch.round(SCALE*SCALE / exp.sum(-1, keepdim=True).double()).long()

    def work(p):
        shared_x = secret(p, x)
        reciprocal = secret(p, inverse, owner=1)
        correct_shift = secret(p, shift, owner=1)
        valid = WitnessSession(p)
        valid.evaluate(softmax_program(), {'x': shared_x},
                       {'shift': correct_shift, 'reciprocal': reciprocal})
        result = p.open(valid.finish()[0])
        for delta in [-1, 1]:
            invalid = WitnessSession(p)
            wrong = p.add(correct_shift, p.public(delta, like=correct_shift))
            invalid.evaluate(softmax_program(), {'x': shared_x},
                             {'shift': wrong, 'reciprocal': reciprocal})
            with pytest.raises(InvalidWitness):
                invalid.finish()
        return result

    truth = torch.softmax(x.double()/SCALE, -1)
    for result in run_four(work):
        assert torch.max(torch.abs(result.double()/SCALE-truth)) < 1e-6


@pytest.mark.parametrize('field', ['shift', 'reciprocal'])
@pytest.mark.parametrize('bad', [-2**63, 2**63-1])
def test_softmax_extreme_witness_rejected(field, bad):
    from rep4gpu.proofs.witness_activations import softmax_program

    def work(p):
        witnesses = {'shift': secret(p, [[0]], owner=1),
                     'reciprocal': secret(p, [[SCALE//2]], owner=1)}
        witnesses[field] = secret(p, [[bad]], owner=1)
        session = WitnessSession(p)
        session.evaluate(softmax_program(), {'x': secret(p, [[0, 0]])}, witnesses)
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_custom_program_requires_semantic_constraint():
    unchecked = WitnessProgram('unchecked', {'w': (0, 1)},
                               lambda ctx, inputs, witnesses: witnesses['w'])

    def work(p):
        session = WitnessSession(p)
        with pytest.raises(ValueError, match='residual'):
            session.evaluate(unchecked, {}, {'w': secret(p, [SCALE], owner=1)})
        with pytest.raises(RuntimeError):
            session.finish()
    run_four(work)


def test_rmsnorm_small_epsilon_honest_root():
    from rep4gpu.proofs.witness_activations import rmsnorm_program
    x = torch.tensor([[0, 0], [SCALE, -SCALE]])
    epsilon_encoded = round(1e-5*SCALE)
    variance = ((x*x) >> 24).sum(-1, keepdim=True) >> 1
    variance += epsilon_encoded
    root = torch.round(torch.rsqrt(variance.double()/SCALE)*SCALE).long()

    def work(p):
        session = WitnessSession(p)
        session.evaluate(rmsnorm_program(), {'x': secret(p, x)},
                         {'inverse_sqrt': secret(p, root, owner=1)})
        return p.open(session.finish()[0])

    for output in run_four(work):
        assert torch.equal(output, (x*root) >> 24)


@pytest.mark.parametrize('width', [1, 3])
def test_prover_normalization_helpers(width):
    from rep4gpu.proofs.witness_activations import rmsnorm_program, softmax_program
    from rep4gpu.proofs.witness_prover import rmsnorm_witnesses, softmax_witnesses
    x = torch.stack((torch.zeros(width, dtype=torch.int64),
                     torch.linspace(-SCALE, SCALE, width).long()))
    rms = rmsnorm_witnesses(x)
    softmax = softmax_witnesses(x)

    def work(p):
        shared_x = secret(p, x)
        session = WitnessSession(p)
        for program, witnesses in ((rmsnorm_program(), rms), (softmax_program(), softmax)):
            shares = {name: secret(p, value, owner=1) for name, value in witnesses.items()}
            session.evaluate(program, {'x': shared_x}, shares)
        return [p.open(output) for output in session.finish()]

    for rms_out, softmax_out in run_four(work):
        assert torch.equal(rms_out, (x*rms['inverse_sqrt']) >> 24)
        assert torch.max(torch.abs(softmax_out.double()/SCALE -
                                   torch.softmax(x.double()/SCALE, -1))) < 1e-6


@pytest.mark.parametrize('invalid', [torch.tensor([], dtype=torch.int64),
                                     torch.tensor([1.], dtype=torch.float64),
                                     torch.tensor([2**63-1]),
                                     torch.tensor([-2**63])])
def test_prover_helpers_validate_inputs(invalid):
    from rep4gpu.proofs.witness_prover import numerical_witnesses, rmsnorm_witnesses, softmax_witnesses
    for compute in (lambda: numerical_witnesses('silu', {'x': invalid}),
                    lambda: rmsnorm_witnesses(invalid),
                    lambda: softmax_witnesses(invalid)):
        with pytest.raises(ValueError):
            compute()


def test_opposite_wrong_residuals_cannot_cancel():
    def work(p):
        session = WitnessSession(p)
        session.evaluate(numerical_program('reciprocal'),
                         {'x': secret(p, [2*SCALE, 2*SCALE])},
                         {'reciprocal': secret(p, [SCALE//2-10, SCALE//2+10], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_reciprocal_rejects_one_ulp_error_at_unit_denominator():
    def work(p):
        session = WitnessSession(p)
        session.evaluate(numerical_program('reciprocal'),
                         {'x': secret(p, [SCALE])},
                         {'reciprocal': secret(p, [SCALE-1], owner=1)})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)


def test_softmax_reciprocal_tolerance_tracks_actual_denominator():
    from rep4gpu.proofs.witness_activations import softmax_program
    from rep4gpu.proofs.witness_prover import softmax_witnesses
    x = torch.full((1, 16), -4*SCALE, dtype=torch.int64)
    x[0, 0] = 4*SCALE
    witnesses = softmax_witnesses(x)
    witnesses['reciprocal'] += 2

    def work(p):
        session = WitnessSession(p)
        session.evaluate(softmax_program(), {'x': secret(p, x)},
                         {name: secret(p, value, owner=1) for name, value in witnesses.items()})
        with pytest.raises(InvalidWitness):
            session.finish()
    run_four(work)

"""Checked scalar and point protocols preserve Rep4 component ownership."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.field_group_runtime import FieldGroupRuntime, FieldShare
from rep4gpu.ristretto import Q, IDENTITY, base_mul, point_mul, point_add, hash_to_point
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


def run_four(work, mutate=None):
    seeds, network = make_seeds(), IdealNetwork(timeout=120)
    def run(role):
        try:
            rt = Party(role, network.endpoint(role), {i:s for i,s in seeds.items() if i != role})
            if mutate is not None:
                mutate(rt)
            return work(rt,FieldGroupRuntime(rt))
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run,range(4)))


def test_lift_unsigned_ring_boundaries():
    values = torch.tensor([0, 1, -(1 << 63), -1, 123456789],dtype=torch.int64)
    expected = tuple(int(value) % (1 << 64) for value in values)
    def work(rt, fg):
        secret = rt.input(0,values if rt.party == 0 else None,values.shape)
        return fg.open_field(fg.lift_ring64(secret))
    assert all(actual == expected for actual in run_four(work))


def test_field_products_and_dot():
    left = (0,1,Q-1,12345678)
    right = (19,Q-1,88,1 << 80)
    weights = (1,7,99,Q-2)
    def work(rt,fg):
        # Pair-known inputs exercise both checked input routes.
        a,b = fg.joint_input_batch([(0,1,3,2,left if rt.party < 2 else None),
                                    (2,3,1,0,right if rt.party >= 2 else None)],4)
        product = fg.open_field(fg.field_mul(a,b))
        dot = fg.open_field(fg.field_weighted_dot(a,b,weights))
        return product,dot
    expected = tuple(a*b%Q for a,b in zip(left,right))
    dot = sum(w*a*b for w,a,b in zip(weights,left,right))%Q
    assert all(actual == (expected,(dot,)) for actual in run_four(work))


def test_secret_group_dot_and_aggregate_opening():
    generator = base_mul(1)
    h = hash_to_point(bytes(range(64)))
    values, blindings = (0,1,(1 << 64)-1), (42,99,Q-2)
    commitments = tuple(point_add(point_mul(generator,a),point_mul(h,b))
                        for a,b in zip(values,blindings))
    def work(rt,fg):
        a,b = fg.joint_input_batch([(0,1,3,2,values if rt.party < 2 else None),
                                    (2,3,1,0,blindings if rt.party >= 2 else None)],3)
        difference = fg.group_linear(a,b,generator,h,commitments)
        openings = []
        original = fg.open_group
        def tracked(value):
            openings.append(value.count)
            return original(value)
        fg.open_group = tracked
        assert fg.check_group_zero(difference)
        bad = list(commitments)
        bad[-1] = point_add(bad[-1],generator)
        difference = fg.group_linear(a,b,generator,h,bad)
        assert not fg.check_group_zero(difference)
        assert openings == [1,1]
        alpha = fg.secret_field(3)
        points = fg.public_group(commitments)
        dot = fg.open_group(fg.group_weighted_dot(alpha,points))[0]
        clear_alpha = fg.open_field(alpha)
        expected = IDENTITY
        for scalar,point in zip(clear_alpha,commitments):
            expected = point_add(expected,point_mul(point,scalar))
        assert dot == expected
        # Wrapper recreation must not reuse PRSS draws.
        second = FieldGroupRuntime(rt).secret_field(3)
        assert second.parts != alpha.parts
    run_four(work)


def test_external_field_component_consistency():
    components = ((1,2),(3,4),(5,6),(7,8))
    def work(rt,fg):
        share = FieldShare(tuple(components[(rt.party+slot+1)%4] for slot in range(3)))
        fg.check_field_share(share)
        return fg.open_field(share)
    assert all(actual == (16,20) for actual in run_four(work))


def test_inconsistent_external_component_is_rejected():
    def work(rt,fg):
        parts = [(0,0)]*3
        if rt.party == 0:
            parts[0] = (1,0)
        fg.check_field_share(FieldShare(tuple(parts)))
    with pytest.raises((ValueError,ConnectionError)):
        run_four(work)


def test_corrupted_joint_integrity_tag_is_rejected():
    def mutate(rt):
        original = rt.transport.exchange
        def exchange(outgoing):
            if rt.party == 1 and 3 in outgoing:
                outgoing = dict(outgoing)
                outgoing[3] = bytes([outgoing[3][0]^1])+outgoing[3][1:]
            return original(outgoing)
        rt.transport.exchange = exchange
    def work(rt,fg):
        fg.joint_input_batch([(0,1,3,2,(123,) if rt.party < 2 else None)],1)
    with pytest.raises((ValueError,ConnectionError)):
        run_four(work,mutate)


def test_pinned_native_generators():
    import hashlib
    from rep4gpu.field_group_runtime import PEDERSEN_G, PEDERSEN_H, check_pedersen_generators
    assert PEDERSEN_G == base_mul(1)
    assert PEDERSEN_H == hash_to_point(hashlib.sha3_512(PEDERSEN_G).digest())
    check_pedersen_generators(PEDERSEN_G,PEDERSEN_H)
    with pytest.raises(ValueError,match='pinned'):
        check_pedersen_generators(PEDERSEN_G,base_mul(2))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_lift_projected_gpu_shares():
    from test_witness_check_optimization import _run_four
    values = torch.tensor([0, -(1 << 63), -1, 123],dtype=torch.int64,device='cuda')
    def work(rt):
        secret = rt.input(0,values if rt.party == 0 else None,values.shape)
        fg = FieldGroupRuntime(rt)
        return fg.open_field(fg.lift_ring64(secret))
    expected = (0,1 << 63,(1 << 64)-1,123)
    assert all(actual == expected for actual in _run_four(work,'cuda'))


def test_lift_rejects_unprojected_large_vectors():
    def work(rt,fg):
        with pytest.raises(ValueError,match='256 projected'):
            fg.lift_ring64(rt.public(torch.zeros(257,dtype=torch.int64)))
    run_four(work)


def test_lift_rejects_truncated_binary_backend():
    def work(rt,fg):
        # A projected arithmetic word can carry information in bit 63 even
        # when a specialized runtime exposes shorter binary payloads.
        rt.binary_bits = 32
        with pytest.raises(ValueError,match='64-bit binary wires'):
            fg.lift_ring64(rt.public(torch.tensor([-(1 << 63)],dtype=torch.int64)))
    run_four(work)

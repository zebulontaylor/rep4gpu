"""Run the independent device fault suite through prepared RBC graphs."""
import functools

import pytest

import test_spdz_device_faults as faults
from rep4gpu.spdz_device_graph import GraphDeviceDistributedParty


CASES = [
    ('test_ten_key_device_beaver_matches_exact_field_product',()),
    ('test_corrupt_ten_key_share_is_rejected_before_local_product',('value',)),
    ('test_corrupt_ten_key_share_is_rejected_before_local_product',('tenth-mac',)),
    *[('test_each_opening_phase_mutation_aborts_before_local_product',(round_id,consistent))
      for round_id in range(6) for consistent in (False,True)],
    ('test_replayed_previous_opening_frame_aborts_before_local_product',()),
    ('test_copied_spent_identifier_aborts_without_another_exchange',()),
    *[('test_consistent_malicious_payload_cannot_bypass_residual_guards',(stage,))
      for stage in (b'values',b'commit',b'reveal')],
    ('test_production_device_open_draws_a_new_salt_for_each_opening',()),
    ('test_canonical_cache_rechecks_mutated_tensor_version_before_sending',()),
    ('test_noncontiguous_ten_key_share_opening_matches_plaintext',()),
]


@pytest.mark.parametrize('name,args',CASES,ids=[name.removeprefix('test_')+str(args) for name,args in CASES])
def test_prepared_broadcast_graph_preserves_protocol_guards(monkeypatch,name,args):
    monkeypatch.setattr(faults,'_run',functools.partial(faults._run,party_type=GraphDeviceDistributedParty))
    getattr(faults,name)(*args)

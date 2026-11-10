"""Clone checked model-preparation state only into fixed transcript replay.

This helper belongs to benchmark fixtures. It cannot reset live handles or
transfer them to live transports. No new preprocessing is created here.
"""
from dataclasses import replace
import weakref

import torch

from rep4gpu.device_transport import DeviceReplay
from rep4gpu.spdz_prepared_right import PreparedRightMatmul,_states,_version
from rep4gpu.transport import ReplayTransport


def replay_prepared_right(party,source):
    """Bind a frozen checked A/F/W snapshot to an isolated replay party."""
    from benchmarks.spdz_streamed_replay import StreamedDeviceReplay
    if type(party.transport) not in (ReplayTransport,DeviceReplay,StreamedDeviceReplay):
        raise TypeError('prepared fixture cloning requires an exact fixed replay transport')
    if type(source) is not PreparedRightMatmul or source not in _states:
        raise ValueError('unknown checked model-preparation fixture')
    state = _states[source]
    owner = state.party()
    alpha = state.alpha()
    if owner is None or owner.poisoned or party.poisoned:
        raise ValueError('fixture source and replay sessions must remain valid')
    if alpha is None or owner.alpha_share is not alpha or _version(alpha) != state.alpha_version:
        raise ValueError('prepared fixture source MAC-key identity or version changed')
    if (party.party != state.role or party.session != state.session or party._session_hash != state.session_hash
            or not torch.equal(party.alpha_share,alpha)):
        raise ValueError('prepared fixture role, session, or MAC-key shares differ')
    handle = object.__new__(PreparedRightMatmul)
    _states[handle] = replace(state,party=weakref.ref(party),alpha=weakref.ref(party.alpha_share),
                             alpha_version=_version(party.alpha_share),used=False)
    return handle

"""Device-buffer exchanges for four Rep4 parties on distinct NCCL GPUs."""
import torch
import torch.distributed as dist

from .device import _buffers, _device, _validate_tensor


class NcclDeviceTransport:
    """Enqueue a public round on NCCL and order its results on the caller stream.

    The caller owns the NCCL process group. The authenticated control endpoint
    is used for seed setup before construction. Peers and the interconnect must
    protect the separate NCCL data channel. NCCL selects its physical route.
    """

    staging = 'nccl'
    authenticated = False

    def __init__(self, control, group, device):
        self.device = _device(device)
        if self.device.type != 'cuda' or self.device.index != torch.cuda.current_device():
            raise ValueError('set the current CUDA device to the assigned GPU before starting NCCL')
        if not dist.is_initialized() or dist.get_backend(group) != 'nccl':
            raise ValueError('a four-rank NCCL process group is required')
        if dist.get_world_size(group) != 4 or dist.get_rank(group) != control.party:
            raise ValueError('the NCCL group rank must equal the TLS party ID in a four-rank group')
        if not control.authenticated:
            raise ValueError('NCCL seed bootstrap requires authenticated TLS control')
        self.control = control
        self.group = group
        self.party = control.party
        self.rounds = self.bytes_sent = self.messages_sent = 0
        # All ranks participate before the first sparse P2P round. Some input
        # rounds involve only two parties and cannot initialize a communicator.
        dist.barrier(group=group, device_ids=[self.device.index])

    @property
    def control_authenticated(self):
        return self.control.authenticated

    def exchange(self, outgoing):
        raise RuntimeError('use device messages for the NCCL data channel')

    def exchange_device(self, outgoing, incoming_sizes):
        buffers = {p: torch.empty(n, dtype=torch.uint8, device=self.device)
                   for p, n in incoming_sizes.items()}
        return self.exchange_device_into(outgoing, incoming_sizes, buffers)

    def exchange_device_into(self, outgoing, incoming_sizes, buffers):
        _buffers(buffers, incoming_sizes, self.device)
        for value in outgoing.values():
            _validate_tensor(value, self.device)
        peers = set(outgoing) | set(buffers)
        if any(peer not in range(4) or peer == self.party for peer in peers):
            raise ValueError('invalid NCCL peer for this party')
        operations = []
        for peer in sorted(peers):
            global_peer = dist.get_global_rank(self.group, peer)
            send, recv = outgoing.get(peer), buffers.get(peer)
            if send is not None and send.numel():
                operations.append(dist.P2POp(dist.isend, send, global_peer, self.group))
            if recv is not None and recv.numel():
                operations.append(dist.P2POp(dist.irecv, recv, global_peer, self.group))
        if operations:
            with torch.cuda.device(self.device):
                for request in dist.batch_isend_irecv(operations):
                    # NCCL Work.wait orders the active CUDA stream before the
                    # runtime checks the received frames or reuses send storage.
                    request.wait()
        self.bytes_sent += sum(t.numel() for t in outgoing.values())
        self.messages_sent += len(outgoing)
        self.rounds += 1
        return buffers

    def close(self):
        self.control.close()

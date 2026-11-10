"""Fixed transcript transport with untimed ideal GPU message delivery."""
import torch

from rep4gpu.device_transport import DeviceReplay as _DeviceReplay


class StreamedDeviceReplay(_DeviceReplay):
    """One incoming fixture resident at a time, with untimed ideal delivery.

    This fixed benchmark-only class never wraps a live transport. Copies into
    GPU memory happen after ending one phase and before restarting the clock.
    """
    def __init__(self,transcript,device='cuda',check_outgoing=True):
        super().__init__(transcript,device=device,check_outgoing=check_outgoing,staging='host')
        self.staging='device'

    def exchange_device(self,outgoing,incoming_sizes):
        from rep4gpu.device_transport import _bytes,_tensor,_sizes,_validate_tensor
        frame=self._next('device')
        for value in outgoing.values():_validate_tensor(value,self.device)
        self._mark_phase()
        if self.check_outgoing and {p:_bytes(v) for p,v in outgoing.items()}!=frame.outgoing:
            raise ValueError(f'streamed replay outgoing mismatch at round {self.round}')
        incoming={p:_tensor(v,self.device) for p,v in frame.incoming.items()}
        _sizes(incoming,incoming_sizes)
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        self._advance([value.numel() for value in outgoing.values()])
        return incoming


class StreamedIncomingReplay:
    """Compatibility factory for the fixed auditable device-fixture transport."""
    @staticmethod
    def create(transcript,device,check_outgoing):
        return StreamedDeviceReplay(transcript,device=device,check_outgoing=check_outgoing)

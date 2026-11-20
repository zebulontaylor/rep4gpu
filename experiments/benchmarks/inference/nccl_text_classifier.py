"""Run one GPU per Rep4 party under torchrun, with local inputs and TLS seed setup."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from .text_classifier import child


def worker_config(manifest, directory, rank, gpu):
    """Build this rank's worker config from public model dimensions and local files."""
    config = dict(party=rank, device=f'cuda:{gpu}', device_transport='nccl',
                  device_input=True, generic_kernel=False, phase_graphs=True,
                  architecture=manifest['architecture'], fractional_bits=manifest['fractional_bits'],
                  weight_shapes=manifest['weight_shapes'], input_shape=manifest['input_shape'],
                  addresses=manifest['addresses'], bits=manifest.get('bits', 32),
                  public_signs=manifest.get('public_signs', []), timeout=manifest.get('timeout', 180),
                  warmup=manifest.get('warmup', 3), repeats=manifest.get('repeats', 10),
                  resident_inputs=manifest.get('resident_inputs', False),
                  identity={'certfile': str(directory/'party.pem'), 'keyfile': str(directory/'party.key')},
                  ca=str(directory/'ca.pem'))
    if len(config['addresses']) != 4 or config['repeats'] < 1 or config['warmup'] < 0 or config['timeout'] <= 0:
        raise ValueError('provide four TLS addresses, a positive timeout/repeats and nonnegative warmup')
    if rank in (0, 1):
        config['input'] = str(directory/'values.npz')
        shapes = {'x': config['input_shape']} if rank == 0 else config['weight_shapes']
        with np.load(config['input'], allow_pickle=False) as owned:
            for name, shape in shapes.items():
                value = owned[name]
                if value.dtype.kind not in 'iu' or value.shape != tuple(shape):
                    raise ValueError(f'party {rank} expects integer {name} with shape {tuple(shape)}')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='public JSON shapes and TLS addresses')
    parser.add_argument('--party-dir', required=True, help='local identity and owner data directory; {rank} expands to the party ID')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', 0)) != 4:
        parser.error('launch four processes with torchrun')
    rank, gpu = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['LOCAL_WORLD_SIZE']) > torch.cuda.device_count() or gpu >= torch.cuda.device_count():
        parser.error('NCCL requires a distinct visible GPU per local rank')
    directory = Path(args.party_dir.format(rank=rank))
    cfg = worker_config(json.loads(args.manifest.read_text()), directory, rank, gpu)
    torch.cuda.set_device(gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group('nccl', device_id=torch.device('cuda', gpu),
                            timeout=timedelta(seconds=cfg['timeout']))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'worker.json'
            path.write_text(json.dumps(cfg))
            return child(path, args.output_dir/f'party{rank}.json')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())

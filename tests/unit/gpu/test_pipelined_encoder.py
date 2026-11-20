"""The pipelined encoder must preserve every local ring contribution."""
import pytest
import torch

from rep4gpu.gpu.matmul.kernels import prepare_weight
from rep4gpu.gpu.matmul.tuned_kernels import local_products_tuned


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
                    reason='SM89 encoder dispatch required')
@pytest.mark.parametrize('prepare_sums, stages',
                         [(prepared, stages) for prepared in (False, True)
                          for stages in range(1, 6 if prepared else 5)])
def test_pipelined_encoder_full_ring_and_partial_tile(stages, prepare_sums):
    generator = torch.Generator().manual_seed(151)
    inputs = [torch.randint(-(1 << 31), 1 << 31, (512, 67), generator=generator).T.long()
              for _ in range(3)]
    weights = [torch.randint(-(1 << 31), 1 << 31, (512, 128), generator=generator)
               for _ in range(3)]
    x0, x1, x2 = inputs
    y0, y1, y2 = weights
    expected = ((x0 + x1) @ y0 + x0 @ y1, (x1 + x2) @ y1 + x1 @ y2, x0 @ y2 + x2 @ y0)
    packed = [prepare_weight(y.cuda(), 32, storage='words32') for y in weights]
    xs = tuple(x.cuda() for x in inputs)
    if prepare_sums:
        from rep4gpu.gpu.matmul.pipelined_encoder import prepare_pipelined_encoder, local_products_pipelined_prepared
        actual = local_products_pipelined_prepared(*xs, prepare_pipelined_encoder(*packed), stages=stages)
    else:
        actual = local_products_tuned(*xs, *packed, bits=32, encoder_pipeline_stages=stages)
    for part, reference in zip(actual, expected):
        assert torch.equal(part.cpu(), reference & 0xffffffff)

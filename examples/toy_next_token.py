"""Train and privately evaluate a synthetic two-token next-token predictor.

This example is a two-layer ReLU MLP, not a transformer. Every one of the 64
contexts is used for training and evaluation; accuracy is not held-out accuracy.
Run from the repository root: python -m examples.toy_next_token
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch
from torch import nn

from rep4gpu.activations import FixedPoint
from rep4gpu.exact_truncation import trunc, trunc_relu
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


def dataset():
    contexts = torch.cartesian_prod(torch.arange(8), torch.arange(8))
    inputs = torch.nn.functional.one_hot(contexts, num_classes=8).reshape(64,16).float()
    targets = contexts.sum(dim=1) % 8
    return contexts, inputs, targets


def train(steps=600):
    """Train on all contexts with a fixed seed and one CPU thread."""
    if steps < 1:
        raise ValueError('training steps must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    model = nn.Sequential(nn.Linear(16,64), nn.ReLU(), nn.Linear(64,8))
    _, inputs, targets = dataset()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def max_abs(value):
    return max(abs(int(value.min())),abs(int(value.max())))


def safe_matmul(left,right):
    if max_abs(left)*max_abs(right)*left.shape[-1] > 2**63-1:
        raise ValueError('int64 reference accumulation bound exceeded')
    return left @ right


def safe_add(left,right):
    if max_abs(left)+max_abs(right)>2**63-1:
        raise ValueError('int64 reference addition bound exceeded')
    return left+right


def quantized_fixture(model, fractional_bits=12, bits=32):
    if not 0 <= fractional_bits <= 30:
        raise ValueError('fractional_bits must be in [0,30]')
    _,inputs,targets=dataset()
    scale=1 << fractional_bits
    values=[inputs,model[0].weight.detach().T,model[0].bias.detach(),
            model[2].weight.detach().T,model[2].bias.detach()]
    encoded=[]
    for value in values:
        quantized=torch.round(value.double()*scale)
        if not bool((torch.isfinite(quantized) & (quantized>=-2.**63) & (quantized<2.**63)).all()):
            raise ValueError('encoded fixture exceeds signed int64')
        encoded.append(quantized.long())
    x,w1,b1,w2,b2=encoded
    if max_abs(b1)*scale>2**63-1:
        raise ValueError('scaled bias exceeds signed int64')
    raw1=safe_add(safe_matmul(x,w1),b1*scale)
    hidden=(raw1 >> fractional_bits).clamp_min(0)
    raw2=safe_matmul(hidden,w2)
    expected=safe_add(raw2 >> fractional_bits,b2)
    named=dict(input=x,weight1=w1,bias1=b1,weight2=w2,bias2=b2,
               first_accumulator=raw1,hidden=hidden,second_accumulator=raw2,output=expected)
    for name,value in named.items():
        if int(value.min()) < -(1 << (bits-1)) or int(value.max()) >= (1 << (bits-1)):
            raise ValueError(f'{name} exceeds signed ring{bits}')
    proof=dict(ring_bits=bits,fractional_bits=fractional_bits,
               first_accumulator_max_abs=max_abs(raw1),second_accumulator_max_abs=max_abs(raw2),
               signed_ring_limit=1 << (bits-1),all_fixture_values_fit_signed_ring=True,
               contract='Public all-context fixture validation before sharing; not a private range check.')
    return encoded,expected,targets,proof


class ExactPolicy:
    def trunc(self,x,shift):
        return trunc(self,x,shift)

    def trunc_relu(self,x,shift):
        return trunc_relu(self,x,shift)


def evaluate_private(model,device='cpu',bits=32,fractional_bits=12,fused=False,prg='shake'):
    encoded,expected,targets,proof=quantized_fixture(model,fractional_bits,bits)
    base=Party
    if fused:
        from rep4gpu.fused_runtime import FusedParty
        base=FusedParty
    inference_party=type('InferenceParty',(ExactPolicy,base),{})
    network,seeds=IdealNetwork(),make_seeds()
    parties=[inference_party(p,network.endpoint(p),{i:s for i,s in seeds.items() if i!=p},
                             device=device,bits=bits,prg=prg) for p in range(4)]
    with ThreadPoolExecutor(4) as pool:
        def worker(party):
            try:
                values=[party.input(owner,value if party.party==owner else None,value.shape)
                        for owner,value in zip([0,1,1,1,1],encoded)]
                x,w1,b1,w2,b2=values
                fp=FixedPoint(party,fractional_bits)
                hidden=fp.matmul_relu(x,w1,b1)
                output=fp.add(fp.matmul(hidden,w2),b2)
                return party.open(output)
            except BaseException as error:
                network.abort(error)
                raise
        outputs=list(pool.map(worker,parties))
    if any(not torch.equal(value.cpu(),expected) for value in outputs):
        raise AssertionError('private output differs from exact fixed-point reference')
    _,inputs,_=dataset()
    with torch.no_grad():
        floating=model(inputs)
    decoded=outputs[0].cpu().double()/(1 << fractional_bits)
    return dict(task='synthetic target=(previous_token_1+previous_token_2)%8',
                model='two-layer ReLU MLP: 16 inputs, 64 hidden units, 8 logits; not a transformer',
                evaluation='all 64 contexts, also used for training; no held-out accuracy claim',
                input_owner=0,model_owner=1,device=str(device),fused_runtime=fused,prg=prg,
                exact_integer_match=True,range_proof=proof,
                float_next_token_accuracy=float((floating.argmax(-1)==targets).float().mean()),
                fixed_next_token_accuracy=float((decoded.argmax(-1)==targets).float().mean()),
                max_logit_error=float((decoded-floating.double()).abs().max()),
                max_logprob_error=float((decoded.log_softmax(-1)-floating.double().log_softmax(-1)).abs().max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--bits',type=int,choices=[32,64],default=32)
    parser.add_argument('--fractional-bits',type=int,default=12)
    parser.add_argument('--fused',action='store_true')
    parser.add_argument('--prg',choices=['shake','aes'],default='shake')
    parser.add_argument('--output',type=Path,default=Path('results/toy_next_token'))
    args=parser.parse_args()
    model=train(args.steps)
    result=evaluate_private(model,args.device,args.bits,args.fractional_bits,args.fused,args.prg)
    result['torch_version']=torch.__version__
    result['checkpoint_format']='PyTorch state_dict for Linear(16,64), ReLU, Linear(64,8)'
    result['training']=dict(seed=20261109,steps=args.steps,optimizer='Adam',learning_rate=.03,
                            contexts=64,cpu_threads=1)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),args.output/'model.pt')
    (args.output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

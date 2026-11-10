"""Train a compact square MLP on all 64 two-token sum-modulo-eight contexts."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from examples.toy_next_token import dataset
from examples.spdz_trained_toy import Square, infer_authenticated
from rep4gpu.spdz import PRIME


def train(steps=2000):
    torch.set_num_threads(1)
    torch.manual_seed(20261110)
    model=nn.Sequential(nn.Linear(16,8),Square(),nn.Linear(8,8))
    _,inputs,targets=dataset()
    optimizer=torch.optim.Adam(model.parameters(),lr=.02,weight_decay=.001)
    for _ in range(steps):
        optimizer.zero_grad()
        loss=nn.functional.cross_entropy(model(inputs),targets)
        loss.backward();optimizer.step()
    return model.eval()


def quantized_fixture(model,fractional_bits=8,batch=64):
    if type(batch)is not int or batch<1:raise ValueError('batch must be positive')
    if type(fractional_bits)is not int or not 0<=fractional_bits<=8:raise ValueError('fractional bits must be in [0,8]')
    _,all_inputs,all_targets=dataset();all_inputs=all_inputs.long()
    rows=torch.arange(batch)%64;inputs=all_inputs[rows];targets=all_targets[rows]
    scale=1<<fractional_bits
    values=[model[0].weight.detach().T,model[0].bias.detach()[None,:],model[2].weight.detach().T,model[2].bias.detach()[None,:]]
    encoded=[]
    for value,factor in zip(values,[scale,scale,scale,scale**3]):
        value=(value.double()*factor).round()
        if not bool((torch.isfinite(value)&(value.abs()<PRIME//2)).all()):raise ValueError('parameter exceeds signed field range')
        encoded.append(value.long())
    w1,b1,w2,b2=encoded
    all_first=all_inputs@w1+b1;firstmax=int(all_first.abs().max())
    if firstmax**2>=PRIME//2:raise ValueError('square exceeds signed field range')
    if firstmax**2*int(w2.abs().max())*w2.shape[0]+int(b2.abs().max())>=2**63:raise ValueError('reference can overflow')
    absbounds=all_first.square()@w2.abs()+b2.abs();bound=int(absbounds.max())
    if bound>=PRIME//2:raise ValueError('output exceeds signed field range')
    expected=(inputs@w1+b1).square()@w2+b2
    proof=dict(field_prime=PRIME,signed_limit=PRIME//2,weight_fractional_bits=fractional_bits,
               input_scale=1,first_affine_scale=scale,square_scale=scale**2,output_scale=scale**3,
               first_affine_abs_bound=firstmax,square_abs_bound=firstmax**2,output_abs_bound=bound,
               actual_output_max_abs=int(expected.abs().max()),no_field_wrap=True,no_private_truncation=True,
               contract='Public bound across all 64 two-one-hot inputs; no private domain check.')
    return dict(encoded=[inputs,w1,b1,w2,b2],expected=expected,targets=targets,proof=proof)


def numerical_report(model,fixture):
    with torch.no_grad():floating=model(fixture['encoded'][0].float()).double()
    decoded=fixture['expected'].double()/fixture['proof']['output_scale']
    return dict(task='synthetic two-token sum modulo 8',architecture=[16,8,8],
                evaluation_rows=len(decoded),unique_contexts=min(len(decoded),64),
                evaluation='All 64 contexts are training contexts; no held-out accuracy claim.',
                float_accuracy=float((floating.argmax(-1)==fixture['targets']).float().mean()),
                integer_accuracy=float((fixture['expected'].argmax(-1)==fixture['targets']).float().mean()),
                max_logit_error=float((decoded-floating).abs().max()),
                max_logprob_error=float((decoded.log_softmax(-1)-floating.log_softmax(-1)).abs().max()),
                range_proof=fixture['proof'])


def preprocessing_manifest(batch=1):
    return dict(matrices=[dict(name='first',m=batch,k=16,n=8),dict(name='second',m=batch,k=8,n=8)],
                squares=[dict(name='hidden',shape=[batch,8])],
                inputs=[dict(name=name,owner=owner,shape=list(shape)) for name,owner,shape in
                        [('x',0,(batch,16)),('w1',1,(16,8)),('b1',1,(1,8)),('w2',1,(8,8)),('b2',1,(1,8))]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/spdz_compact_toy'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--batch',type=int,default=1)
    args=parser.parse_args();model=train()
    if args.checkpoint:model.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location='cpu'))
    report=numerical_report(model,quantized_fixture(model));report['private_batch']=args.batch
    report['training']=dict(seed=20261110,steps=2000,optimizer='Adam',learning_rate=.02,weight_decay=.001,cpu_threads=1)
    args.output.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),args.output/'model.pt')
    (args.output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'preprocessing.json').write_text(json.dumps(preprocessing_manifest(args.batch),indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

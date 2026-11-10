"""Measure fresh prepared-correlation batches and distinct single-row queries.

The trusted dealer is a functional fixture, not secure preprocessing. Every row
is consumed once in a live checked query. Only the first query is then replayed
on fixed benchmark transports to estimate its perfect-network online cost.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import secrets
import statistics
import tempfile
import time

import torch
from threadpoolctl import threadpool_limits

from benchmarks.inference import synchronize
from benchmarks.spdz_inference import fixture,baselines,compact_correlations
from benchmarks.spdz_native_replay import entropy_fixture_class,begin_recording,end_recording,run_local_replay,combine_replay_metrics
from benchmarks.spdz_prepared_model import MODEL_IDS,infer_prepared
from rep4gpu.spdz import TrustedDealer
from rep4gpu.spdz_distributed import AuthShare,split_share,split_triple,DistributedParty
from rep4gpu.spdz_polynomial import make_square_masks
from rep4gpu.spdz_prepared_right import prepare_right_matmul,partition_prepared_right
from rep4gpu.transport import IdealNetwork


def run_case(q=32,width=1024,device='cuda',repeats=25,warmups=20):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
    values,expected=fixture(1,width,width,4*width,'square_mlp')
    dealer=TrustedDealer(4,7,device)
    shares=[split_share(dealer.share(value.to(device))) for value in values]
    if device.type=='cuda':
        shares=[[AuthShare(item.values.to(torch.int32),item.macs.to(torch.int32)) for item in items] for items in shares]
    local_values=[[items[p] for items in shares] for p in range(4)]
    network=IdealNetwork(timeout=300)
    if device.type=='cuda':
        from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty as Base
        from rep4gpu.device_transport import DeviceIdealEndpoint
        endpoints=[DeviceIdealEndpoint(network.endpoint(p),device) for p in range(4)]
        backend='device-deferred'
    else:
        Base=DistributedParty;endpoints=[network.endpoint(p) for p in range(4)];backend='host'
    Runtime=entropy_fixture_class(Base)
    session=secrets.token_bytes(32)
    parties=[Runtime(p,endpoints[p],dealer.alpha_shares[:,p],session=session,
                     stack_macs=True,_allow_insecure_test_transport=True) for p in range(4)]
    def clock():synchronize(device);return time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        def all_roles(function):
            def guarded(p):
                try:return function(p)
                except BaseException as error:network.abort(error);raise
            futures=[pool.submit(guarded,p) for p in parties]
            return [future.result() for future in futures]
        prep_ms=0.;generation_ms=0.;children=[[],[]]
        for layer,(k,n,weight_index) in enumerate(((width,4*width,1),(4*width,width,3))):
            started=clock();triples=split_triple(dealer.triple(q,k,n))
            if device.type=='cuda':triples=compact_correlations(triples)
            generation_ms+=(clock()-started)*1000
            started=clock()
            parents=all_roles(lambda p:prepare_right_matmul(p,local_values[p.party][weight_index],triples[p.party],model_id=MODEL_IDS[layer]))
            for triple in triples:triple.a=triple.b=triple.c=None
            del triples
            partitioned=all_roles(lambda p:partition_prepared_right(p,parents[p.party],(1,)*q,model_id=MODEL_IDS[layer]))
            prep_ms+=(clock()-started)*1000
            children[layer]=partitioned
            del parents
        started=clock()
        masks=[make_square_masks(dealer,(1,4*width)) for _ in range(q)]
        if device.type=='cuda':masks=[compact_correlations(items) for items in masks]
        generation_ms+=(clock()-started)*1000
        states=[begin_recording(p,backend) for p in parties]
        outputs=[];query_ms=[]
        for index in range(q):
            started=clock()
            opened=all_roles(lambda p:p.open(infer_prepared(p,local_values[p.party],
                   (children[0][p.party][index],children[1][p.party][index]),masks[index][p.party])))
            query_ms.append((clock()-started)*1000)
            if any(not torch.equal(output.cpu(),expected) for output in opened):raise AssertionError('batched prepared query differs from exact reference')
            if index==0:
                for p,state in zip(parties,states):end_recording(p,state)
                outputs=opened
                entropy=[(list(p._benchmark_salt_log),{k:list(v) for k,v in p._benchmark_coin_logs.items()}) for p in parties]
    records=[]
    with tempfile.TemporaryDirectory() as temporary:
        for p,state in zip(parties,states):
            p._benchmark_salt_log,p._benchmark_coin_logs=entropy[p.party]
            material=dict(square_masks={'hidden':masks[0][p.party]})
            records.append(run_local_replay(p,material,local_values[p.party],state,outputs[p.party],
               repeats=repeats,replay_warmups=warmups,gpu_lock_path=Path(temporary)/'gpu.lock',
               stream_incoming=device.type=='cuda',prepared_model=(children[0][p.party][0],children[1][p.party][0])))
    timings,errors,measurements=baselines(values,'square_mlp',device,max(20,repeats),return_details=True)
    ideal=combine_replay_metrics(records,timings)
    return dict(query_capacity=q,width=width,hidden=4*width,mac_keys=7,
                preprocessing_provider='trusted functional fixture; secure cryptographic preprocessing unknown',security_target_met=False,
                distinct_live_queries_checked=q,exact_field_match=True,correlation_rows_reused=False,
                trusted_fixture_generation_ms=generation_ms,
                model_preparation_colocated_batch_ms=prep_ms,model_preparation_colocated_per_query_ms=prep_ms/q,
                model_preparation_timing='Actual four-role colocated wall time including checked E, F and partition; not isolated ideal estimate',
                live_query_samples_ms=query_ms,live_query_median_ms=statistics.median(query_ms),
                isolated_query=ideal,baselines=timings,baseline_errors=errors,baseline_measurements=measurements,
                observed_partial_aggregate_ms=prep_ms/q+ideal['ideal_network_phase_max_ms'],
                omitted_costs=['secure cryptographic preprocessing','new input sharing','static model sharing'],
                amortization='Divide preparation only by distinct fresh one-use row handles; immutable W shared. E checked once per newly generated Q-row correlation batch.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q',type=int,nargs='+',default=[32,128]);parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--device',default='cuda');parser.add_argument('--repeats',type=int,default=25)
    parser.add_argument('--warmups',type=int,default=20);parser.add_argument('--output',type=Path,default=Path('results/spdz_batched_preparation.json'))
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*Path('rep4gpu').glob('*.py'),*Path('benchmarks').glob('spdz*.py')]}
    report=dict(source_sha256=source,results=[])
    for q in args.q:
        row=run_case(q,args.width,args.device,args.repeats,args.warmups);report['results'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:row[k] for k in ('query_capacity','width','model_preparation_colocated_batch_ms','observed_partial_aggregate_ms')}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()

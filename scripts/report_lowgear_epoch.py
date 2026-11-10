"""Account for measured stages across distinct, actually served LowGear pools."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(paths):
    pools=[];sessions=set();epoch=None
    for source in map(Path,paths):
        raw=source.read_bytes();report=json.loads(raw);native=report['public_preprocessing_manifest']
        if not (report.get('passed') and report.get('exact_integer_match') and
                report.get('source_stable_during_run') and report.get('return_codes')==[0]*4 and
                native.get('ready') and native.get('return_codes')==[0]*4):
            raise ValueError('only successful, source-stable, actually served pools can be counted')
        if (native.get('format')!='lowgear-goldilocks-s40-native-field-v1' or
                native.get('prime')!=2**64-2**32+1 or native.get('finite_bit_probe')is not False):
            raise ValueError('expected the Goldilocks LowGear production experiment')
        session=native['session'];root=native.get('key_epoch_session',session)
        if session in sessions:raise ValueError('duplicate pool cannot increase served capacity')
        if epoch is None:
            if native.get('key_epoch_session'):raise ValueError('start with the original cold pool')
            epoch=root
        if root!=epoch:raise ValueError('reports belong to different key epochs')
        if pools and native.get('parent_session')!=pools[-1]['session']:
            raise ValueError('pool lineage must be complete and ordered')
        sessions.add(session)
        rows=report['batch']
        if type(rows)is not int or rows<1:raise ValueError('invalid served row count')
        if len(report['expected'])!=rows:raise ValueError('output row count differs')
        if sorted(p['party'] for p in report['parties'])!=list(range(4)):
            raise ValueError('exactly four distinct role results are required')
        expected_flat=[value for row in report['expected'] for value in row]
        for party in report['parties']:
            if party.get('passed')is not True or party['output']!=expected_flat:
                raise ValueError('a role did not verify every expected output')
        stages={
            'native_provisioning_ms':native['preprocessing_ms'],
            'role_loading_ms':max(p['preprocessing_load_ms'] for p in report['parties']),
            'input_sharing_ms':sum(report['input_protocol_by_name_ms'].values()),
            'checked_online_ms':report['checked_online_ms'],
        }
        clear=report['clear_baselines']['timings']
        values=[*stages.values(),clear['fp32_eager_ms'],clear['numpy_cpu_ms']]
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid timing')
        pools.append(dict(source=str(source),sha256=hashlib.sha256(raw).hexdigest(),session=session,
                          served_rows=rows,stages=stages,measured_staged_sum_ms=sum(stages.values()),
                          matched_clear_batch_ms={k:clear[k] for k in ('fp32_eager_ms','numpy_cpu_ms')}))
    if not pools:raise ValueError('at least one served pool is required')
    total=sum(p['measured_staged_sum_ms'] for p in pools);rows=sum(p['served_rows'] for p in pools)
    denominators={k:sum(p['matched_clear_batch_ms'][k] for p in pools) for k in pools[0]['matched_clear_batch_ms']}
    return dict(format='lowgear-actual-epoch-accounting-v1',pools=pools,actual_served_rows=rows,
                measured_staged_sum_ms=total,measured_staged_ms_per_row=total/rows,
                matched_clear_batches_total_ms=denominators,
                matched_batch_slowdowns={k:total/v if v else None for k,v in denominators.items()},
                cold_key_generation_included_once=True,cost_complete=False,performance_target_met=False,
                exclusions=['online certificate provisioning, process startup, and TLS setup are not separately timed'],
                scope='Independent matrix batches. All native provisioning, including the original key setup, is charged to actual served rows. Sum of observed stage maxima, not a single wall stopwatch or a perfect-network projection. These are not dependent-token latency ratios.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();report=summarize(args.reports)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('actual_served_rows','measured_staged_ms_per_row','matched_batch_slowdowns')}))

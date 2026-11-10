"""Shared benchmark helpers for fresh model-dependent matrix preprocessing."""
from rep4gpu.spdz_distributed import AuthShare
from rep4gpu.spdz_polynomial import square
from rep4gpu.spdz_prepared_right import prepare_right_matmul,matmul_prepared_right


MODEL_IDS=(b'benchmark/static-model/first/v1',b'benchmark/static-model/second/v1')


def prepare_model(party,values,correlations,workload='square_mlp'):
    """Consume fresh triples and check E before returning per-query A/F packs."""
    if workload!='matmul':
        batch=correlations[0].a.values.shape[0]
        for index in (2,4):
            if values[index].values.shape[0]!=batch:
                bias=_bias_rows(values[index],batch)
                values[index]=AuthShare(bias.values.contiguous(),bias.macs.contiguous())
            validate=getattr(party,'validate_share',None)
            if validate is not None:validate(values[index])
    first=prepare_right_matmul(party,values[1],correlations[0],model_id=MODEL_IDS[0])
    correlations[0].a=correlations[0].b=correlations[0].c=None
    if workload=='matmul':return (first,)
    second=prepare_right_matmul(party,values[3],correlations[2],model_id=MODEL_IDS[1])
    correlations[2].a=correlations[2].b=correlations[2].c=None
    party.local_backend='prepared-right kernel dispatch'
    return first,second


def _bias_rows(bias,batch):
    if bias.values.shape[0]==batch:return bias
    if bias.values.shape[0]!=1:raise ValueError('bias must have one row or match the batch')
    return AuthShare(bias.values.expand(batch,-1),bias.macs.expand(-1,batch,-1))


def infer_prepared(party,values,packs,square_mask=None,workload='square_mlp'):
    """Return a secret result; final checked output remains the caller's job."""
    first=matmul_prepared_right(party,values[0],packs[0],model_id=MODEL_IDS[0])
    if workload=='matmul':return first
    hidden=party.add(first,_bias_rows(values[2],first.values.shape[0]))
    hidden=square(party,hidden,square_mask)
    result=matmul_prepared_right(party,hidden,packs[1],model_id=MODEL_IDS[1])
    return party.add(result,_bias_rows(values[4],result.values.shape[0]))


def replay_model(local,packs):
    from benchmarks.spdz_prepared_fixture import replay_prepared_right
    return tuple(replay_prepared_right(local,pack) for pack in packs)

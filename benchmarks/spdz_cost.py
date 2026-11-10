"""Account for recurring preprocessing and reusable model-sharing costs."""


def pipeline_cost(*,query_ms,model_preparation_ms,input_sharing_ms,
                  cryptographic_preprocessing_ms=None,preprocessing_query_capacity=1,
                  static_model_sharing_ms=0.,model_queries=1,baselines=None):
    """Return per-query costs; unknown secure preprocessing leaves totals unknown.

    ``preprocessing_query_capacity`` counts distinct single-use correlation packs
    produced by the measured preprocessing batch. Replaying or reusing a pack
    never increases this capacity. ``model_queries`` amortizes only static model
    sharing, not fresh correlations or model-dependent E/F preparation.
    """
    for name,value in [('preprocessing_query_capacity',preprocessing_query_capacity),('model_queries',model_queries)]:
        if type(value) is not int or value<1:raise ValueError(f'{name} must be a positive integer')
    stages=dict(checked_query_ms=query_ms,fresh_model_preparation_ms=model_preparation_ms,
                new_input_sharing_ms=input_sharing_ms,
                amortized_static_model_sharing_ms=static_model_sharing_ms/model_queries,
                fresh_cryptographic_preprocessing_ms=None if cryptographic_preprocessing_ms is None else
                    cryptographic_preprocessing_ms/preprocessing_query_capacity)
    if any(value is not None and value<0 for value in stages.values()):
        raise ValueError('pipeline stage times must be nonnegative')
    known=sum(value for value in stages.values() if value is not None)
    total=None if cryptographic_preprocessing_ms is None else known
    return dict(stages=stages,preprocessing_query_capacity=preprocessing_query_capacity,
                model_queries=model_queries,known_stage_sum_ms=known,total_per_query_ms=total,
                total_complete=total is not None,
                total_slowdowns=None if total is None else {name:total/value for name,value in (baselines or {}).items()},
                formula='crypto_batch_ms / distinct_single_use_query_packs + fresh_model_preparation_ms + new_input_sharing_ms + checked_query_ms + static_model_sharing_ms / model_queries',
                reuse_contract='Each query consumes its own fresh correlations and E/F preparation; only static model sharing is reusable.')

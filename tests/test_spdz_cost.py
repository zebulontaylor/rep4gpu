import pytest
from benchmarks.spdz_cost import pipeline_cost


def test_fresh_correlations_are_not_amortized_by_model_reuse():
    result=pipeline_cost(query_ms=2,model_preparation_ms=3,input_sharing_ms=1,
                         cryptographic_preprocessing_ms=100,preprocessing_query_capacity=5,
                         static_model_sharing_ms=40,model_queries=10,baselines={'clear_ms':.5})
    assert result['total_per_query_ms']==30
    assert result['total_slowdowns']['clear_ms']==60
    assert result['stages']['fresh_cryptographic_preprocessing_ms']==20


def test_unknown_secure_preprocessing_does_not_become_zero():
    result=pipeline_cost(query_ms=2,model_preparation_ms=3,input_sharing_ms=1)
    assert result['total_per_query_ms'] is None and not result['total_complete']
    assert result['known_stage_sum_ms']==6
    with pytest.raises(ValueError):pipeline_cost(query_ms=2,model_preparation_ms=3,input_sharing_ms=1,preprocessing_query_capacity=0)

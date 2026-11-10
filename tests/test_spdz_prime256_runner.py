from scripts.run_spdz_prime256 import integer_reference


def test_independent_signed_integer_network_reference():
    x=[[1,0]];w1=[[2,-3],[4,1]];b1=[[1,2]];w2=[[2,1],[-1,3]];b2=[[5,-2]]
    assert integer_reference([x,w1,b1,w2,b2])==[[22,10]]

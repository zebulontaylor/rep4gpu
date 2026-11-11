// SPDX-License-Identifier: Apache-2.0
#include "fp127ext4.h"
#include <gmpxx.h>
#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using U=FP127::u128;
using E=FP127Ext4;
using Poly=std::array<mpz_class,4>;
mpz_class big(U value) {
    mpz_class result;
    mpz_import(result.get_mpz_t(),1,-1,sizeof(value),0,0,&value);
    return result;
}
const mpz_class P=big(FP127::PR);
mpz_class mod(mpz_class value) { value%=P;if(value<0)value+=P;return value; }
void require(bool condition,const char* message) {
    if (!condition) { std::cerr<<"FAIL "<<message<<'\n';std::exit(1); }
}

// Independent polynomial oracle: v^4=4*v^2-5, not tower multiplication.
Poly polynomial(const E& value) {
    return {mod(big(value[0].value())-2*big(value[1].value())),
            mod(big(value[2].value())-2*big(value[3].value())),
            big(value[1].value()),big(value[3].value())};
}
Poly multiply(const Poly& a,const Poly& b) {
    std::array<mpz_class,7> work{};
    for(unsigned i=0;i<4;++i)for(unsigned j=0;j<4;++j)work[i+j]+=a[i]*b[j];
    for(int degree=6;degree>=4;--degree) {
        work[degree-2]+=4*work[degree];
        work[degree-4]-=5*work[degree];
        work[degree]=0;
    }
    return {mod(work[0]),mod(work[1]),mod(work[2]),mod(work[3])};
}
Poly power(Poly base,mpz_class exponent) {
    Poly result{mpz_class(1),mpz_class(0),mpz_class(0),mpz_class(0)};
    while(exponent>0) {
        if(mpz_tstbit(exponent.get_mpz_t(),0))result=multiply(result,base);
        exponent>>=1;
        if(exponent>0)base=multiply(base,base);
    }
    return result;
}
void expect(const E& got,const Poly& polynomial_result,const char* operation) {
    const std::array<mpz_class,4> coordinates{
        mod(polynomial_result[0]+2*polynomial_result[2]),mod(polynomial_result[2]),
        mod(polynomial_result[1]+2*polynomial_result[3]),mod(polynomial_result[3])};
    for(unsigned h=0;h<4;++h) {
        if(big(got[h].value())!=coordinates[h]) {
            std::cerr<<"FAIL "<<operation<<" coordinate="<<h<<" got="<<big(got[h].value())
                     <<" want="<<coordinates[h]<<'\n';std::exit(1);
        }
    }
}

struct BytesPRG {
    std::vector<U> draws;
    std::size_t count=0;
    void random_data_unaligned(void* output,std::size_t size) {
        require(size==16 && count<draws.size(),"rejection sampler draw size/count");
        std::memcpy(output,&draws[count++],size);
    }
};
struct MemoryIO {
    std::vector<unsigned char> bytes;
    std::size_t offset=0;
    void send_data(const void* source,std::size_t size) {
        const auto* begin=static_cast<const unsigned char*>(source);
        bytes.insert(bytes.end(),begin,begin+size);
    }
    void recv_data(void* destination,std::size_t size) {
        require(offset+size<=bytes.size(),"memory wire underrun");
        std::memcpy(destination,bytes.data()+offset,size);offset+=size;
    }
};

int main() {
    require(P%4==3 && P%5==2,"field congruences");
    const mpz_class minus_one=P-1,five=5;
    require(mpz_legendre(minus_one.get_mpz_t(),P.get_mpz_t())==-1,"-1 is a nonsquare");
    require(mpz_legendre(five.get_mpz_t(),P.get_mpz_t())==-1,"norm5 is a nonsquare");
    require(mpz_probab_prime_p(P.get_mpz_t(),64)!=0,"base field primality");
    const E zero,one(FP127(1)),i=E::basis_unit(1),v=E::basis_unit(2),iv=E::basis_unit(3);
    require(E::basis_unit(0)==one,"basis embedding");
    require(i*i==-one,"i squared");
    require(v*v==E(FP127(2))+i,"v squared");
    require(i*v==iv,"iv basis order");
    require(v.pow(4)-E(FP127(4))*v.pow(2)+E(FP127(5))==zero,"quartic relation");
    require(i.pow(FP127::PR)==-i,"quadratic Frobenius");
    require(v.pow(FP127::PR).pow(FP127::PR)==-v,"degree four Frobenius");

    std::mt19937_64 rng(0xf127e4);
    auto raw=[&](){return ((U)rng()<<64)|rng();};
    auto random_element=[&](){return E(FP127(FP127::mod(raw())),FP127(FP127::mod(raw())),
                                      FP127(FP127::mod(raw())),FP127(FP127::mod(raw())));};
    std::vector<E> edges{zero,one,-one,i,-i,v,-v,iv,-iv,
        E(FP127(FP127::PR-1),FP127(FP127::PR-1),FP127(FP127::PR-1),FP127(FP127::PR-1)),
        E(FP127((U)1<<126),FP127((U)1<<64),FP127((U)1<<63),FP127(255))};
    std::size_t pairs=0,inverses=0;
    mpz_class inverse_exponent;mpz_pow_ui(inverse_exponent.get_mpz_t(),P.get_mpz_t(),4);inverse_exponent-=2;
    auto check=[&](const E& a,const E& b,bool oracle_inverse) {
        expect(a*b,multiply(polynomial(a),polynomial(b)),"polynomial multiplication");
        for(unsigned h=0;h<4;++h) {
            require(big((a+b)[h].value())==mod(big(a[h].value())+big(b[h].value())),"addition");
            require(big((a-b)[h].value())==mod(big(a[h].value())-big(b[h].value())),"subtraction");
            require(big((-a)[h].value())==mod(-big(a[h].value())),"negation");
        }
        const E c=random_element();
        require(a*(b+c)==a*b+a*c,"distributivity");
        require((a*b)*c==a*(b*c),"associativity");
        require(a*b==b*a,"commutativity");
        const FP127 scalar(FP127::mod(raw()));
        require(a*scalar==a*E(scalar) && scalar*a==a*scalar,"base scalar embedding");
        E compound=a;compound+=b;require(compound==a+b,"compound addition");
        compound=a;compound-=b;require(compound==a-b,"compound subtraction");
        compound=a;compound*=b;require(compound==a*b,"compound multiplication");
        compound=a;compound*=scalar;require(compound==a*scalar,"compound scalar multiplication");
        const auto bytes=a.encode();require(bytes.size()==64,"codec size");
        require(E::decode(bytes)==a,"codec round trip");
        for(unsigned h=0;h<4;++h)for(unsigned j=0;j<16;++j)
            require(bytes[16*h+j]==(unsigned char)(a[h].value()>>(8*j)),"coordinate wire order");
        MemoryIO wire;a.send(&wire);b.send(&wire);require(wire.bytes.size()==128,"two exact wire elements");
        E decoded_a,decoded_b;decoded_a.recv(&wire);decoded_b.recv(&wire);
        require(decoded_a==a && decoded_b==b && wire.offset==128,"send recv order");
        if(!a.is_zero() && inverses<2000) {
            const E reciprocal=a.inv();require(a*reciprocal==one,"inverse identity");
            if(oracle_inverse)expect(reciprocal,power(polynomial(a),inverse_exponent),"GMP polynomial inverse");
            require((b/a)*a==b,"division");
            ++inverses;
        }
        ++pairs;
    };
    for(const auto& a:edges)for(const auto& b:edges)check(a,b,true);
    for(unsigned count=0;count<10000;++count)check(random_element(),random_element(),count<64);
    for(unsigned count=0;count<32;++count) {
        E a=random_element(),f=a;
        for(unsigned power=0;power<4;++power)f=f.pow(FP127::PR);
        require(f==a,"Frobenius order four");
    }
    for(unsigned coordinate=0;coordinate<4;++coordinate)for(U bad:{FP127::PR,FP127::PR+1,~(U)0}) {
        auto bytes=zero.encode();
        for(unsigned j=0;j<16;++j)bytes[16*coordinate+j]=(unsigned char)(bad>>(8*j));
        bool threw=false;try{(void)E::decode(bytes);}catch(const std::invalid_argument&){threw=true;}
        require(threw,"noncanonical coordinate rejected");
        MemoryIO wire;wire.bytes.assign(bytes.begin(),bytes.end());E target=one;
        threw=false;try{target.recv(&wire);}catch(const std::invalid_argument&){threw=true;}
        require(threw && target==one,"invalid receive preserves previous value");
    }
    std::array<unsigned char,128> bytes{};
    for(std::size_t length:{0,1,16,63,65,128}) {
        bool threw=false;try{(void)E::decode(bytes.data(),length);}catch(const std::invalid_argument&){threw=true;}
        require(threw,"wrong codec length rejected");
    }
    bool threw=false;try{(void)E::decode(nullptr,64);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"null codec pointer rejected");
    threw=false;try{(void)E::basis_unit(4);}catch(const std::out_of_range&){threw=true;}
    require(threw,"invalid basis index rejected");
    threw=false;try{(void)zero.inv();}catch(const std::domain_error&){threw=true;}
    require(threw,"zero inverse rejected");
    FP127 malformed;malformed.assign_no_mod(FP127::PR);
    threw=false;try{E bad(malformed);(void)bad;}catch(const std::invalid_argument&){threw=true;}
    require(threw,"noncanonical constructor rejected");
    const U wide=((U)1<<126)|77;
    BytesPRG prg{{FP127::PR,~(U)0,((U)1<<127)|123,0,1,wide}};
    const E sampled=E::uniform(prg);
    require(sampled==E(FP127(123),FP127(0),FP127(1),FP127(wide)) && prg.count==6,"four independent rejection coordinates");
    BytesPRG again{{FP127::PR,5,6,7,8}};E resampled=one;resampled.rand(again);
    require(resampled==E(FP127(5),FP127(6),FP127(7),FP127(8)) && again.count==5,"fresh mutating sampler");
    require(sizeof(FP1274)==sizeof(E) && E::size()==64,"alias and encoded size");
    std::cout<<"PASS "<<pairs<<" GMP polynomial arithmetic pairs, "<<inverses
             <<" inverse identities, irreducibility/Frobenius, strict64-byte codec, and exact rejection checks\n";
}

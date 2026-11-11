#include "fp127.h"
#include "fp127x2.h"
#include <gmpxx.h>
#include <random>
#include <iostream>
#include <vector>
#include <climits>

using U = FP127::u128;
mpz_class big(U x) { mpz_class r; mpz_import(r.get_mpz_t(), 1, -1, sizeof(x), 0, 0, &x); return r; }
void expect(U got, const mpz_class& want, const char* op) {
    if (big(got) != want) { std::cerr << op << " got=" << big(got) << " want=" << want << '\n'; std::exit(1); }
}
struct BytesPRG {
    std::vector<U> draws; std::size_t count=0;
    void random_data_unaligned(void* out, std::size_t n) {
        if (n != 16 || count >= draws.size()) std::abort();
        std::memcpy(out, &draws[count++], n);
    }
};
struct MemoryIO {
    std::vector<unsigned char> bytes; std::size_t position=0;
    void send_data(const void* data, std::size_t n) {
        auto p=static_cast<const unsigned char*>(data); bytes.insert(bytes.end(),p,p+n);
    }
    void recv_data(void* data, std::size_t n) {
        if(position+n>bytes.size())std::abort();
        std::memcpy(data,bytes.data()+position,n);position+=n;
    }
};
int main() {
    const mpz_class p=big(FP127::PR);
    std::mt19937_64 rng(0xb14687af);
    auto random_raw=[&]() { return ((U)rng()<<64)|rng(); };
    std::vector<U> edges={0,1,2,3,((U)1<<63)-1,(U)1<<63,((U)1<<64)-1,(U)1<<64,FP127::HALF_PR,FP127::PR-2,FP127::PR-1};
    std::size_t checked=0;
    auto check=[&](U a,U b, bool inverse) {
        FP127 x(a),y(b); mpz_class A=big(a),B=big(b);
        expect((x+y).val,(A+B)%p,"add");
        expect((x-y).val,(A-B+p)%p,"sub");
        expect((x*y).val,(A*B)%p,"mul");
        expect((x*x).val,(A*A)%p,"square");
        expect(x.negate().val, a==0?mpz_class(0):p-A,"negate");
        FP127 acc=x; acc.add_raw(y); acc.reduce(); expect(acc.val,(A+B)%p,"raw");
        for(int shift: {0,1,63,64,126,127,128,255}) {
            mpz_class shifted=A; mpz_mul_2exp(shifted.get_mpz_t(),shifted.get_mpz_t(),shift);
            expect((x<<shift).val,shifted%p,"shift");
        }
        auto bytes=x.encode(); expect(FP127::decode(bytes.data()).val,A,"encoding");
        MemoryIO wire; x.send(&wire); FP127 decoded; decoded.recv(&wire);expect(decoded.val,A,"wire");
        for(unsigned i=0;i<16;++i) if(bytes[i]!=(unsigned char)(a>>(8*i))) std::abort();
        FP127x2 pair(x,y), other(y,x);
        MemoryIO pairwire; pair.send(&pairwire);FP127x2 decodedpair;decodedpair.recv(&pairwire);
        if(pairwire.bytes.size()!=32 || decodedpair!=pair)std::abort();
        auto prod=pair*other; expect(prod.val[0],(A*B)%p,"pairmul0");expect(prod.val[1],(A*B)%p,"pairmul1");
        auto sum=pair+other;expect(sum.val[0],(A+B)%p,"pairadd");
        pair.add_raw(other);pair.reduce();expect(pair.val[1],(A+B)%p,"pairraw");
        if(inverse) { mpz_class inv=0;if(a)mpz_invert(inv.get_mpz_t(),A.get_mpz_t(),p.get_mpz_t());expect(x.inv().val,inv,"inverse"); }
        ++checked;
    };
    for(U a:edges)for(U b:edges)check(a,b,true);
    for(unsigned i=0;i<20000;++i)check(FP127::mod(random_raw()),FP127::mod(random_raw()),i<1000);
    for(unsigned i=0;i<20000;++i){U raw=random_raw();expect(FP127::mod(raw),big(raw)%p,"rawmod");expect(FP127::from_lo_hi((uint64_t)raw,(uint64_t)(raw>>64)).val,big(raw)%p,"splitdecode");}
    for(U bad: {FP127::PR,FP127::PR+1,~(U)0}) {
        unsigned char bytes[16];for(unsigned i=0;i<16;++i)bytes[i]=(unsigned char)(bad>>(8*i));
        bool threw=false;try{FP127::decode(bytes);}catch(const std::invalid_argument&){threw=true;}if(!threw)std::abort();
    }
    BytesPRG prg{{FP127::PR,~(U)0,((U)1<<127)|123}};
    FP127 sampled; sampled.rand(prg);if(sampled.val!=123 || prg.count!=3)std::abort();
    BytesPRG wide{{((U)1<<126)|123}}; sampled.rand(wide);
    if(sampled.val!=(((U)1<<126)|123))std::abort();
    expect(FP127(INT_MIN).val,p+INT_MIN,"signed constructor");
    bool threw=false;try{FP127(1)<<-1;}catch(const std::invalid_argument&){threw=true;}if(!threw)std::abort();
    std::cout<<"PASS "<<checked<<" arithmetic pairs, 20000 full-width reductions, inverse/serialization/rejection checks\n";
}

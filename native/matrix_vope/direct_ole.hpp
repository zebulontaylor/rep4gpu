#pragma once

#include "wolverine_base.hpp"

namespace matrix_vope {

struct DirectOleOutput {
    std::vector<FP127> share;
    std::array<double,3> phase_seconds{}; // Full base, input corrections, completion.
};

// ALICE supplies scalar a; BOB supplies v. Their output shares satisfy
// L + R = a*v. Both inputs remain role-local; their own adversarially chosen
// inputs need later authentication when a larger protocol requires consistency.
// One call consumes one complete fresh r1 Wolverine base batch.
template<typename IO>
inline DirectOleOutput direct_ole(
        IO& io,int party,FP127 scalar,const std::vector<FP127>& vector,
        size_t length,PeerContext<FP127,IO>& peer,EndpointEntropy& entropy,
        const std::string& domain,const std::string& fault="") {
    RetireEntropyOnFailure retire{entropy};
    entropy.ensure_live();
    if((party!=emp::ALICE && party!=emp::BOB) || !length || length>8192 ||
       (party==emp::BOB && vector.size()!=length) || scalar.val>=FP127::PR)
        throw std::invalid_argument("invalid role-local OLE input");
    for(const auto& value:vector)if(value.val>=FP127::PR)
        throw std::invalid_argument("noncanonical OLE vector");
    using Clock=std::chrono::steady_clock;
    auto start=Clock::now();
    auto mark=[&] {auto now=Clock::now();double seconds=std::chrono::duration<double>(now-start).count();start=now;return seconds;};
    auto base=wolverine_base(io,party,length,peer,entropy,domain+"/random-base",nullptr,fault,1);
    DirectOleOutput out;
    out.phase_seconds[0]=mark();
    std::vector<FP127> correction(length);
    FP127 scalar_correction;
    out.share.resize(length);
    if(party==emp::BOB) {
        for(size_t i=0;i<length;++i)correction[i]=vector[i]-base.value[i];
        send_fields(io,correction.data(),correction.size());io.flush();
        receive_fields(io,&scalar_correction,1);
        for(size_t i=0;i<length;++i)out.share[i]=base.share[i]+scalar_correction*vector[i];
    } else {
        // Complete canonical reception fixes d before c reveals a-Gamma.
        receive_fields(io,correction.data(),correction.size());
        scalar_correction=scalar-base.gamma;
        send_fields(io,&scalar_correction,1);io.flush();
        for(size_t i=0;i<length;++i)out.share[i]=(base.share[i]-base.gamma*correction[i]).negate();
    }
    out.phase_seconds[1]=mark();
    const unsigned char accepted=1;
    auto ack=commit_exchange(io,party,entropy,&accepted,1,domain+"/accepted-ole");
    if(ack.size()!=1 || ack[0]!=1)throw std::runtime_error("peer did not complete OLE");
    out.phase_seconds[2]=mark();
    return out;
}

} // namespace matrix_vope

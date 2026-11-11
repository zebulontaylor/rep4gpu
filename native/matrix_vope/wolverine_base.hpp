#pragma once

#include "seeded_vole.hpp"
#include "fp127ext4.h"

namespace matrix_vope {

struct WolverineBaseOutput {
    FP127 gamma, gamma_second; // Only ALICE receives the compressed scalar(s).
    std::vector<FP127> value; // Only BOB receives the random or programmed input.
    std::vector<FP127> share, share_second; // ALICE K or BOB M, for each output coordinate.
    std::array<double,4> phase_seconds{}; // Raw COPE, check, compression, derandomization.
};

// Source: https://www.mccormick.northwestern.edu/computer-science/documents/nu-cs-2024-11-chenkai-weng.pdf
// Figure 2.3 uses Fp pads/corrections for all r*log2(p)
// choices, then a gadget sum produces Fp^r outputs. These four coordinate
// batches are that raw COPEe; none performs a separate consistency check.
template<typename IO>
inline std::vector<FP127> wolverine_raw_coordinate(
        IO& io, int party, FP127 delta, std::vector<FP127>& values,
        size_t count, PeerContext<FP127,IO>& peer, const std::string& fault) {
    constexpr size_t bits=FP127::slot_stride_bits;
    CopeFp<IO,FP127> cope(party,&io);
    struct BorrowedPrgs {
        CopeFp<IO,FP127>& cope;
        ~BorrowedPrgs() { cope.G0=nullptr;cope.G1=nullptr; }
    } borrowed{cope};
    std::vector<emp::block> first(bits),second(bits);
    if(party==emp::ALICE) {
        cope.delta=delta;
        cope.delta_bool=new bool[bits];
        cope.delta64_to_bool(cope.delta_bool,delta.val,bits);
        peer.cope_ot.recv_rot(first.data(),cope.delta_bool,bits);
    } else peer.cope_ot.send_rot(first.data(),second.data(),bits);
    io.flush();
    cope.G0=peer.cope_prg0.data();
    if(party==emp::BOB)cope.G1=peer.cope_prg1.data();
    for(size_t bit=0;bit<bits;++bit) {
        cope.G0[bit].reseed(&first[bit]);
        if(party==emp::BOB)cope.G1[bit].reseed(&second[bit]);
    }
    peer.cope_indices+=bits;
    std::vector<FP127> correlation(count);
    canonical_cope(io,party,cope,correlation.data(),
                   party==emp::BOB?values.data():nullptr,count,fault);
    return correlation;
}

// Figure 2.4 with p=2^127-1: r=1,rho=128,ell=4 or r=2,rho=120,ell=2.
// Paired mode views [1,i,J,iJ] as two Fp2 blocks [1,J], compresses over
// Fp2, and exposes both output coordinates against the SAME input vector.
// Fresh raw keys are consumed once: every row and mask is fixed before the
// full Fp4 check; the compression coin is drawn only after that check succeeds.
// The persistent object retains OT engines, never a raw key across calls.
template<typename IO>
inline WolverineBaseOutput wolverine_base(
        IO& io, int party, size_t count, PeerContext<FP127,IO>& peer,
        EndpointEntropy& entropy, const std::string& domain,
        const emp::block* programmed_seed=nullptr, const std::string& fault="", size_t output_degree=1) {
    RetireEntropyOnFailure retire{entropy};
    entropy.ensure_live();
    if(!count || count>8192 || (output_degree!=1 && output_degree!=2))throw std::invalid_argument("unsupported Wolverine base length");
    using Clock=std::chrono::steady_clock;
    auto start=Clock::now();
    auto mark=[&] { auto now=Clock::now();double elapsed=std::chrono::duration<double>(now-start).count();start=now;return elapsed; };
    const std::array<uint64_t,7> configuration{2,127,4,count,programmed_seed?1UL:0UL,output_degree,output_degree==1?128UL:120UL};
    auto other=commit_exchange(io,party,entropy,configuration.data(),sizeof(configuration),domain+"/config");
    if(other.size()!=sizeof(configuration) || std::memcmp(other.data(),configuration.data(),sizeof(configuration)))
        throw std::runtime_error("Wolverine base configuration mismatch");
    auto private_rng=entropy.fork(domain+"/raw-private-inputs");
    FP127Ext4::Coordinates raw_delta{};
    std::vector<FP127> values(count+4);
    if(party==emp::ALICE) {
        for(auto& coordinate:raw_delta)coordinate=uniform<FP127>(private_rng);
        // Fault witnesses condition one *raw* key bit. This branch is excluded
        // from honest generation and never conditions the compressed gamma.
        if(fault=="cope-high-zero")raw_delta[0].val&=~(FP127::u128(1)<<64);
        if(fault=="cope-high-one") {
            do { raw_delta[0]=uniform<FP127>(private_rng);raw_delta[0].val|=(FP127::u128(1)<<64); }
            while(raw_delta[0].val>=FP127::PR);
        }
    } else for(auto& value:values)value=uniform<FP127>(private_rng);
    std::array<std::vector<FP127>,4> lanes;
    for(size_t lane=0;lane<4;++lane) {
        auto lane_values=values;
        if(party==emp::BOB && fault=="split-input" && lane==1)
            lane_values[0]=lane_values[0]+FP127(1);
        const std::string raw_fault=lane==0 && (fault=="cope-high-zero" || fault=="cope-high-one")?"cope-high":"";
        lanes[lane]=wolverine_raw_coordinate(io,party,raw_delta[lane],lane_values,count+4,peer,raw_fault);
    }
    std::vector<FP127Ext4> correlation(count+4);
    for(size_t row=0;row<count+4;++row)
        correlation[row]=FP127Ext4(lanes[0][row],lanes[1][row],lanes[2][row],lanes[3][row]);
    WolverineBaseOutput out;
    out.phase_seconds[0]=mark();

    std::vector<FP127Ext4> chi(count);
    if(party==emp::ALICE) {
        auto challenge_rng=entropy.fork(domain+"/full-extension-challenge");
        for(auto& coefficient:chi) { coefficient=FP127Ext4::uniform(challenge_rng);coefficient.send(&io); }
        io.flush();
    } else for(auto& coefficient:chi)coefficient.recv(&io);
    FP127Ext4 folded;
    for(size_t row=0;row<count;++row)folded+=chi[row]*correlation[row];
    for(size_t h=0;h<4;++h)folded+=correlation[count+h]*FP127Ext4::basis_unit(h);
    if(party==emp::BOB) {
        FP127Ext4 x;
        for(size_t row=0;row<count;++row)x+=chi[row]*values[row];
        for(size_t h=0;h<4;++h)x+=FP127Ext4::basis_unit(h)*values[count+h];
        if(fault=="response-x")x+=FP127Ext4::basis_unit(0);
        if(fault=="response-z")folded+=FP127Ext4::basis_unit(0);
        x.send(&io);folded.send(&io);io.flush();
        unsigned char accepted=0;io.recv_data(&accepted,1);
        if(accepted!=1)throw std::runtime_error("Wolverine full-extension check failed");
    } else {
        FP127Ext4 x,z;x.recv(&io);z.recv(&io);
        const bool valid=z==folded+FP127Ext4(raw_delta)*x;
        unsigned char accepted=valid?1:0;io.send_data(&accepted,1);io.flush();
        if(!valid)throw std::runtime_error("Wolverine full-extension check failed");
    }
    out.phase_seconds[1]=mark();

    std::array<FP127,4> alpha{};
    if(party==emp::ALICE) {
        auto compression_rng=entropy.fork(domain+"/post-check-compression");
        for(auto& coefficient:alpha)coefficient=uniform<FP127>(compression_rng);
        send_fields(io,alpha.data(),alpha.size());io.flush();
        if(output_degree==1) {
            for(size_t lane=0;lane<4;++lane)out.gamma=out.gamma+alpha[lane]*raw_delta[lane];
        } else {
            out.gamma=alpha[0]*raw_delta[0]-alpha[1]*raw_delta[1]+alpha[2]*raw_delta[2]-alpha[3]*raw_delta[3];
            out.gamma_second=alpha[0]*raw_delta[1]+alpha[1]*raw_delta[0]+alpha[2]*raw_delta[3]+alpha[3]*raw_delta[2];
        }
    } else receive_fields(io,alpha.data(),alpha.size());
    out.share.resize(count);
    if(output_degree==2)out.share_second.resize(count);
    for(size_t row=0;row<count;++row) {
        if(output_degree==1) {
            for(size_t lane=0;lane<4;++lane)out.share[row]=out.share[row]+alpha[lane]*lanes[lane][row];
        } else {
            out.share[row]=alpha[0]*lanes[0][row]-alpha[1]*lanes[1][row]+alpha[2]*lanes[2][row]-alpha[3]*lanes[3][row];
            out.share_second[row]=alpha[0]*lanes[1][row]+alpha[1]*lanes[0][row]+alpha[2]*lanes[3][row]+alpha[3]*lanes[2][row];
        }
    }
    if(party==emp::BOB)out.value.assign(values.begin(),values.begin()+count);
    out.phase_seconds[2]=mark();

    // Ordinary random-to-chosen vector derandomization. This fixes one call's
    // vector; persistent same-vector binding across calls is a separate layer.
    if(programmed_seed) {
        std::vector<FP127> correction(count);
        if(party==emp::BOB) {
            emp::PRG programmed(programmed_seed);
            for(size_t row=0;row<count;++row) {
                auto chosen=uniform<FP127>(programmed);
                correction[row]=chosen-out.value[row];out.value[row]=chosen;
            }
            send_fields(io,correction.data(),correction.size());io.flush();
        } else {
            receive_fields(io,correction.data(),correction.size());
            for(size_t row=0;row<count;++row) {
                out.share[row]=out.share[row]-out.gamma*correction[row];
                if(output_degree==2)out.share_second[row]=out.share_second[row]-out.gamma_second*correction[row];
            }
        }
    }
    const unsigned char accepted=1;
    auto ack=commit_exchange(io,party,entropy,&accepted,1,domain+"/accepted-output");
    if(ack.size()!=1 || ack[0]!=1)throw std::runtime_error("peer did not accept Wolverine output");
    out.phase_seconds[3]=mark();
    return out;
}

} // namespace matrix_vope

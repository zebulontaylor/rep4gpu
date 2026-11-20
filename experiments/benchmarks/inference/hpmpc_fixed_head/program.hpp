#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../functions/comparisons.hpp"
#include <fstream>
#define FUNCTION YourFirstProgram
#define RESULTTYPE DATATYPE

template <typename Share>
void YourFirstProgram(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    constexpr int width=512, hidden=4, lanes=DATTYPE/BITLENGTH, batch=8192, groups=batch/lanes;
    auto t0=std::chrono::steady_clock::now();
    std::vector<A> xs(groups*width), ws(width*hidden+hidden+1), hs(groups*hidden), scores(groups), signs(groups);
    alignas(sizeof(DATATYPE)) UINT_TYPE plain[lanes]{};
    DATATYPE packed;
    std::ifstream input;
    if (PARTY==0) input.open("evidence/owner0.bin",std::ios::binary);
    if (PARTY==1) input.open("evidence/owner1.bin",std::ios::binary);
    for(int j=0;j<int(ws.size());j++) {
        if (PARTY==1) { UINT_TYPE value;input.read(reinterpret_cast<char*>(&value),sizeof(value));for(int l=0;l<lanes;l++)plain[l]=value; }
        orthogonalize_arithmetic(plain,&packed,1);
        ws[j].template prepare_receive_from<P_1>(packed);
    }
    for(int g=0;g<groups;g++)for(int j=0;j<width;j++) {
        if(PARTY==0)input.read(reinterpret_cast<char*>(plain),sizeof(plain));
        orthogonalize_arithmetic(plain,&packed,1);
        xs[g*width+j].template prepare_receive_from<P_0>(packed);
    }
    Share::communicate();
    for(auto &w:ws)w.template complete_receive_from<P_1>();
    for(auto &x:xs)x.template complete_receive_from<P_0>();
    auto t1=std::chrono::steady_clock::now();
    for(int g=0;g<groups;g++)for(int h=0;h<hidden;h++) {
        auto &v=hs[g*hidden+h]; v=A(0);
        for(int j=0;j<width;j++)v+=xs[g*width+j].prepare_dot(ws[j*hidden+h]);
        v.mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for(int g=0;g<groups;g++)for(int h=0;h<hidden;h++) {
        auto &v=hs[g*hidden+h]; v.complete_mult_without_trunc();
        v+=ws[width*hidden+h];
    }
    for(int g=0;g<groups;g++) {
        scores[g]=A(0);
        for(int h=0;h<hidden;h++) {
            auto &v=hs[g*hidden+h];
            if(h<hidden/2)scores[g]+=v.prepare_dot(v);
            else scores[g]=scores[g]-v.prepare_dot(v);
        }
        scores[g].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for(auto &v:scores){v.complete_mult_without_trunc();v+=ws.back();}
    auto t2=std::chrono::steady_clock::now();
    pack_additive<0,BITLENGTH>(scores.data(),signs.data(),groups,LTZ<0,BITLENGTH,Share,DATATYPE>);
    auto t3=std::chrono::steady_clock::now();
    for(auto &s:signs)s.prepare_reveal_to_all();
    Share::communicate();
    std::ofstream output;
    if(current_phase==PHASE_LIVE)output.open("evidence/predictions"+std::to_string(PARTY)+".bin",std::ios::binary);
    for(auto &s:signs){ packed=s.complete_reveal_to_all();unorthogonalize_arithmetic(&packed,plain,1);if(current_phase==PHASE_LIVE)output.write(reinterpret_cast<char*>(plain),sizeof(plain)); }
    if(current_phase==PHASE_LIVE)std::cout<<"PHASE_MS input "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<" matmul "<<std::chrono::duration<double,std::milli>(t2-t1).count()<<" sign "<<std::chrono::duration<double,std::milli>(t3-t2).count()<<" reveal "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t3).count()<<"\n";
}

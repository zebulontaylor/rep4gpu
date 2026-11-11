#include "seeded_vole.hpp"
#include "fp127x2.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace matrix_vope;
using Field = FP127;
using Packed = FP127x2;

static std::string decimal(Field::u128 value) {
    std::string out;
    do { out.push_back(char('0' + value % 10)); value /= 10; } while (value);
    std::reverse(out.begin(), out.end());
    return out;
}
static void fields(std::ostream& out, const std::vector<Field>& values) {
    out << '[';
    for (size_t i=0;i<values.size();++i) { if(i)out<<',';out<<decimal(values[i].val); }
    out << ']';
}

int main(int argc, char** argv) {
    // This guard precedes construction of EMP entropy sources and OT contexts.
    if (std::getenv("EMP_TEST_MODE")) {
        std::cerr << "EMP_TEST_MODE is forbidden for private correlation generation\n";
        return 2;
    }
    if (argc < 5 || argc > 7) {
        std::cerr << "usage: direct_seeded_cope <1|2> <port> <private-output.json> <sessions> [length] [fault]\n";
        return 2;
    }
    try {
        const int party=std::stoi(argv[1]), port=std::stoi(argv[2]);
        const size_t sessions=std::stoul(argv[4]);
        const size_t length=argc>5?std::stoul(argv[5]):128;
        const std::string fault=argc>6?argv[6]:"";
        if ((party!=emp::ALICE && party!=emp::BOB) || port<1 || port>65535 ||
            sessions<1 || sessions>32 || length<1 || length>4096)
            throw std::invalid_argument("unsupported direct COPE configuration");
        if (!fault.empty() && fault!="cope-high" && fault!="base-response0" &&
            fault!="base-response1" && fault!="base-mac" && fault!="cope-high-zero")
            throw std::invalid_argument("unknown direct COPE fault");
        const auto setup_start=std::chrono::steady_clock::now();
        emp::NetIO io(party==emp::ALICE?nullptr:"127.0.0.1",port);
        const auto setup_sent=io.send_counter, setup_received=io.recv_counter;
        EndpointEntropy entropy;
        RetireEntropyOnFailure retire{entropy};
        auto coin=joint_coin(io,party,entropy,"direct-seeded-cope/connection-v1");
        auto h=digest(&coin,sizeof(coin));
        std::string domain="direct-seeded-cope/v1/";
        constexpr char hex[]="0123456789abcdef";
        for(auto byte:h){domain.push_back(hex[byte>>4]);domain.push_back(hex[byte&15]);}
        PeerContext<Field> peer(io,party);
        auto private_entropy=entropy.fork(domain+"/private-inputs/"+std::to_string(party));
        emp::block vector_seed=emp::zero_block;
        if(party==emp::BOB)private_entropy.random_block(&vector_seed,1);
        std::ofstream out(argv[3]);
        if(!out)throw std::runtime_error("cannot open private output fixture");
        const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
        out << "{\"protocol\":\"direct-checked-seeded-vector-COPE\",\"field\":\"fp127\",\"prime\":\""
            <<decimal(Field::PR)<<"\",\"security_reviewed\":false,\"regular_noise_assumption\":false,"
            <<"\"network_debug_disclosure\":false,\"length\":"<<length
            <<",\"base_check_degree\":"<<length<<",\"context_setup_seconds\":"<<setup_seconds
            <<",\"context_sent_bytes\":"<<io.send_counter-setup_sent
            <<",\"context_received_bytes\":"<<io.recv_counter-setup_received<<",\"sessions\":[";
        for(size_t session=0;session<sessions;++session) {
            const auto started=std::chrono::steady_clock::now();
            const auto sent=io.send_counter, received=io.recv_counter;
            const std::string sid=domain+"/call/"+std::to_string(session);
            const std::array<uint64_t,5> config{1,127,length,sessions,session};
            auto other=commit_exchange(io,party,entropy,config.data(),sizeof(config),sid+"/configuration");
            if(std::memcmp(other.data(),config.data(),sizeof(config)))
                throw std::runtime_error("direct COPE configuration mismatch");
            std::vector<Field> keys(length), values(length+1);
            std::vector<Packed> macs(length);
            Field delta;
            if(party==emp::ALICE) {
                do {
                    delta=uniform<Field>(private_entropy);
                    if(fault=="cope-high")delta.val|=(Field::u128(1)<<64);
                    if(fault=="cope-high-zero")delta.val&=~(Field::u128(1)<<64);
                } while(delta==Field{} || delta.val>=Field::PR);
            } else {
                emp::PRG programmed(&vector_seed);
                for(size_t i=0;i<length;++i)values[i]=uniform<Field>(programmed);
                auto mask=entropy.fork(sid+"/fresh-check-mask");
                values[length]=uniform<Field>(mask);
            }
            std::array<double,4> wall{},cpu{};
            checked_base_extended<Field,Packed>(io,party,delta,length,keys,macs,values,
                                               peer,entropy,sid,wall,cpu,
                                               fault=="cope-high-zero"?"cope-high":fault);
            const unsigned char accepted=1;
            auto ack=commit_exchange(io,party,entropy,&accepted,1,sid+"/accepted");
            if(ack.size()!=1 || ack[0]!=1)throw std::runtime_error("peer did not accept direct COPE");
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            if(session)out<<',';
            out<<"{\"index\":"<<session<<",\"seconds\":"<<seconds
               <<",\"sent_bytes\":"<<io.send_counter-sent<<",\"received_bytes\":"<<io.recv_counter-received
               <<",\"entropy_substreams_consumed\":"<<entropy.substreams();
            if(party==emp::ALICE)out<<",\"delta\":"<<decimal(delta.val);
            else {values.resize(length);out<<",\"value\":";fields(out,values);for(size_t i=0;i<length;++i)keys[i]=macs[i].getLow();}
            out<<",\"share\":";fields(out,keys);
            out<<",\"base_wall_seconds\":[";
            for(size_t i=0;i<4;++i){if(i)out<<',';out<<wall[i];}
            out<<"]}";
            std::cout<<"role="<<party<<" call="<<session<<" checked seconds="<<seconds<<'\n';
        }
        out<<"]}\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"protocol aborted: "<<error.what()<<'\n';
        return 1;
    }
}

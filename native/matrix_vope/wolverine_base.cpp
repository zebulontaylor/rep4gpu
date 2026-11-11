#include "wolverine_base.hpp"
#include <fstream>
#include <iostream>

using namespace matrix_vope;
static std::string decimal(FP127::u128 value) {
    std::string text;
    do { text.push_back(char('0'+value%10));value/=10; }while(value);
    std::reverse(text.begin(),text.end());return text;
}
static void fields(std::ostream& output,const std::vector<FP127>& values) {
    output<<'[';
    for(size_t i=0;i<values.size();++i){if(i)output<<',';output<<decimal(values[i].val);}
    output<<']';
}
int main(int argc,char** argv) {
    try {
        require_secure_randomness();
        if(argc==5 && std::string(argv[1])=="reflection") {
            const int party=std::stoi(argv[2]),port=std::stoi(argv[3]);
            const size_t size=std::stoul(argv[4]);
            if((party!=1 && party!=2) || (size!=16 && size!=32))return 2;
            emp::NetIO io(party==emp::ALICE?nullptr:"127.0.0.1",port);
            if(party==emp::BOB) {
                Digest commitment;io.recv_data(commitment.data(),commitment.size());
                io.send_data(commitment.data(),commitment.size());io.flush();
                std::vector<unsigned char> opening(32+size);io.recv_data(opening.data(),opening.size());
                io.send_data(opening.data(),opening.size());io.flush();return 0;
            }
            EndpointEntropy entropy;
            std::vector<unsigned char> value(size,7);
            try { commit_exchange(io,party,entropy,value.data(),value.size(),"wolverine-reflection-test"); }
            catch(const std::exception&) {
                try { (void)entropy.fork("after-abort"); }
                catch(const std::runtime_error&) { std::cerr<<"reflection rejected; entropy retired\n";return 1; }
                return 2;
            }
            return 0;
        }
        if(argc<6 || argc>8)throw std::invalid_argument("usage: wolverine_base <1|2> <port> <private-output.json> <sessions> <length> [random|programmed|paired-random|paired-programmed] [fault]");
        const int party=std::stoi(argv[1]),port=std::stoi(argv[2]);
        const size_t sessions=std::stoul(argv[4]),length=std::stoul(argv[5]);
        const std::string mode=argc>6?argv[6]:"random",fault=argc>7?argv[7]:"";
        const bool programmed=mode=="programmed" || mode=="paired-programmed";
        const size_t output_degree=mode.starts_with("paired-")?2:1;
        if((party!=1 && party!=2) || port<1 || port>65535 || !sessions || sessions>32 || !length || length>8192 || (mode!="random" && mode!="programmed" && mode!="paired-random" && mode!="paired-programmed"))throw std::invalid_argument("unsupported configuration");
        if(!fault.empty() && fault!="split-input" && fault!="response-x" && fault!="response-z" && fault!="cope-high-zero" && fault!="cope-high-one")throw std::invalid_argument("unknown fault");
        const auto setup_start=std::chrono::steady_clock::now();
        emp::NetIO io(party==emp::ALICE?nullptr:"127.0.0.1",port);
        const auto before_sent=io.send_counter,before_received=io.recv_counter;
        EndpointEntropy entropy;
        RetireEntropyOnFailure retire{entropy};
        auto coin=joint_coin(io,party,entropy,"wolverine-base/connection-v1");
        auto hash=digest(&coin,sizeof(coin));
        std::string domain="wolverine-base/v1/";
        constexpr char hex[]="0123456789abcdef";
        for(auto byte:hash){domain+=hex[byte>>4];domain+=hex[byte&15];}
        PeerContext<FP127> peer(io,party);
        auto input_rng=entropy.fork(domain+"/private-program-seed");
        emp::block program_seed=emp::zero_block;
        if(party==emp::BOB)input_rng.random_block(&program_seed,1);
        std::ofstream output(argv[3]);
        if(!output)throw std::runtime_error("cannot open private fixture");
        const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
        output<<"{\"protocol\":\"wolverine-figure2.4-base-svole\",\"field\":\"fp127\",\"prime_bits\":127,\"raw_extension_degree\":4,\"output_degree\":"<<output_degree<<",\"rho\":"<<(output_degree==1?128:120)<<",\"security_reviewed\":false,\"full_preprocessing\":false,\"mode\":\""<<mode<<"\",\"diagnostic_fault\":\""<<fault<<"\",\"context_setup_seconds\":"<<setup_seconds<<",\"context_sent_bytes\":"<<io.send_counter-before_sent<<",\"context_received_bytes\":"<<io.recv_counter-before_received<<",\"sessions\":[";
        for(size_t session=0;session<sessions;++session) {
            const auto start=std::chrono::steady_clock::now();
            const auto sent=io.send_counter,received=io.recv_counter,indices=peer.cope_indices;
            auto result=wolverine_base(io,party,length,peer,entropy,domain+"/call/"+std::to_string(session),programmed?&program_seed:nullptr,fault,output_degree);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(session)output<<',';
            output<<"{\"index\":"<<session<<",\"seconds\":"<<seconds<<",\"sent_bytes\":"<<io.send_counter-sent<<",\"received_bytes\":"<<io.recv_counter-received<<",\"fresh_rot_choices\":"<<peer.cope_indices-indices<<",\"entropy_substreams\":"<<entropy.substreams();
            if(party==emp::ALICE) {
                output<<",\"gamma\":"<<decimal(result.gamma.val);
                if(output_degree==2)output<<",\"gamma_second\":"<<decimal(result.gamma_second.val);
            }
            else {output<<",\"value\":";fields(output,result.value);}
            output<<",\"share\":";fields(output,result.share);
            if(output_degree==2){output<<",\"share_second\":";fields(output,result.share_second);}
            output<<",\"phase_seconds\":[";
            for(size_t phase=0;phase<4;++phase){if(phase)output<<',';output<<result.phase_seconds[phase];}
            output<<"]}";output.flush();
            std::cout<<"role="<<party<<" session="<<session<<" accepted seconds="<<seconds<<'\n';
        }
        output<<"]}\n";return 0;
    }catch(const std::exception& error){std::cerr<<"protocol aborted: "<<error.what()<<'\n';return 1;}
}

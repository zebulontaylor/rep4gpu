#include "direct_ole.hpp"
#ifdef SIV_OLE_TLS
#include "tls_io.h"
#endif
#include <fstream>
#include <iostream>

using namespace matrix_vope;
static uint64_t read_u64(std::istream& input) {
    std::array<unsigned char,8> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
    if(!input)throw std::runtime_error("truncated OLE request");
    uint64_t value=0;for(size_t i=0;i<8;++i)value|=uint64_t(bytes[i])<<(8*i);return value;
}
static FP127 read_field(std::istream& input) {
    std::array<unsigned char,16> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
    if(!input)throw std::runtime_error("truncated private field input");
    return FP127::decode(bytes.data());
}
static std::string decimal(FP127::u128 value) {
    std::string out;do{out.push_back(char('0'+value%10));value/=10;}while(value);
    std::reverse(out.begin(),out.end());return out;
}
static void fields(std::ostream& out,const std::vector<FP127>& values) {
    out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<decimal(values[i].val);}out<<']';
}
int main(int argc,char** argv) {
    try {
        require_secure_randomness();
#ifdef SIV_OLE_TLS
        if(argc<12 || argc>13)throw std::invalid_argument("usage: direct_ole_tls <1|2> <port> <request> <output> <self-role> <peer-role> <ca> <cert> <key> <peer-name> <session-hex> [base-fault]");
#else
        if(argc<5 || argc>6)throw std::invalid_argument("usage: direct_ole <1|2> <port> <private-request.bin> <private-output.json> [base-fault]");
#endif
        const int party=std::stoi(argv[1]),port=std::stoi(argv[2]);
#ifdef SIV_OLE_TLS
        const std::string fault=argc>12?argv[12]:"";
        TLSOptions tls;
        const auto self_role=std::stoul(argv[5]),peer_role=std::stoul(argv[6]);
        if(self_role>3 || peer_role>3 || self_role==peer_role)throw std::invalid_argument("invalid TLS role pair");
        tls.self_role=self_role;tls.peer_role=peer_role;
        tls.ca_file=argv[7];tls.cert_file=argv[8];tls.key_file=argv[9];tls.peer_name=argv[10];
        const std::string session_hex=argv[11];
        if(session_hex.size()!=64)throw std::invalid_argument("TLS session must contain 32 bytes of hex");
        for(size_t byte=0;byte<32;++byte) {
            auto nibble=[](char c)->unsigned char {if(c>='0' && c<='9')return c-'0';if(c>='a' && c<='f')return c-'a'+10;if(c>='A' && c<='F')return c-'A'+10;throw std::invalid_argument("invalid session hex");};
            tls.session[byte]=(nibble(session_hex[2*byte])<<4)|nibble(session_hex[2*byte+1]);
        }
#else
        const std::string fault=argc>5?argv[5]:"";
#endif
        if((party!=1 && party!=2) || port<1 || port>65535)throw std::invalid_argument("invalid endpoint");
        if(!fault.empty() && fault!="split-input" && fault!="response-x" && fault!="response-z" && fault!="cope-high-one" && fault!="cope-high-zero")throw std::invalid_argument("unknown base fault");
        std::ifstream input(argv[3],std::ios::binary);
        std::array<char,8> magic{};input.read(magic.data(),magic.size());
        if(!input || std::string(magic.data(),magic.size())!="SIVOLE01")throw std::invalid_argument("invalid OLE request magic");
        const size_t length=read_u64(input),calls=read_u64(input);
        if(!length || length>8192 || !calls || calls>4096 || length*calls>(1UL<<24))throw std::invalid_argument("unsupported OLE request dimensions");
        std::vector<FP127> private_input(party==emp::ALICE?calls:calls*length);
        for(auto& value:private_input)value=read_field(input);
        if(input.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing private request bytes");
#ifdef SIV_OLE_TLS
        // Bind the native protocol and its exact layout as a TLS subchannel,
        // in addition to the caller's epoch/batch/ordered-pair session.
        const uint64_t scalar_role=party==emp::ALICE?tls.self_role:tls.peer_role;
        const uint64_t vector_role=party==emp::BOB?tls.self_role:tls.peer_role;
        const std::array<uint64_t,6> channel_configuration{1,127,length,calls,scalar_role,vector_role};
        const std::string channel_domain="direct-ole/channel-v1";
        std::vector<unsigned char> channel_binding(channel_domain.begin(),channel_domain.end());
        channel_binding.insert(channel_binding.end(),tls.session.begin(),tls.session.end());
        for(auto word:channel_configuration)
            for(size_t byte=0;byte<8;++byte)channel_binding.push_back(word>>(8*byte));
        tls.channel_id=digest(channel_binding.data(),channel_binding.size());
#endif
        const auto setup_start=std::chrono::steady_clock::now();
#ifdef SIV_OLE_TLS
        matrix_vope::TLSIO io(party==emp::ALICE?nullptr:"127.0.0.1",port,tls);
        constexpr bool authenticated=true;
#else
        emp::NetIO io(party==emp::ALICE?nullptr:"127.0.0.1",port);
        constexpr bool authenticated=false;
#endif
        const auto initial_sent=io.send_counter,initial_received=io.recv_counter;
        EndpointEntropy entropy;RetireEntropyOnFailure retire{entropy};
        auto coin=joint_coin(io,party,entropy,"direct-ole/connection-v1");
        auto hash=digest(&coin,sizeof(coin));std::string domain="direct-ole/v1/";
        constexpr char hex[]="0123456789abcdef";for(auto byte:hash){domain+=hex[byte>>4];domain+=hex[byte&15];}
        const std::array<uint64_t,4> configuration{1,127,length,calls};
        auto other=commit_exchange(io,party,entropy,configuration.data(),sizeof(configuration),domain+"/configuration");
        if(other.size()!=sizeof(configuration) || std::memcmp(other.data(),configuration.data(),sizeof(configuration)))throw std::runtime_error("OLE request configuration mismatch");
        PeerContext<FP127,decltype(io)> peer(io,party);
        const double context_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
        std::ofstream out(argv[4]);if(!out)throw std::runtime_error("cannot open private OLE fixture");
        out<<"{\"protocol\":\"wolverine-r1-chosen-input-ole\",\"role\":"<<party<<",\"prime_bits\":127,\"rho\":128,\"security_reviewed\":false,\"full_preprocessing\":false,\"transport_authenticated\":"<<(authenticated?"true":"false")<<",\"length\":"<<length<<",\"calls\":"<<calls<<",\"context_setup_seconds\":"<<context_seconds<<",\"context_sent_bytes\":"<<io.send_counter-initial_sent<<",\"context_received_bytes\":"<<io.recv_counter-initial_received;
#ifdef SIV_OLE_TLS
        out<<",\"self_role\":"<<tls.self_role<<",\"peer_role\":"<<tls.peer_role<<",\"tls_session\":\""<<session_hex<<"\"";
#endif
        out<<",\"sessions\":[";
        for(size_t call=0;call<calls;++call) {
            FP127 scalar;
            std::vector<FP127> vector;
            if(party==emp::ALICE)scalar=private_input[call];
            else vector.assign(private_input.begin()+call*length,private_input.begin()+(call+1)*length);
            const auto start=std::chrono::steady_clock::now();
            const auto sent=io.send_counter,received=io.recv_counter,indices=peer.cope_indices;
            auto result=[&] {
                try { return direct_ole(io,party,scalar,vector,length,peer,entropy,domain+"/call/"+std::to_string(call),fault); }
                catch(const std::exception& error) {
                    // EMP's active-stream destructor exits during a failed OT
                    // session. Preserve the original public error first.
                    std::cerr<<"OLE call aborted: "<<error.what()<<'\n';
                    throw;
                }
            }();
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(call)out<<',';
            out<<"{\"index\":"<<call<<",\"seconds\":"<<seconds<<",\"sent_bytes\":"<<io.send_counter-sent<<",\"received_bytes\":"<<io.recv_counter-received<<",\"fresh_rot_choices\":"<<peer.cope_indices-indices<<",\"entropy_substreams\":"<<entropy.substreams()<<",\"share\":";fields(out,result.share);
            out<<",\"phase_seconds\":[";for(size_t phase=0;phase<3;++phase){if(phase)out<<',';out<<result.phase_seconds[phase];}out<<"]}";out.flush();
            std::cout<<"role="<<party<<" call="<<call<<" completed seconds="<<seconds<<'\n';
        }
        out<<"]}\n";return 0;
    }catch(const std::exception& error){std::cerr<<"protocol aborted: "<<error.what()<<'\n';return 1;}
}

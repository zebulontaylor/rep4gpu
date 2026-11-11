#include "direct_ole.hpp"
#include "tls_io.h"
#include "ole_service_limits.h"
#include <fstream>
#include <iostream>
#include <set>

using namespace matrix_vope;
namespace {
uint64_t read_u64(std::istream& input) {
    std::array<unsigned char,8> bytes{};input.read(reinterpret_cast<char*>(bytes.data()),8);
    if(!input)throw std::runtime_error("truncated service frame");
    uint64_t value=0;for(size_t j=0;j<8;++j)value|=uint64_t(bytes[j])<<(8*j);return value;
}
void append_u64(std::vector<unsigned char>& bytes,uint64_t value) {
    for(size_t j=0;j<8;++j)bytes.push_back(value>>(8*j));
}
std::string hex(const unsigned char* bytes,size_t count) {
    constexpr char alphabet[]="0123456789abcdef";std::string text;
    for(size_t j=0;j<count;++j){text+=alphabet[bytes[j]>>4];text+=alphabet[bytes[j]&15];}return text;
}
Digest unhex(const std::string& text) {
    if(text.size()!=64)throw std::invalid_argument("session must contain 64 hex characters");
    Digest bytes{};
    auto nibble=[](char c)->unsigned char {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;throw std::invalid_argument("invalid session hex");};
    for(size_t j=0;j<32;++j)bytes[j]=(nibble(text[2*j])<<4)|nibble(text[2*j+1]);
    return bytes;
}
std::string decimal(FP127::u128 value) {
    std::string text;do{text.push_back(char('0'+value%10));value/=10;}while(value);std::reverse(text.begin(),text.end());return text;
}
void fields(std::ostream& out,const std::vector<FP127>& values) {
    out<<'[';for(size_t j=0;j<values.size();++j){if(j)out<<',';out<<decimal(values[j].val);}out<<']';
}
bool zero(const Digest& value) {return std::all_of(value.begin(),value.end(),[](auto byte){return byte==0;});}
struct Frame {
    uint64_t operation{},sequence{},length{},calls{};Digest identifier{};std::string input,output;
};
Frame read_frame() {
    std::array<char,8> magic{};std::cin.read(magic.data(),magic.size());
    if(!std::cin)throw std::runtime_error("service control EOF or truncated frame");
    if(std::string(magic.data(),magic.size())!="SIVSVC01")throw std::runtime_error("invalid service frame magic");
    Frame frame;frame.operation=read_u64(std::cin);frame.sequence=read_u64(std::cin);
    std::cin.read(reinterpret_cast<char*>(frame.identifier.data()),frame.identifier.size());
    if(!std::cin)throw std::runtime_error("truncated service request identifier");
    frame.length=read_u64(std::cin);frame.calls=read_u64(std::cin);
    const auto input_bytes=read_u64(std::cin),output_bytes=read_u64(std::cin);
    if(input_bytes>4096||output_bytes>4096)throw std::runtime_error("service path exceeds 4096 bytes");
    frame.input.resize(input_bytes);frame.output.resize(output_bytes);
    std::cin.read(frame.input.data(),input_bytes);std::cin.read(frame.output.data(),output_bytes);
    if(!std::cin||frame.input.find('\0')!=std::string::npos||frame.output.find('\0')!=std::string::npos)throw std::runtime_error("invalid service path encoding");
    if(frame.operation==1) {
        if(!frame.length||frame.length>8192||!frame.calls||frame.calls>4096||frame.length*frame.calls>(1UL<<24)||zero(frame.identifier)||frame.input.empty()||frame.output.empty()||frame.input==frame.output)throw std::runtime_error("invalid service request shape or identifier");
    } else if(frame.operation==2) {
        if(frame.length||frame.calls||!zero(frame.identifier)||!frame.input.empty()||!frame.output.empty())throw std::runtime_error("noncanonical service close frame");
    } else throw std::runtime_error("unknown service operation");
    return frame;
}
std::vector<FP127> read_private_input(const Frame& frame,int party) {
    std::ifstream input(frame.input,std::ios::binary);std::array<char,8> magic{};input.read(magic.data(),magic.size());
    if(!input||std::string(magic.data(),magic.size())!="SIVOLE01")throw std::runtime_error("invalid private OLE request magic");
    if(read_u64(input)!=frame.length||read_u64(input)!=frame.calls)throw std::runtime_error("private request shape differs from service frame");
    std::vector<FP127> values(party==emp::ALICE?frame.calls:frame.length*frame.calls);
    for(auto& value:values) {
        std::array<unsigned char,16> bytes{};input.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
        if(!input)throw std::runtime_error("truncated private service input");
        value=FP127::decode(bytes.data());
    }
    if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing private request data");
    return values;
}
std::vector<unsigned char> binding(const Frame& frame,uint64_t indices) {
    std::vector<unsigned char> bytes;
    for(auto word:{uint64_t(1),uint64_t(127),frame.operation,frame.sequence,frame.length,frame.calls,indices})append_u64(bytes,word);
    bytes.insert(bytes.end(),frame.identifier.begin(),frame.identifier.end());return bytes;
}
}

int main(int argc,char** argv) {
    try {
        require_secure_randomness();
        if(argc<10||argc>11)throw std::invalid_argument("usage: ole_service ole_role port self_role peer_role ca cert key peer_name epoch_session_hex [base_fault]");
        const int party=std::stoi(argv[1]),port=std::stoi(argv[2]);
        const auto self=std::stoul(argv[3]),peer_role=std::stoul(argv[4]);
        if((party!=1&&party!=2)||port<1||port>65535||self>3||peer_role>3||self==peer_role)throw std::invalid_argument("invalid service endpoint roles");
        TLSOptions tls;tls.self_role=self;tls.peer_role=peer_role;tls.ca_file=argv[5];tls.cert_file=argv[6];tls.key_file=argv[7];tls.peer_name=argv[8];tls.session=unhex(argv[9]);
        const std::string session=hex(tls.session.data(),tls.session.size()),fault=argc>10?argv[10]:"";
        if(!fault.empty()&&fault!="split-input"&&fault!="response-x"&&fault!="response-z"&&fault!="cope-high-one"&&fault!="cope-high-zero")throw std::invalid_argument("unknown service fault");
        const uint64_t scalar_role=party==emp::ALICE?self:peer_role,vector_role=party==emp::BOB?self:peer_role;
        const std::string protocol="wolverine-r1-ole-service-v1";
        std::vector<unsigned char> channel(protocol.begin(),protocol.end());channel.insert(channel.end(),tls.session.begin(),tls.session.end());append_u64(channel,scalar_role);append_u64(channel,vector_role);append_u64(channel,127);
        tls.channel_id=digest(channel.data(),channel.size());
        const auto setup_start=std::chrono::steady_clock::now();
        matrix_vope::TLSIO io(party==emp::ALICE?nullptr:"127.0.0.1",port,tls);
        EndpointEntropy entropy;RetireEntropyOnFailure retire{entropy};
        auto coin=joint_coin(io,party,entropy,protocol+"/"+session+"/connection");
        const auto coin_hash=digest(&coin,sizeof(coin));
        const std::string domain=protocol+"/"+session+"/"+hex(coin_hash.data(),coin_hash.size());
        PeerContext<FP127,matrix_vope::TLSIO> peer(io,party);
        const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
        std::cout<<"{\"status\":\"ready\",\"sequence\":0,\"ole_role\":"<<party<<",\"self_role\":"<<self<<",\"peer_role\":"<<peer_role<<",\"tls_session\":\""<<session<<"\",\"context_setup_seconds\":"<<setup_seconds<<",\"context_sent_bytes\":"<<io.send_counter<<",\"context_received_bytes\":"<<io.recv_counter<<",\"entropy_substreams\":"<<entropy.substreams()<<",\"security_reviewed\":false,\"transport_authenticated\":true}"<<std::endl;
        uint64_t sequence=0,total_calls=0;std::set<std::string> used;
        try {
            for(;;) {
                auto frame=read_frame();
                if(frame.sequence!=sequence)throw std::runtime_error("service sequence must advance exactly once");
                if(frame.operation==1)require_ole_service_capacity(sequence,peer.cope_indices,total_calls,frame.calls);
                const std::string request_id=hex(frame.identifier.data(),frame.identifier.size());
                if(frame.operation==1&&!used.insert(request_id).second)throw std::runtime_error("service request identifier was already used");
                auto public_binding=binding(frame,peer.cope_indices);
                auto other=commit_exchange(io,party,entropy,public_binding.data(),public_binding.size(),domain+"/header/"+std::to_string(sequence));
                if(other!=public_binding)throw std::runtime_error("service peer operation, sequence, identifier, shape, or OT position mismatch");
                if(frame.operation==2) {
                    entropy.retire();io.close();
                    std::cout<<"{\"status\":\"closed\",\"sequence\":"<<sequence<<",\"total_calls\":"<<total_calls<<",\"total_rot_choices\":"<<peer.cope_indices<<",\"poisoned\":true}"<<std::endl;return 0;
                }
                auto input=read_private_input(frame,party);
                std::ofstream out(frame.output);if(!out)throw std::runtime_error("cannot create private service output");
                const auto started=std::chrono::steady_clock::now();
                const auto sent=io.send_counter,received=io.recv_counter,indices=peer.cope_indices;
                out<<"{\"protocol\":\"wolverine-r1-chosen-input-ole\",\"role\":"<<party<<",\"prime_bits\":127,\"rho\":128,\"security_reviewed\":false,\"full_preprocessing\":false,\"transport_authenticated\":true,\"self_role\":"<<self<<",\"peer_role\":"<<peer_role<<",\"tls_session\":\""<<session<<"\",\"service_sequence\":"<<sequence<<",\"request_id\":\""<<request_id<<"\",\"length\":"<<frame.length<<",\"calls\":"<<frame.calls<<",\"start_rot_choices\":"<<indices<<",\"sessions\":[";
                for(size_t call=0;call<frame.calls;++call) {
                    FP127 scalar;std::vector<FP127> vector;
                    if(party==emp::ALICE)scalar=input[call];
                    else vector.assign(input.begin()+call*frame.length,input.begin()+(call+1)*frame.length);
                    const auto call_start=std::chrono::steady_clock::now();
                    const auto call_sent=io.send_counter,call_received=io.recv_counter,call_indices=peer.cope_indices;
                    auto output=direct_ole(io,party,scalar,vector,frame.length,peer,entropy,domain+"/request/"+std::to_string(sequence)+"/"+request_id+"/call/"+std::to_string(call),fault);
                    if(peer.cope_indices-call_indices!=508)throw std::runtime_error("fresh OT count differs from complete base contract");
                    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-call_start).count();
                    if(call)out<<',';
                    out<<"{\"index\":"<<call<<",\"seconds\":"<<seconds<<",\"sent_bytes\":"<<io.send_counter-call_sent<<",\"received_bytes\":"<<io.recv_counter-call_received<<",\"fresh_rot_choices\":508,\"total_rot_choices\":"<<peer.cope_indices<<",\"entropy_substreams\":"<<entropy.substreams()<<",\"share\":";fields(out,output.share);out<<'}';
                }
                out<<"],\"final_rot_choices\":"<<peer.cope_indices<<"}\n";out.close();if(!out)throw std::runtime_error("private service output write failed");
                const unsigned char accepted=1;
                auto ack=commit_exchange(io,party,entropy,&accepted,1,domain+"/complete/"+std::to_string(sequence)+"/"+request_id);
                if(ack.size()!=1||ack[0]!=1)throw std::runtime_error("peer did not accept complete service request");
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                total_calls+=frame.calls;
                std::cout<<"{\"status\":\"complete\",\"sequence\":"<<sequence<<",\"request_id\":\""<<request_id<<"\",\"length\":"<<frame.length<<",\"calls\":"<<frame.calls<<",\"request_seconds\":"<<seconds<<",\"fresh_rot_choices\":"<<peer.cope_indices-indices<<",\"total_rot_choices\":"<<peer.cope_indices<<",\"sent_bytes\":"<<io.send_counter-sent<<",\"received_bytes\":"<<io.recv_counter-received<<",\"entropy_substreams\":"<<entropy.substreams()<<"}"<<std::endl;
                if(sequence==UINT64_MAX)throw std::runtime_error("service sequence exhausted");
                ++sequence;
            }
        }catch(const std::exception& error) {
            entropy.retire();
            std::cerr<<"OLE service aborted: "<<error.what()<<std::endl;
            std::cout<<"{\"status\":\"aborted\",\"sequence\":"<<sequence<<",\"poisoned\":true}"<<std::endl;
            // Failed EMP streaming sessions cannot be destructed safely. This
            // process is the retirement boundary: it never accepts a retry.
            std::_Exit(1);
        }
    }catch(const std::exception& error) {
        std::cerr<<"OLE service setup aborted: "<<error.what()<<std::endl;return 1;
    }
}

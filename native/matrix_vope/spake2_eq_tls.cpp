// SPDX-License-Identifier: Apache-2.0
#include "spake2_eq_tls.h"
#include "emp-tool/runtime/core/test_mode.h"
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace siv::spake2 {
namespace {
void require(bool condition){if(!condition)throw EqualityFailure();}
Bytes ascii(std::string_view value){return Bytes(value.begin(),value.end());}
void frame(Bytes& output,std::span<const unsigned char> value){
    const auto count=static_cast<std::uint64_t>(value.size());
    for(unsigned i=0;i<8;++i)output.push_back(static_cast<unsigned char>(count>>(8*i)));
    output.insert(output.end(),value.begin(),value.end());
}
void put32(unsigned char* output,std::uint32_t value){for(unsigned i=0;i<4;++i)output[i]=static_cast<unsigned char>(value>>(8*i));}
void put64(unsigned char* output,std::uint64_t value){for(unsigned i=0;i<8;++i)output[i]=static_cast<unsigned char>(value>>(8*i));}
Role role(const matrix_vope::TLSOptions& options){return options.self_role<options.peer_role?Role::A:Role::B;}
Bytes identity(std::uint32_t value){return ascii("party"+std::to_string(value));}
std::array<unsigned char,124> accepted_payload(const matrix_vope::TLSOptions& options,const Binding& binding,
                                              std::uint64_t sequence,bool peer){
    std::array<unsigned char,124> output{};
    std::memcpy(output.data(),"SPEQACK1",8);put64(output.data()+8,sequence);
    const auto sender=peer?options.peer_role:options.self_role;
    const auto receiver=peer?options.self_role:options.peer_role;
    put32(output.data()+16,sender);put32(output.data()+20,receiver);
    output[24]=sender<receiver?0:1;output[25]=1;
    std::copy(options.session.begin(),options.session.end(),output.begin()+28);
    std::copy(options.channel_id.begin(),options.channel_id.end(),output.begin()+60);
    std::copy(binding.begin(),binding.end(),output.begin()+92);
    return output;
}
Binding commitment(std::span<const unsigned char> payload,std::span<const unsigned char> nonce){
    Bytes input;frame(input,ascii("SIV-SPAKE2-TLS-accepted-output-v1"));frame(input,payload);frame(input,nonce);
    Binding out{};unsigned size=0;
    require(EVP_Digest(input.data(),input.size(),out.data(),&size,EVP_sha256(),nullptr)==1 && size==out.size());
    return out;
}
}  // namespace

Context bind_tls_context(const matrix_vope::TLSOptions& options,const Context& context,std::uint64_t sequence){
    require(options.self_role<=3 && options.peer_role<=3 && options.self_role!=options.peer_role);
    require(context.sid.size()==options.session.size() && CRYPTO_memcmp(context.sid.data(),options.session.data(),options.session.size())==0);
    require(std::any_of(options.session.begin(),options.session.end(),[](unsigned char value){return value!=0;}));
    require(context.identity_a==identity(std::min(options.self_role,options.peer_role))
            && context.identity_b==identity(std::max(options.self_role,options.peer_role)));
    require(!context.configuration.empty() && context.configuration.size()<=65000);
    Context output=context;output.configuration.clear();
    frame(output.configuration,ascii("SIV-SPAKE2-TLS-adapter-v1"));
    frame(output.configuration,context.configuration);frame(output.configuration,options.session);frame(output.configuration,options.channel_id);
    const auto begin=output.configuration.size();output.configuration.resize(begin+16);
    put64(output.configuration.data()+begin,sequence);
    put32(output.configuration.data()+begin+8,std::min(options.self_role,options.peer_role));
    put32(output.configuration.data()+begin+12,std::max(options.self_role,options.peer_role));
    return output;
}

struct TLSAdapter::Impl {
    matrix_vope::TLSOptions options;
    std::function<void()> retire;
    std::unique_ptr<matrix_vope::TLSIO> io;
    TLSAdapterStats saved;
    std::uint64_t next=0;
    Impl(const char* address,int port,const matrix_vope::TLSOptions& config,std::function<void()> retirement)
        :options(config),retire(std::move(retirement)) {
        try {
            require(bool(retire) && std::getenv("EMP_TEST_MODE")==nullptr && !emp::is_test_mode());
            io=std::make_unique<matrix_vope::TLSIO>(address,port,options);
        }catch(...){poison();throw EqualityFailure();}
    }
    TLSAdapterStats stats() const {
        TLSAdapterStats output=saved;
        if(io){output.sent_bytes=io->send_counter;output.received_bytes=io->recv_counter;
               output.rounds=io->rounds;output.flushes=io->flushes_count;output.tls_write_calls=io->tls_write_calls();}
        return output;
    }
    void poison() noexcept {
        if(saved.poisoned)return;
        saved=stats();saved.poisoned=true;
        // TLSIO's destructor closes immediately and never flushes pending data.
        // It also closes a channel whose preceding read/write already failed.
        io.reset();
        try{if(retire)retire();}catch(...){}
    }
    void ready() const {require(io && !saved.poisoned);}
    void send(std::span<const unsigned char> bytes){
        ready();require(bytes.size()<=static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        io->send_data(bytes.data(),static_cast<std::int64_t>(bytes.size()));io->flush();
    }
    Bytes receive(std::size_t size){
        ready();require(size<=static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
        Bytes output(size);io->recv_data(output.data(),static_cast<std::int64_t>(size));return output;
    }
    void barrier(const Binding& binding,std::uint64_t sequence){
        ready();
        const auto own=accepted_payload(options,binding,sequence,false);
        const auto expected=accepted_payload(options,binding,sequence,true);
        std::array<unsigned char,32> nonce{};require(RAND_priv_bytes(nonce.data(),nonce.size())==1);
        const auto own_commitment=commitment(own,nonce);
        send(own_commitment);const auto peer_commitment=receive(32);
        Bytes opening(nonce.begin(),nonce.end());opening.insert(opening.end(),own.begin(),own.end());
        send(opening);const auto peer_opening=receive(32+expected.size());
        const std::span<const unsigned char> peer_nonce(peer_opening.data(),32);
        const std::span<const unsigned char> peer_payload(peer_opening.data()+32,expected.size());
        const auto checked=commitment(peer_payload,peer_nonce);
        require(CRYPTO_memcmp(checked.data(),peer_commitment.data(),checked.size())==0);
        require(CRYPTO_memcmp(peer_payload.data(),expected.data(),expected.size())==0);
    }
    void check(const Context& context,std::span<const unsigned char> residual){
        try {
            ready();require(std::getenv("EMP_TEST_MODE")==nullptr && !emp::is_test_mode());
            require(next!=std::numeric_limits<std::uint64_t>::max());
            const auto sequence=next++;
            const auto bound=bind_tls_context(options,context,sequence);
            Callbacks callbacks;
            callbacks.send=[this](std::span<const unsigned char> message){send(message);};
            callbacks.receive=[this](std::size_t size){return receive(size);};
            callbacks.accepted_output_barrier=[this,sequence](const Binding& binding){barrier(binding,sequence);};
            callbacks.poison=[this]{poison();};
            check_private_equal(role(options),bound,residual,callbacks);
            ++saved.completed_checks;
        }catch(...){poison();throw EqualityFailure();}
    }
};

TLSAdapter::TLSAdapter(const char* address,int port,const matrix_vope::TLSOptions& options,
                       std::function<void()> retire_epoch)
    :impl_(std::make_unique<Impl>(address,port,options,std::move(retire_epoch))){}
TLSAdapter::~TLSAdapter()=default;
void TLSAdapter::check(const Context& context,std::span<const unsigned char> residual){impl_->check(context,residual);}
emp::IOChannel& TLSAdapter::transport(){impl_->ready();return *impl_->io;}
void TLSAdapter::poison() noexcept{impl_->poison();}
TLSAdapterStats TLSAdapter::statistics() const{return impl_->stats();}
}  // namespace siv::spake2

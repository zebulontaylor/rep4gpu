// SPDX-License-Identifier: Apache-2.0
// RFC 9382 SPAKE2-P256-SHA256-HKDF-HMAC, with a success-only equality wrapper.
#include "spake2_eq.h"
#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/obj_mac.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace siv::spake2 {
namespace {
using BN=std::unique_ptr<BIGNUM,decltype(&BN_clear_free)>;
using Group=std::unique_ptr<EC_GROUP,decltype(&EC_GROUP_free)>;
using Point=std::unique_ptr<EC_POINT,decltype(&EC_POINT_clear_free)>;
using Ctx=std::unique_ptr<BN_CTX,decltype(&BN_CTX_free)>;
void require(bool condition) { if(!condition)throw EqualityFailure(); }
void wipe(Bytes& bytes) noexcept { if(!bytes.empty())OPENSSL_cleanse(bytes.data(),bytes.size());bytes.clear(); }
struct Erase {
    Bytes& value;
    ~Erase(){wipe(value);}
};
template<std::size_t N>struct EraseArray {
    std::array<unsigned char,N>& value;
    ~EraseArray(){OPENSSL_cleanse(value.data(),value.size());}
};
BN bn() { BN value(BN_secure_new(),BN_clear_free);require(bool(value));BN_set_flags(value.get(),BN_FLG_CONSTTIME);return value; }
Bytes ascii(std::string_view value) { return Bytes(value.begin(),value.end()); }
Bytes hash(const EVP_MD* algorithm,std::span<const unsigned char> input) {
    Bytes result(EVP_MD_get_size(algorithm));unsigned size=0;
    require(EVP_Digest(input.data(),input.size(),result.data(),&size,algorithm,nullptr)==1 && size==result.size());
    return result;
}
Bytes hmac(std::span<const unsigned char> key,std::span<const unsigned char> message) {
    require(key.size()<=INT_MAX);Bytes result(32);unsigned size=0;
    require(HMAC(EVP_sha256(),key.data(),static_cast<int>(key.size()),message.data(),message.size(),result.data(),&size)!=nullptr && size==32);
    return result;
}
void frame(Bytes& output,std::span<const unsigned char> value) {
    const auto size=static_cast<std::uint64_t>(value.size());
    for(unsigned j=0;j<8;++j)output.push_back(static_cast<unsigned char>(size>>(8*j)));
    output.insert(output.end(),value.begin(),value.end());
}
Bytes hex(std::string_view text) {
    require(text.size()%2==0);Bytes out(text.size()/2);
    auto digit=[](char c)->unsigned {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;throw EqualityFailure();};
    for(std::size_t j=0;j<out.size();++j)out[j]=static_cast<unsigned char>(16*digit(text[2*j])+digit(text[2*j+1]));
    return out;
}

// Fixed-work reduction of a 512-bit big-endian hash modulo the P-256 order.
// At every step r<n, so 2*r+bit<2*n; one 257-bit subtraction is sufficient.
std::array<unsigned char,32> reduce512(const std::array<unsigned char,64>& input) {
    constexpr std::uint64_t order[5]={0xf3b9cac2fc632551ULL,0xbce6faada7179e84ULL,
                                     0xffffffffffffffffULL,0xffffffff00000000ULL,0};
    std::uint64_t r[4]={},shifted[5]={},difference[5]={};
    for(unsigned bit=0;bit<512;++bit) {
        std::uint64_t carry=(input[bit/8]>>(7-bit%8))&1;
        for(unsigned word=0;word<4;++word) {
            const std::uint64_t next=r[word]>>63;
            shifted[word]=(r[word]<<1)|carry;carry=next;
        }
        shifted[4]=carry;
        std::uint64_t borrow=0;
        for(unsigned word=0;word<5;++word) {
            const unsigned __int128 full=static_cast<unsigned __int128>(shifted[word])+
                (static_cast<unsigned __int128>(1)<<64)-order[word]-borrow;
            difference[word]=static_cast<std::uint64_t>(full);
            borrow=1-static_cast<std::uint64_t>(full>>64);
        }
        const std::uint64_t mask=0-(borrow^1);
        for(unsigned word=0;word<4;++word)r[word]=(difference[word]&mask)|(shifted[word]&~mask);
    }
    std::array<unsigned char,32> output{};
    for(unsigned word=0;word<4;++word)for(unsigned byte=0;byte<8;++byte)
        output[31-(word*8+byte)]=static_cast<unsigned char>(r[word]>>(8*byte));
    OPENSSL_cleanse(r,sizeof(r));OPENSSL_cleanse(shifted,sizeof(shifted));OPENSSL_cleanse(difference,sizeof(difference));
    return output;
}

class Session {
 public:
    Session(Role role,Bytes identity_a,Bytes identity_b,std::span<const unsigned char> w,Bytes aad)
        :Session(role,std::move(identity_a),std::move(identity_b),w,std::move(aad),Internal{}) {
        do {require(BN_priv_rand_range(private_.get(),order_.get())==1);}while(BN_is_zero(private_.get()));
        BN_set_flags(private_.get(),BN_FLG_CONSTTIME);
    }
#ifdef SIV_SPAKE2_EQ_TESTING
    Session(DeterministicTestOnly,Role role,Bytes identity_a,Bytes identity_b,std::span<const unsigned char> w,
            std::span<const unsigned char> ephemeral,Bytes aad)
        :Session(role,std::move(identity_a),std::move(identity_b),w,std::move(aad),Internal{}) {
        test_mode();require(ephemeral.size()==32);
        require(BN_bin2bn(ephemeral.data(),ephemeral.size(),private_.get())!=nullptr);
        require(!BN_is_zero(private_.get()) && BN_cmp(private_.get(),order_.get())<0);
        BN_set_flags(private_.get(),BN_FLG_CONSTTIME);fixture_=true;
    }
#endif
    ~Session(){erase();}
    Bytes message1() {return guard([&]{require(state_==State::AReady);make_public();state_=State::AWaitPoint;return own_;});}
    Bytes message2(std::span<const unsigned char> peer) {
        return guard([&]{require(state_==State::BReady);auto point=parse(peer);make_public();require(!equal(peer,own_));derive(point.get(),peer);state_=State::BWaitTag;return own_;});
    }
    Bytes message3(std::span<const unsigned char> peer) {
        return guard([&]{require(state_==State::AWaitPoint);require(!equal(peer,own_));auto point=parse(peer);derive(point.get(),peer);state_=State::AWaitTag;return ca_;});
    }
    Bytes message4(std::span<const unsigned char> confirmation) {
        return guard([&]{require(state_==State::BWaitTag);require(equal(confirmation,ca_));state_=State::Confirmed;return cb_;});
    }
    void finish(std::span<const unsigned char> confirmation) {
        guard([&]{require(state_==State::AWaitTag);require(equal(confirmation,cb_));state_=State::Confirmed;});
    }
    bool confirmed() const{return state_==State::Confirmed;}
    bool poisoned() const{return state_==State::Poisoned;}
#ifdef SIV_SPAKE2_EQ_TESTING
    TestTrace trace() const {test_mode();require(fixture_ && state_!=State::Poisoned);return trace_;}
    static void test_mode() {
        const char* value=std::getenv("SIV_SPAKE2_EQ_TEST_MODE");
        require(value && std::strcmp(value,"1")==0);
    }
#endif
 private:
    struct Internal{};
    enum class State{AReady,BReady,AWaitPoint,BWaitTag,AWaitTag,Confirmed,Poisoned};
    Role role_;
    State state_;
    Bytes identity_a_,identity_b_,aad_,own_,ca_,cb_;
    Group group_{nullptr,EC_GROUP_free};
    Ctx ctx_{nullptr,BN_CTX_free};
    BN order_{nullptr,BN_clear_free},field_{nullptr,BN_clear_free},password_{nullptr,BN_clear_free},private_{nullptr,BN_clear_free};
    Point m_{nullptr,EC_POINT_clear_free},n_{nullptr,EC_POINT_clear_free};
#ifdef SIV_SPAKE2_EQ_TESTING
    bool fixture_=false;
    TestTrace trace_;
#endif
    Session(Role role,Bytes identity_a,Bytes identity_b,std::span<const unsigned char> w,Bytes aad,Internal)
        :role_(role),state_(role==Role::A?State::AReady:State::BReady),
         identity_a_(std::move(identity_a)),identity_b_(std::move(identity_b)),aad_(std::move(aad)) {
        require((role==Role::A || role==Role::B) && w.size()==32 && aad_.size()<=8176);
        group_.reset(EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1));ctx_.reset(BN_CTX_secure_new());
        require(group_ && ctx_);order_=bn();field_=bn();password_=bn();private_=bn();
        require(EC_GROUP_get_order(group_.get(),order_.get(),ctx_.get())==1);
        require(EC_GROUP_get_curve(group_.get(),field_.get(),nullptr,nullptr,ctx_.get())==1);
        require(BN_bin2bn(w.data(),w.size(),password_.get())!=nullptr && BN_cmp(password_.get(),order_.get())<0);
        BN_set_flags(password_.get(),BN_FLG_CONSTTIME);
        m_=constant("02886e2f97ace46e55ba9dd7242579f2993b64e16ef3dcab95afd497333d8fa12f");
        n_=constant("03d8bbd6c639c62937b04d997f38c3770719c629d7014d49a24b4f98baa1292b49");
    }
    template<typename Function>auto guard(Function&& function)->decltype(function()) {
        try {require(state_!=State::Poisoned);return function();}
        catch(...) {state_=State::Poisoned;erase();throw EqualityFailure();}
    }
    static bool equal(std::span<const unsigned char> a,std::span<const unsigned char> b) {
        return a.size()==b.size() && CRYPTO_memcmp(a.data(),b.data(),a.size())==0;
    }
    Point point() const {Point out(EC_POINT_new(group_.get()),EC_POINT_clear_free);require(bool(out));return out;}
    Point constant(std::string_view encoding) {
        const auto bytes=hex(encoding);auto out=point();
        require(EC_POINT_oct2point(group_.get(),out.get(),bytes.data(),bytes.size(),ctx_.get())==1);
        require(EC_POINT_is_on_curve(group_.get(),out.get(),ctx_.get())==1 && !EC_POINT_is_at_infinity(group_.get(),out.get()));
        return out;
    }
    Bytes serialize(const EC_POINT* input) const {
        Bytes out(65);
        require(!EC_POINT_is_at_infinity(group_.get(),input));
        require(EC_POINT_point2oct(group_.get(),input,POINT_CONVERSION_UNCOMPRESSED,out.data(),out.size(),ctx_.get())==65);
        return out;
    }
    Point parse(std::span<const unsigned char> encoding) const {
        require(encoding.size()==65 && encoding[0]==4);
        auto x=bn(),y=bn();
        require(BN_bin2bn(encoding.data()+1,32,x.get()) && BN_bin2bn(encoding.data()+33,32,y.get()));
        require(BN_cmp(x.get(),field_.get())<0 && BN_cmp(y.get(),field_.get())<0);
        auto out=point();
        require(EC_POINT_oct2point(group_.get(),out.get(),encoding.data(),encoding.size(),ctx_.get())==1);
        require(EC_POINT_is_on_curve(group_.get(),out.get(),ctx_.get())==1 && !EC_POINT_is_at_infinity(group_.get(),out.get()));
        require(equal(serialize(out.get()),encoding));
        return out;
    }
    void make_public() {
        auto out=point(),mask=point();
        require(EC_POINT_mul(group_.get(),out.get(),private_.get(),nullptr,nullptr,ctx_.get())==1);
        require(EC_POINT_mul(group_.get(),mask.get(),nullptr,role_==Role::A?m_.get():n_.get(),password_.get(),ctx_.get())==1);
        require(EC_POINT_add(group_.get(),out.get(),out.get(),mask.get(),ctx_.get())==1);
        own_=serialize(out.get());
#ifdef SIV_SPAKE2_EQ_TESTING
        if(fixture_)trace_.public_message=own_;
#endif
    }
    void derive(const EC_POINT* peer,std::span<const unsigned char> peer_bytes) {
        auto mask=point(),unmasked=point(),shared=point();
        require(EC_POINT_mul(group_.get(),mask.get(),nullptr,role_==Role::A?n_.get():m_.get(),password_.get(),ctx_.get())==1);
        require(EC_POINT_invert(group_.get(),mask.get(),ctx_.get())==1);
        require(EC_POINT_add(group_.get(),unmasked.get(),peer,mask.get(),ctx_.get())==1);
        require(!EC_POINT_is_at_infinity(group_.get(),unmasked.get()));
        require(EC_POINT_mul(group_.get(),shared.get(),nullptr,unmasked.get(),private_.get(),ctx_.get())==1);
        require(!EC_POINT_is_at_infinity(group_.get(),shared.get()));
        Bytes shared_bytes=serialize(shared.get());Erase erase_shared{shared_bytes};
        Bytes password_bytes(32);Erase erase_password{password_bytes};
        require(BN_bn2binpad(password_.get(),password_bytes.data(),password_bytes.size())==32);
        Bytes transcript;Erase erase_transcript{transcript};
        frame(transcript,identity_a_);frame(transcript,identity_b_);
        if(role_==Role::A){frame(transcript,own_);frame(transcript,peer_bytes);}
        else {frame(transcript,peer_bytes);frame(transcript,own_);}
        frame(transcript,shared_bytes);frame(transcript,password_bytes);
        Bytes digest=hash(EVP_sha256(),transcript);Erase erase_digest{digest};
        const std::span<const unsigned char> ka(digest.data()+16,16);
        const std::array<unsigned char,32> salt{};
        Bytes prk=hmac(salt,ka);Erase erase_prk{prk};
        Bytes info=ascii("ConfirmationKeys");info.insert(info.end(),aad_.begin(),aad_.end());info.push_back(1);
        Bytes confirmation_keys=hmac(prk,info);Erase erase_keys{confirmation_keys};
        ca_=hmac(std::span<const unsigned char>(confirmation_keys.data(),16),transcript);
        cb_=hmac(std::span<const unsigned char>(confirmation_keys.data()+16,16),transcript);
#ifdef SIV_SPAKE2_EQ_TESTING
        if(fixture_) {
            trace_.shared_point=shared_bytes;trace_.transcript=transcript;trace_.hash_tt=digest;
            trace_.ke=Bytes(digest.begin(),digest.begin()+16);trace_.ka=Bytes(digest.begin()+16,digest.end());
            trace_.kc_a=Bytes(confirmation_keys.begin(),confirmation_keys.begin()+16);
            trace_.kc_b=Bytes(confirmation_keys.begin()+16,confirmation_keys.end());trace_.confirm_a=ca_;trace_.confirm_b=cb_;
        }
#endif
    }
    void erase() noexcept {
        if(password_)BN_clear(password_.get());if(private_)BN_clear(private_.get());
        wipe(ca_);wipe(cb_);
#ifdef SIV_SPAKE2_EQ_TESTING
        wipe(trace_.shared_point);wipe(trace_.transcript);wipe(trace_.hash_tt);wipe(trace_.ke);wipe(trace_.ka);
        wipe(trace_.kc_a);wipe(trace_.kc_b);wipe(trace_.confirm_a);wipe(trace_.confirm_b);
#endif
    }
};

Bytes context_bytes(const Context& context) {
    require(!context.sid.empty() && context.sid.size()<=1024 && !context.configuration.empty() && context.configuration.size()<=65536);
    require(!context.identity_a.empty() && !context.identity_b.empty() && context.identity_a!=context.identity_b);
    require(context.identity_a.size()<=1024 && context.identity_b.size()<=1024);
    Bytes out;frame(out,ascii("SIV-matrix-FEQ/RFC9382/SPAKE2-P256-SHA256-HKDF-HMAC/v1"));
    frame(out,context.sid);frame(out,context.configuration);frame(out,context.identity_a);frame(out,context.identity_b);
    return out;
}
}  // namespace

void check_private_equal(Role role,const Context& context,std::span<const unsigned char> residual,const Callbacks& callbacks) {
    try {
        require(callbacks.send && callbacks.receive && callbacks.accepted_output_barrier && callbacks.poison);
        require(std::getenv("EMP_TEST_MODE")==nullptr);
        require((role==Role::A || role==Role::B) && !residual.empty() && residual.size()<=1048576);
        const Bytes common=context_bytes(context);
        Bytes fixed_residual(residual.begin(),residual.end());Erase erase_residual{fixed_residual};
        Bytes residual_digest=hash(EVP_sha256(),fixed_residual);Erase erase_residual_digest{residual_digest};
        Bytes password_input;Erase erase_input{password_input};
        frame(password_input,ascii("SIV-matrix-FEQ-password-SHA512-v1"));frame(password_input,common);frame(password_input,residual_digest);
        Bytes digest=hash(EVP_sha512(),password_input);Erase erase_digest{digest};
        std::array<unsigned char,64> digest_array{};EraseArray<64> erase_array{digest_array};
        std::copy(digest.begin(),digest.end(),digest_array.begin());
        auto password=reduce512(digest_array);EraseArray<32> erase_password{password};
        // The output barrier binds only public context. TT contains K and w;
        // publishing SHA256(TT) would reveal the RFC Ke/Ka key split.
        Bytes binding_bytes=hash(EVP_sha256(),common);Binding binding{};
        std::copy(binding_bytes.begin(),binding_bytes.end(),binding.begin());
        Bytes a_identity,b_identity;frame(a_identity,ascii("SIV-matrix-FEQ-role-A-v1"));frame(a_identity,common);
        frame(b_identity,ascii("SIV-matrix-FEQ-role-B-v1"));frame(b_identity,common);
        Session session(role,std::move(a_identity),std::move(b_identity),password,binding_bytes);
        auto receive=[&](std::size_t size){Bytes message=callbacks.receive(size);require(message.size()==size);return message;};
        if(role==Role::A) {
            callbacks.send(session.message1());
            callbacks.send(session.message3(receive(65)));
            session.finish(receive(32));
        } else {
            callbacks.send(session.message2(receive(65)));
            callbacks.send(session.message4(receive(32)));
        }
        require(session.confirmed());
        callbacks.accepted_output_barrier(binding);
    } catch(...) {
        try {if(callbacks.poison)callbacks.poison();}catch(...){}
        throw EqualityFailure();
    }
}

#ifdef SIV_SPAKE2_EQ_TESTING
struct TestSession::Impl {Session session;
    Impl(Role role,Bytes a,Bytes b,Bytes w,Bytes ephemeral,Bytes aad)
        :session(DeterministicTestOnly{},role,std::move(a),std::move(b),w,ephemeral,std::move(aad)){}
};
TestSession::TestSession(DeterministicTestOnly,Role role,Bytes a,Bytes b,Bytes w,Bytes ephemeral,Bytes aad)
    :impl_(std::make_unique<Impl>(role,std::move(a),std::move(b),std::move(w),std::move(ephemeral),std::move(aad))){}
TestSession::~TestSession()=default;
Bytes TestSession::message1(){return impl_->session.message1();}
Bytes TestSession::message2(std::span<const unsigned char> peer){return impl_->session.message2(peer);}
Bytes TestSession::message3(std::span<const unsigned char> peer){return impl_->session.message3(peer);}
Bytes TestSession::message4(std::span<const unsigned char> peer){return impl_->session.message4(peer);}
void TestSession::finish(std::span<const unsigned char> peer){impl_->session.finish(peer);}
bool TestSession::locally_confirmed()const{return impl_->session.confirmed();}
bool TestSession::poisoned()const{return impl_->session.poisoned();}
TestTrace TestSession::trace()const{return impl_->session.trace();}
std::array<unsigned char,32> test_reduce512(DeterministicTestOnly,const std::array<unsigned char,64>& input){Session::test_mode();return reduce512(input);}
#endif
}  // namespace siv::spake2

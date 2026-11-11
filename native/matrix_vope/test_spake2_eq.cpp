#include "spake2_eq.h"
#include "spake2_rfc9382_vectors.h"
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
using namespace siv::spake2;
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
Bytes hex(const char* s) { Bytes out;std::string v(s);require(v.size()%2==0,"hex length");for(size_t i=0;i<v.size();i+=2)out.push_back(static_cast<unsigned char>(std::stoul(v.substr(i,2),nullptr,16)));return out; }
Bytes ascii(const char* s) { return Bytes(s,s+std::char_traits<char>::length(s)); }
using Vector=siv::spake2::test::Rfc9382Vector;
std::unique_ptr<TestSession> session(Role r,const Vector& v,Bytes aad={}) { return std::make_unique<TestSession>(DeterministicTestOnly{},r,ascii(v.identity_a),ascii(v.identity_b),hex(v.w),hex(r==Role::A?v.x:v.y),aad); }
void trace_matches(const TestTrace& t,const Vector& v,Role role) {
 require(t.public_message==hex(role==Role::A?v.pA:v.pB),"public point");
 require(t.shared_point==hex(v.K),"shared point");require(t.transcript==hex(v.TT),"transcript");require(t.hash_tt==hex(v.hash_TT),"hash transcript");
 require(t.ke==hex(v.Ke)&&t.ka==hex(v.Ka),"session keys");require(t.kc_a==hex(v.KcA)&&t.kc_b==hex(v.KcB),"confirmation keys");require(t.confirm_a==hex(v.cA)&&t.confirm_b==hex(v.cB),"confirmations");
}
template<class F> void rejected(TestSession& s,F f) { bool failed=false;try{f();}catch(const EqualityFailure&){failed=true;}require(failed,"expected rejection");require(s.poisoned(),"failure must poison");bool again=false;try{s.message1();}catch(const EqualityFailure&){again=true;}require(again&&s.poisoned(),"poison permanent"); }
void vectors() {
 for(const auto& v:siv::spake2::test::kRfc9382Vectors) {
 auto a=session(Role::A,v),b=session(Role::B,v);require(!a->locally_confirmed()&&!b->locally_confirmed(),"premature confirmation");auto pa=a->message1();require(pa==hex(v.pA),"pA");auto pb=b->message2(pa);require(pb==hex(v.pB),"pB");trace_matches(b->trace(),v,Role::B);auto ca=a->message3(pb);require(ca==hex(v.cA),"cA");trace_matches(a->trace(),v,Role::A);require(!a->locally_confirmed()&&!b->locally_confirmed(),"premature confirmation");auto cb=b->message4(ca);require(cb==hex(v.cB)&&b->locally_confirmed(),"cB acceptance");a->finish(cb);require(a->locally_confirmed(),"A acceptance");
 }
}
Bytes masked_identity(const Vector& v,bool use_m) {
 auto encoded=hex(use_m?"02886e2f97ace46e55ba9dd7242579f2993b64e16ef3dcab95afd497333d8fa12f":"03d8bbd6c639c62937b04d997f38c3770719c629d7014d49a24b4f98baa1292b49");
 auto w=hex(v.w);auto group=EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1);auto point=EC_POINT_new(group);auto scalar=BN_bin2bn(w.data(),32,nullptr);auto ctx=BN_CTX_new();
 require(group&&point&&scalar&&ctx,"EC allocation");require(EC_POINT_oct2point(group,point,encoded.data(),encoded.size(),ctx)==1,"decode mask");require(EC_POINT_mul(group,point,nullptr,point,scalar,ctx)==1,"multiply mask");Bytes out(65);require(EC_POINT_point2oct(group,point,POINT_CONVERSION_UNCOMPRESSED,out.data(),out.size(),ctx)==65,"encode mask");EC_POINT_free(point);EC_GROUP_free(group);BN_free(scalar);BN_CTX_free(ctx);return out;
}
void fixture_boundaries() {
 const auto& v=siv::spake2::test::kRfc9382Vectors[0];
 unsetenv("SIV_SPAKE2_EQ_TEST_MODE");bool denied=false;try{session(Role::A,v);}catch(const EqualityFailure&){denied=true;}setenv("SIV_SPAKE2_EQ_TEST_MODE","1",1);require(denied,"fixture runtime gate");
 const auto order=hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
 for(int mode=0;mode<3;++mode){auto w=hex(v.w),x=hex(v.x);if(mode==0)w=order;if(mode==1)x=Bytes(32,0);if(mode==2)x=order;bool failed=false;try{TestSession t(DeterministicTestOnly{},Role::A,ascii(v.identity_a),ascii(v.identity_b),w,x);}catch(const EqualityFailure&){failed=true;}require(failed,"invalid scalar constructor");}
 {TestSession a(DeterministicTestOnly{},Role::A,ascii(v.identity_a),ascii(v.identity_b),Bytes(32,0),hex(v.x));TestSession b(DeterministicTestOnly{},Role::B,ascii(v.identity_a),ascii(v.identity_b),Bytes(32,0),hex(v.y));auto pa=a.message1();auto pb=b.message2(pa);auto ca=a.message3(pb);auto cb=b.message4(ca);a.finish(cb);require(a.locally_confirmed()&&b.locally_confirmed(),"zero password scalar handshake");}
 {Bytes aad={'a','a','d',0,'f','i','x','t','u','r','e'};auto a=session(Role::A,v,aad),b=session(Role::B,v,aad);auto pa=a->message1();auto pb=b->message2(pa);auto ca=a->message3(pb);auto cb=b->message4(ca);a->finish(cb);auto t=a->trace();
 // Independent Python hashlib/hmac HKDF-SHA256 values for RFC vector 1 and aad\0fixture.
 require(t.kc_a==hex("b9fea8794b8a54590a1737364407ceb7")&&t.kc_b==hex("79c4202355e4eb22aef18b264c919fda"),"AAD confirmation keys");require(ca==hex("345b753c0ea7ad3c1580371423f97e1066251600b705e2e0731dc467239eb60c")&&cb==hex("120c4e5142eca6264c30b17b268a84059e1cca76c226ea7b21b4dbedc3183bfe"),"AAD confirmations");require(t.transcript==hex(v.TT),"AAD is KDF info");}
}
void negatives() {
 const auto& v=siv::spake2::test::kRfc9382Vectors[0];
 {auto a=session(Role::A,v);auto pa=a->message1();rejected(*a,[&]{a->message3(pa);});}
 {auto b=session(Role::B,v);rejected(*b,[&]{b->message2(hex(v.pB));});}
 for(unsigned char prefix:{2,6}){auto bad=hex(v.pA);bad[0]=prefix;auto b=session(Role::B,v);rejected(*b,[&]{b->message2(bad);});}
 {auto a=session(Role::A,v),b=session(Role::B,v);auto pa=a->message1();auto pb=b->message2(pa);auto ca=a->message3(pb);auto cb=b->message4(ca);a->finish(cb);rejected(*a,[&]{a->finish(cb);});rejected(*b,[&]{b->message4(ca);});}
 {auto b=session(Role::B,v);auto point=masked_identity(v,true);rejected(*b,[&]{b->message2(point);});}
 {auto a=session(Role::A,v);a->message1();auto point=masked_identity(v,false);rejected(*a,[&]{a->message3(point);});}

 for(Bytes bad:{Bytes{},Bytes{0},Bytes(64,0),Bytes(66,0),Bytes(33,2),Bytes(65,0),Bytes(65,255)}) {auto b=session(Role::B,v);rejected(*b,[&]{b->message2(bad);});}
 {auto bad=hex(v.pA);bad[64]^=1;auto b=session(Role::B,v);rejected(*b,[&]{b->message2(bad);});}
 {auto bad=hex(v.pA);std::fill(bad.begin()+1,bad.begin()+33,255);auto b=session(Role::B,v);rejected(*b,[&]{b->message2(bad);});}
 {auto a=session(Role::A,v);rejected(*a,[&]{a->message3(hex(v.pB));});}
 {auto b=session(Role::B,v);rejected(*b,[&]{b->message1();});}
 {auto a=session(Role::A,v);rejected(*a,[&]{a->message2(hex(v.pB));});}
 {auto b=session(Role::B,v);rejected(*b,[&]{b->message3(hex(v.pA));});}
 {auto a=session(Role::A,v);a->message1();a->message3(hex(v.pB));rejected(*a,[&]{a->message4(hex(v.cA));});}
 {auto b=session(Role::B,v);b->message2(hex(v.pA));rejected(*b,[&]{b->finish(hex(v.cB));});}

 {auto a=session(Role::A,v);a->message1();rejected(*a,[&]{a->message1();});}
 {auto b=session(Role::B,v);rejected(*b,[&]{b->message4(hex(v.cA));});}
 {auto a=session(Role::A,v);rejected(*a,[&]{a->finish(hex(v.cB));});}
 for(int mode=0;mode<4;++mode) {
 auto a=session(Role::A,v);auto w=hex(v.w);auto ia=ascii(v.identity_a);Bytes aad;
 if(mode==0)w.back()^=1;if(mode==1)ia.push_back('x');if(mode==2)aad.push_back(1);
 auto b=std::make_unique<TestSession>(DeterministicTestOnly{},Role::B,ia,ascii(v.identity_b),w,hex(v.y),aad);
 auto pa=a->message1();auto pb=b->message2(pa);auto ca=a->message3(pb);if(mode==3)ca=a->trace().confirm_b;
 rejected(*b,[&]{b->message4(ca);});
 }
 {auto a=session(Role::A,v),b=session(Role::B,v);auto pa=a->message1();auto pb=b->message2(pa);auto ca=a->message3(pb);auto cb=b->message4(ca);cb.pop_back();rejected(*a,[&]{a->finish(cb);});}
}
void reduction() {
 const auto order=hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
 BIGNUM* n=BN_bin2bn(order.data(),32,nullptr);BN_CTX* ctx=BN_CTX_new();BIGNUM* x=BN_new();BIGNUM* rem=BN_new();require(n&&ctx&&x&&rem,"BN allocation");std::mt19937_64 rng(9382);
 for(int i=0;i<259;++i) {std::array<unsigned char,64> in{};if(i==1)in.fill(255);else if(i==2)std::copy(order.begin(),order.end(),in.begin()+32);else if(i>2)for(auto& c:in)c=static_cast<unsigned char>(rng());require(BN_bin2bn(in.data(),64,x)&&BN_mod(rem,x,n,ctx),"BN mod");std::array<unsigned char,32> expected{};require(BN_bn2binpad(rem,expected.data(),32)==32,"BN encode");require(test_reduce512(DeterministicTestOnly{},in)==expected,"reduce512");}
 BN_free(n);BN_free(x);BN_free(rem);BN_CTX_free(ctx);
}
struct Channel { std::mutex mu;std::condition_variable cv;std::deque<Bytes> q[2];bool closed=false;int arrived=0;Binding binding{};bool has_binding=false;bool barrier_failed=false; };
Binding production(int mode, unsigned char residual_value=0) {
 Channel ch;bool success[2]={};bool poisoned[2]={};std::exception_ptr unexpected[2];
 auto worker=[&](int side) {Context c{ascii("sid"),ascii("config"),ascii("server"),ascii("client")};Bytes residual=ascii("canonical residual");residual.push_back(residual_value);if(side==1){if(mode==1)residual.push_back(1);if(mode==2)c.configuration.push_back(1);if(mode==3)c.sid.push_back(1);if(mode==4)c.identity_a.push_back(1);if(mode==9)c.identity_b.push_back(1);if(mode==10)std::swap(c.identity_a,c.identity_b);}
 bool first_send=true;
 Callbacks cb;
 cb.send=[&](std::span<const unsigned char> b){if(mode==8&&first_send)std::fill(residual.begin(),residual.end(),static_cast<unsigned char>(side+1));first_send=false;std::lock_guard l(ch.mu);if(ch.closed)throw EqualityFailure();ch.q[1-side].emplace_back(b.begin(),b.end());ch.cv.notify_all();};
 cb.receive=[&](size_t n){std::unique_lock l(ch.mu);if(!ch.cv.wait_for(l,std::chrono::seconds(3),[&]{return ch.closed||!ch.q[side].empty();}))throw EqualityFailure();if(ch.closed)throw EqualityFailure();auto b=std::move(ch.q[side].front());ch.q[side].pop_front();if(b.size()!=n)throw EqualityFailure();return b;};
 cb.poison=[&]{std::lock_guard l(ch.mu);poisoned[side]=true;ch.closed=true;ch.cv.notify_all();};
 cb.accepted_output_barrier=[&](const Binding& b){std::unique_lock l(ch.mu);if(mode==7||(ch.has_binding&&b!=ch.binding)){ch.barrier_failed=true;ch.closed=true;ch.cv.notify_all();throw EqualityFailure();}ch.binding=b;ch.has_binding=true;++ch.arrived;ch.cv.notify_all();ch.cv.wait(l,[&]{return ch.closed||ch.arrived==2;});if(ch.closed||ch.barrier_failed)throw EqualityFailure();};
 try{check_private_equal(mode==5?Role::B:mode==6?Role::A:(side==0?Role::A:Role::B),c,residual,cb);success[side]=true;}catch(const EqualityFailure&){}catch(...){unexpected[side]=std::current_exception();cb.poison();}
 };
 std::thread a(worker,0),b(worker,1);a.join();b.join();for(auto e:unexpected)if(e)std::rethrow_exception(e);
 require((mode==0||mode==8)?(success[0]&&success[1]&&ch.arrived==2):(!success[0]&&!success[1]&&poisoned[0]&&poisoned[1]),"production result");
 return ch.binding;
}
}
int main(int argc,char** argv) {
 if(argc!=2||std::string(argv[1])!="--test-mode"){std::cerr<<"requires explicit --test-mode\n";return 2;}
 setenv("SIV_SPAKE2_EQ_TEST_MODE","1",1);
 try{vectors();fixture_boundaries();negatives();reduction();for(int i=0;i<11;++i)production(i);
 // The output barrier authenticates public context only. A transcript hash
 // depends on the residual and exposes Ke/Ka, so it cannot be this binding.
 require(production(0,17)==production(0,29),"barrier binding must not depend on residual");
 std::cout<<"RFC vectors, negative cases, reduction, production transport: PASS\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

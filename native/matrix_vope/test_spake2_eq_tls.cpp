#include "spake2_eq_tls.h"
#include "spake2_rfc9382_vectors.h"
#include "emp-tool/runtime/core/test_mode.h"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
using namespace siv::spake2;
namespace {
Bytes ascii(const std::string& s){return Bytes(s.begin(),s.end());}
Bytes hex(const char* s){Bytes b;std::string x(s);for(size_t i=0;i<x.size();i+=2)b.push_back(static_cast<unsigned char>(std::stoul(x.substr(i,2),nullptr,16)));return b;}
void little(Bytes& out,uint64_t n,size_t count){for(size_t i=0;i<count;++i)out.push_back(static_cast<unsigned char>(n>>(8*i)));}
void frame(Bytes& out,std::span<const unsigned char> b){little(out,b.size(),8);out.insert(out.end(),b.begin(),b.end());}
Bytes digest(const Bytes& b){Bytes out(32);unsigned n=0;if(EVP_Digest(b.data(),b.size(),out.data(),&n,EVP_sha256(),nullptr)!=1||n!=32)throw EqualityFailure();return out;}
void faulty_barrier(matrix_vope::TLSIO& io,const matrix_vope::TLSOptions& o,const Binding& binding,const std::string& mode){
 Bytes payload=ascii("SPEQACK1");little(payload,0,8);little(payload,o.self_role,4);little(payload,o.peer_role,4);payload.push_back(o.self_role<o.peer_role?0:1);payload.push_back(1);little(payload,0,2);payload.insert(payload.end(),o.session.begin(),o.session.end());payload.insert(payload.end(),o.channel_id.begin(),o.channel_id.end());payload.insert(payload.end(),binding.begin(),binding.end());
 if(payload.size()!=124)throw EqualityFailure();
 if(mode=="barrier_role")payload[24]^=1;else if(mode=="barrier_status")payload[25]=0;else payload[92]^=1;
 Bytes nonce(32);if(RAND_bytes(nonce.data(),32)!=1)throw EqualityFailure();Bytes input;frame(input,ascii("SIV-SPAKE2-TLS-accepted-output-v1"));frame(input,payload);frame(input,nonce);auto commit=digest(input);Bytes peer_commit(32),peer_open(156);io.send_data(commit.data(),32);io.flush();io.recv_data(peer_commit.data(),32);io.send_data(nonce.data(),32);io.send_data(payload.data(),124);io.flush();io.recv_data(peer_open.data(),156);throw EqualityFailure();
}
}
int main(int argc,char** argv){
 if(argc!=7)return 2;
 const int party=std::stoi(argv[1]),port=std::stoi(argv[2]);const std::string mode=argv[3],dir=argv[4];const bool expected_success=std::string(argv[5])=="success",raw=std::string(argv[6])=="raw";
 matrix_vope::TLSOptions o;o.ca_file=dir+"/ca.crt";o.cert_file=dir+"/party"+std::to_string(party)+".crt";o.key_file=dir+"/party"+std::to_string(party)+".key";o.self_role=party;o.peer_role=1-party;o.timeout_ms=800;o.session.fill(31);o.channel_id.fill(47);
 Context c{Bytes(o.session.begin(),o.session.end()),ascii("tls regression"),ascii("party0"),ascii("party1")};Bytes residual(16,0);residual[0]=19;
 if(party==1){if(mode=="unequal"||mode=="wrong_guess")residual[0]^=1;if(mode=="sid")c.sid[0]^=1;if(mode=="configuration")c.configuration.push_back(1);if(mode=="identities")std::swap(c.identity_a,c.identity_b);}
 // A known delta fixes the candidate correction; the candidate still has to match.
 if(mode=="correct_guess"||mode=="wrong_guess"){const unsigned delta=7,correction=52,candidate=party==1&&mode=="wrong_guess"?6:5;residual[0]=static_cast<unsigned char>(correction-delta*candidate);}
 bool retired=false,ok=false,barrier_reached=false,withheld_after_points=false,reuse_rejected=true;unsigned retire_count=0;TLSAdapterStats stats;std::unique_ptr<TLSAdapter> adapter;std::unique_ptr<matrix_vope::TLSIO> peer;const auto start=std::chrono::steady_clock::now();
 if(mode=="emp_test_flag")emp::set_test_mode(true);
 try{
 if(raw){peer=std::make_unique<matrix_vope::TLSIO>(party==0?nullptr:"127.0.0.1",port,o);
 if(mode=="withhold"){auto pa=hex(siv::spake2::test::kRfc9382Vectors[0].pA);Bytes pb(65);peer->send_data(pa.data(),65);peer->flush();peer->recv_data(pb.data(),65);withheld_after_points=true;std::this_thread::sleep_for(std::chrono::milliseconds(1100));retired=true;peer.reset();throw EqualityFailure();}
 auto bound=bind_tls_context(o,c,0);Callbacks cb;cb.send=[&](auto bytes){peer->send_data(bytes.data(),bytes.size());peer->flush();};cb.receive=[&](size_t n){Bytes bytes(n);peer->recv_data(bytes.data(),n);return bytes;};cb.poison=[&]{retired=true;peer.reset();};cb.accepted_output_barrier=[&](const Binding& b){barrier_reached=true;faulty_barrier(*peer,o,b,mode);};check_private_equal(party==0?Role::A:Role::B,bound,residual,cb);ok=true;
 }else{adapter=std::make_unique<TLSAdapter>(party==0?nullptr:"127.0.0.1",port,o,[&]{retired=true;++retire_count;});adapter->check(c,residual);if(mode=="equal_twice")adapter->check(c,residual);ok=true;stats=adapter->statistics();}
 }catch(const std::exception&){if(adapter)stats=adapter->statistics();else {if(raw)retired=true;peer.reset();stats.poisoned=true;}}
 if(adapter&&!ok&&(mode=="unequal"||mode=="wrong_guess")){
 const auto before=adapter->statistics();const auto retire_before=retire_count;bool check_denied=false,transport_denied=false;
 try{adapter->check(c,residual);}catch(const EqualityFailure&){check_denied=true;}
 try{adapter->transport();}catch(const EqualityFailure&){transport_denied=true;}
 const auto after=adapter->statistics();
 reuse_rejected=check_denied&&transport_denied&&retire_count==retire_before&&retire_count==1&&before.sent_bytes==after.sent_bytes&&before.received_bytes==after.received_bytes&&before.rounds==after.rounds&&before.flushes==after.flushes&&before.tls_write_calls==after.tls_write_calls&&before.completed_checks==after.completed_checks&&after.poisoned;
 }
 auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
 std::cout<<"{\"party\":"<<party<<",\"success\":"<<(ok?"true":"false")<<",\"retired\":"<<(retired?"true":"false")<<",\"poisoned\":"<<(stats.poisoned?"true":"false")<<",\"retire_count\":"<<retire_count<<",\"reuse_rejected\":"<<(reuse_rejected?"true":"false")<<",\"sent_bytes\":"<<stats.sent_bytes<<",\"received_bytes\":"<<stats.received_bytes<<",\"rounds\":"<<stats.rounds<<",\"flushes\":"<<stats.flushes<<",\"tls_write_calls\":"<<stats.tls_write_calls<<",\"completed_checks\":"<<stats.completed_checks<<",\"elapsed_ms\":"<<elapsed<<"}\n";
 const bool exercise=(!raw||((mode=="withhold"&&withheld_after_points)||(mode!="withhold"&&barrier_reached)));
 const uint64_t checks=mode=="equal_twice"?2:1;
 const bool test_flag_ok=mode!="emp_test_flag"||(!adapter&&retire_count==1&&stats.sent_bytes==0&&stats.received_bytes==0&&stats.completed_checks==0);
 const bool counters=!ok||raw||(stats.sent_bytes==285*checks&&stats.received_bytes==285*checks&&stats.completed_checks==checks);
 return ok==expected_success&&exercise&&counters&&reuse_rejected&&test_flag_ok&&(ok?!retired:(retired&&stats.poisoned))?0:1;
}

#include "tls_io.h"
#include <openssl/err.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc,char** argv) {
    try {
        if(argc!=12)throw std::runtime_error("usage: test_tls_io server|client port self peer ca cert key sessionhex mode timeout_ms channelhex");
        matrix_vope::TLSOptions options;
        options.self_role=std::stoul(argv[3]);options.peer_role=std::stoul(argv[4]);
        options.ca_file=argv[5];options.cert_file=argv[6];options.key_file=argv[7];options.timeout_ms=std::stoi(argv[10]);
        auto decode=[](const char* text,std::array<unsigned char,32>& output){
            if(std::strlen(text)!=64)throw std::runtime_error("binding must have 64 hex digits");
            for(size_t i=0;i<32;++i){std::string byte(text+2*i,2);size_t read=0;unsigned long value=std::stoul(byte,&read,16);if(read!=2)throw std::runtime_error("invalid hex");output[i]=static_cast<unsigned char>(value);}
        };
        decode(argv[8],options.session);decode(argv[11],options.channel_id);
        bool server=std::string(argv[1])=="server";
        std::string mode=argv[9];
        if(mode=="error-queue")ERR_raise(ERR_LIB_USER,76);
        matrix_vope::TLSIO io(server?nullptr:"127.0.0.1",std::stoi(argv[2]),options);
        if(mode=="stall") {std::this_thread::sleep_for(std::chrono::milliseconds(options.timeout_ms*3));return 0;}
        if(mode=="wait") {unsigned char value;io.recv_data(&value,1);throw std::runtime_error("unexpected byte");}
        if(mode=="flood") {std::vector<unsigned char> data(16*1024*1024,7);ERR_raise(ERR_LIB_USER,78);io.send_data(data.data(),data.size());io.flush();throw std::runtime_error("write unexpectedly completed");}
        if(mode=="drip") {for(int i=0;i<4;++i){unsigned char value=static_cast<unsigned char>(i);io.send_data(&value,1);io.flush();std::this_thread::sleep_for(std::chrono::milliseconds(options.timeout_ms*2/3));}return 0;}
        if(mode=="truncated-send") {const char bytes[]="abc";io.send_data(bytes,3);io.close();return 0;}
        if(mode=="truncated-receive") {char data[4];io.recv_data(data,4);throw std::runtime_error("truncation accepted");}
        if(mode=="abrupt") {const char bytes[]="abc";io.send_data(bytes,3);io.flush();return 0;}
        if(mode=="error-queue") {
            ERR_raise(ERR_LIB_USER,77);
            if(server)std::this_thread::sleep_for(std::chrono::milliseconds(30));
            mode="transfer";
        }
        if(mode!="transfer")throw std::runtime_error("unknown mode");
        std::vector<unsigned char> expected(200003),received(expected.size());
        for(size_t i=0;i<expected.size();++i)expected[i]=static_cast<unsigned char>((i*71+options.session[0])&255);
        if(server) {
            // Thousands of field-sized sends must coalesce before SSL_write.
            for(size_t i=0;i<expected.size();i+=7)io.send_data(expected.data()+i,std::min<size_t>(7,expected.size()-i));
            io.recv_data(received.data(),received.size());
        } else {
            for(size_t i=0;i<received.size();i+=3)io.recv_data(received.data()+i,std::min<size_t>(3,received.size()-i));
            io.send_data(expected.data(),expected.size());io.flush();
        }
        if(received!=expected)throw std::runtime_error("payload differs");
        io.sync();
        std::cout<<"{\"passed\":true,\"sent\":"<<io.send_counter<<",\"received\":"<<io.recv_counter
                 <<",\"tls_write_calls\":"<<io.tls_write_calls()<<",\"flushes\":"<<io.flushes_count<<"}\n";
        io.close();return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}

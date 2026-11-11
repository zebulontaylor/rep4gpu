#include "tls_io.h"
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509v3.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace matrix_vope {
namespace {
using Clock=std::chrono::steady_clock;
using Deadline=Clock::time_point;
[[noreturn]] void fail(const std::string& message) {
    std::string detail=message;
    for(unsigned long error;(error=ERR_get_error());) {
        char text[256];ERR_error_string_n(error,text,sizeof(text));detail+="; ";detail+=text;
    }
    throw std::runtime_error(detail);
}
void wait_fd(int fd,short events,Deadline end) {
    for(;;) {
        auto left=std::chrono::duration_cast<std::chrono::milliseconds>(end-Clock::now()).count();
        if(left<=0)fail("TLS timeout");
        pollfd descriptor{fd,events,0};
        int result=::poll(&descriptor,1,static_cast<int>(std::min<int64_t>(left+1,INT_MAX)));
        if(result>0)return;
        if(result==0)fail("TLS timeout");
        if(errno!=EINTR)fail("TLS poll failed");
    }
}
struct Address {
    sockaddr_storage storage{};socklen_t length{};int family{};
    Address(const std::string& ip,int port) {
        auto* v4=reinterpret_cast<sockaddr_in*>(&storage);
        if(inet_pton(AF_INET,ip.c_str(),&v4->sin_addr)==1) {
            family=AF_INET;length=sizeof(sockaddr_in);v4->sin_family=AF_INET;v4->sin_port=htons(port);return;
        }
        auto* v6=reinterpret_cast<sockaddr_in6*>(&storage);
        if(inet_pton(AF_INET6,ip.c_str(),&v6->sin6_addr)==1) {
            family=AF_INET6;length=sizeof(sockaddr_in6);v6->sin6_family=AF_INET6;v6->sin6_port=htons(port);return;
        }
        throw std::invalid_argument("TLS address must be a numeric IP literal");
    }
};
struct FD {
    int value=-1;
    ~FD(){if(value>=0)::close(value);}
    int release(){int result=value;value=-1;return result;}
};
int connect_socket(const char* peer,int port,const TLSOptions& options,Deadline end) {
    Address address(peer?peer:options.bind_address,port);
    if(!peer) {
        FD listener;listener.value=::socket(address.family,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if(listener.value<0)fail("TLS listener socket failed");
        int yes=1;setsockopt(listener.value,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
        if(::bind(listener.value,reinterpret_cast<sockaddr*>(&address.storage),address.length)||::listen(listener.value,1))fail("TLS listen failed");
        for(;;) {
            int fd=::accept4(listener.value,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);
            if(fd>=0)return fd;
            if(errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR)fail("TLS accept failed");
            wait_fd(listener.value,POLLIN,end);
        }
    }
    for(;;) {
        FD socket;socket.value=::socket(address.family,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);
        if(socket.value<0)fail("TLS client socket failed");
        int result=::connect(socket.value,reinterpret_cast<sockaddr*>(&address.storage),address.length);
        int error=result?errno:0;
        if(error==EINPROGRESS) {
            wait_fd(socket.value,POLLOUT,end);socklen_t length=sizeof(error);
            if(getsockopt(socket.value,SOL_SOCKET,SO_ERROR,&error,&length))fail("TLS connect status failed");
        }
        if(!error)return socket.release();
        if(error!=ECONNREFUSED&&error!=EINTR&&error!=ETIMEDOUT&&error!=EHOSTUNREACH&&error!=ENETUNREACH)fail("TLS connect failed");
        // A late-starting peer may not yet have opened its listener.
        auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(end-Clock::now()).count();
        if(remaining<=0)fail("TLS connect timeout");
        ::poll(nullptr,0,static_cast<int>(std::min<int64_t>(remaining,10)));
    }
}
// Socket BIO uses MSG_NOSIGNAL, so failed TLS writes cannot change the process's
// SIGPIPE handler or kill a role before its controller records the failure.
int bio_write(BIO* bio,const char* data,int length) {
    BIO_clear_retry_flags(bio);
    int fd=*static_cast<int*>(BIO_get_data(bio));
    int result=static_cast<int>(::send(fd,data,length,MSG_NOSIGNAL));
    if(result<0&&(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR))BIO_set_retry_write(bio);
    return result;
}
int bio_read(BIO* bio,char* data,int length) {
    BIO_clear_retry_flags(bio);
    int fd=*static_cast<int*>(BIO_get_data(bio));
    int result=static_cast<int>(::recv(fd,data,length,0));
    if(result<0&&(errno==EAGAIN||errno==EWOULDBLOCK||errno==EINTR))BIO_set_retry_read(bio);
    return result;
}
long bio_control(BIO* bio,int command,long,void* pointer) {
    if(command==BIO_CTRL_FLUSH)return 1;
    if(command==BIO_C_GET_FD) {int fd=*static_cast<int*>(BIO_get_data(bio));if(pointer)*static_cast<int*>(pointer)=fd;return fd;}
    return 0;
}
const BIO_METHOD* socket_method() {
    static BIO_METHOD* method=[] {
        BIO_METHOD* value=BIO_meth_new(BIO_get_new_index()|BIO_TYPE_SOURCE_SINK,"matrix VOPE nonblocking socket");
        if(!value||!BIO_meth_set_write(value,bio_write)||!BIO_meth_set_read(value,bio_read)||!BIO_meth_set_ctrl(value,bio_control)
                ||!BIO_meth_set_create(value,[](BIO* b){BIO_set_init(b,1);return 1;})
                ||!BIO_meth_set_destroy(value,[](BIO*){return 1;}))fail("TLS BIO initialization failed");
        return value;
    }();
    return method;
}
void put32(unsigned char* out,uint32_t value){for(int i=0;i<4;++i)out[i]=static_cast<unsigned char>(value>>(8*i));}
}

struct TLSIO::Impl {
    int fd=-1;SSL_CTX* context=nullptr;SSL* ssl=nullptr;
    TLSOptions options;bool server;bool closed=false;
    std::vector<unsigned char> send_buffer,recv_buffer;
    size_t send_size=0,recv_offset=0,recv_size=0;
    uint64_t write_calls=0;
    Deadline deadline()const{return Clock::now()+std::chrono::milliseconds(options.timeout_ms);}
    void release()noexcept {if(ssl){SSL_free(ssl);ssl=nullptr;}if(context){SSL_CTX_free(context);context=nullptr;}if(fd>=0){::close(fd);fd=-1;}closed=true;}
    ~Impl(){release();}
    void ready()const{if(closed)throw std::runtime_error("TLS channel is closed");}
    void retry(int result,Deadline end) {
        int error=SSL_get_error(ssl,result);
        if(error==SSL_ERROR_WANT_READ)wait_fd(fd,POLLIN,end);
        else if(error==SSL_ERROR_WANT_WRITE)wait_fd(fd,POLLOUT,end);
        else if(error==SSL_ERROR_ZERO_RETURN||(error==SSL_ERROR_SYSCALL&&result==0))fail("TLS EOF before requested bytes completed");
        else fail("TLS handshake or I/O failed");
    }
    void write(const void* source,size_t bytes,Deadline end) {
        auto* data=static_cast<const unsigned char*>(source);
        while(bytes) {
            if(Clock::now()>=end)fail("TLS write timeout");
            ERR_clear_error();size_t done=0;int result=SSL_write_ex(ssl,data,bytes,&done);
            if(result==1) {if(!done)fail("TLS write made no progress");data+=done;bytes-=done;++write_calls;}
            else retry(result,end);
        }
    }
    size_t read(void* destination,size_t capacity,Deadline end) {
        for(;;) {
            if(Clock::now()>=end)fail("TLS read timeout");
            ERR_clear_error();size_t done=0;int result=SSL_read_ex(ssl,destination,capacity,&done);
            if(result==1) {if(!done)fail("TLS EOF");return done;}
            retry(result,end);
        }
    }
    void read_exact(void* destination,size_t bytes,Deadline end) {
        auto* data=static_cast<unsigned char*>(destination);
        while(bytes){size_t done=read(data,bytes,end);data+=done;bytes-=done;}
    }
    bool drain(Deadline end) {
        if(!send_size)return false;
        write(send_buffer.data(),send_size,end);send_size=0;return true;
    }
    Impl(const char* address,int port,const TLSOptions& config):options(config),server(address==nullptr) {
        try {
            if(port<1||port>65535||options.self_role>3||options.peer_role>3||options.self_role==options.peer_role
                    ||options.timeout_ms<1||options.buffer_bytes<1024||options.buffer_bytes>(1u<<24))throw std::invalid_argument("invalid TLS role, port, deadline, or buffer size");
            std::string self_name="party"+std::to_string(options.self_role);
            std::string peer_name="party"+std::to_string(options.peer_role);
            if(!options.peer_name.empty()&&options.peer_name!=peer_name)throw std::invalid_argument("TLS peer name must match its role");
            if(std::all_of(options.session.begin(),options.session.end(),[](unsigned char b){return b==0;}))throw std::invalid_argument("TLS session binding must be nonzero");
            context=SSL_CTX_new(TLS_method());if(!context)fail("TLS context creation failed");
            if(!SSL_CTX_set_min_proto_version(context,TLS1_3_VERSION)||!SSL_CTX_set_max_proto_version(context,TLS1_3_VERSION)
                    ||!SSL_CTX_load_verify_locations(context,options.ca_file.c_str(),nullptr)
                    ||SSL_CTX_use_certificate_chain_file(context,options.cert_file.c_str())!=1
                    ||SSL_CTX_use_PrivateKey_file(context,options.key_file.c_str(),SSL_FILETYPE_PEM)!=1
                    ||SSL_CTX_check_private_key(context)!=1)fail("TLS certificate, key, or CA loading failed");
            unsigned int name_flags=X509_CHECK_FLAG_NEVER_CHECK_SUBJECT|X509_CHECK_FLAG_NO_WILDCARDS;
            if(X509_check_host(SSL_CTX_get0_certificate(context),self_name.c_str(),self_name.size(),name_flags,nullptr)!=1)fail("TLS local certificate role mismatch");
            SSL_CTX_set_verify(context,SSL_VERIFY_PEER|SSL_VERIFY_FAIL_IF_NO_PEER_CERT,nullptr);
            SSL_CTX_set_options(context,SSL_OP_NO_COMPRESSION|SSL_OP_NO_RENEGOTIATION);
            SSL_CTX_set_session_cache_mode(context,SSL_SESS_CACHE_OFF);
            Deadline end=deadline();fd=connect_socket(address,port,options,end);
            int yes=1;if(setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&yes,sizeof(yes)))fail("TLS TCP_NODELAY failed");
            ssl=SSL_new(context);if(!ssl)fail("TLS session creation failed");
            SSL_set_mode(ssl,SSL_MODE_ENABLE_PARTIAL_WRITE|SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
            BIO* bio=BIO_new(socket_method());if(!bio)fail("TLS socket BIO creation failed");
            BIO_set_data(bio,&fd);SSL_set_bio(ssl,bio,bio);
            X509_VERIFY_PARAM* parameters=SSL_get0_param(ssl);
            X509_VERIFY_PARAM_set_hostflags(parameters,name_flags);
            if(X509_VERIFY_PARAM_set1_host(parameters,peer_name.c_str(),peer_name.size())!=1)fail("TLS peer-name setup failed");
            for(;;){ERR_clear_error();int result=server?SSL_accept(ssl):SSL_connect(ssl);if(result==1)break;retry(result,end);}
            X509* certificate=SSL_get1_peer_certificate(ssl);
            bool valid=certificate&&SSL_get_verify_result(ssl)==X509_V_OK&&X509_check_host(certificate,peer_name.c_str(),peer_name.size(),name_flags,nullptr)==1;
            X509_free(certificate);if(!valid)fail("TLS peer certificate role mismatch");
            std::array<unsigned char,80> own{},expected{},received{};
            std::memcpy(own.data(),"MVOPTLS1",8);put32(own.data()+8,options.self_role);put32(own.data()+12,options.peer_role);
            std::copy(options.session.begin(),options.session.end(),own.begin()+16);std::copy(options.channel_id.begin(),options.channel_id.end(),own.begin()+48);
            expected=own;put32(expected.data()+8,options.peer_role);put32(expected.data()+12,options.self_role);
            write(own.data(),own.size(),end);read_exact(received.data(),received.size(),end);
            if(CRYPTO_memcmp(expected.data(),received.data(),expected.size()))fail("TLS peer role, session, or channel binding mismatch");
            write_calls=0;send_buffer.resize(options.buffer_bytes);recv_buffer.resize(options.buffer_bytes);
        }catch(...){release();throw;}
    }
};

TLSIO::TLSIO(const char* address,int port,const TLSOptions& options):impl_(std::make_unique<Impl>(address,port,options)){}
TLSIO::~TLSIO()=default;
void TLSIO::send_data_internal(const void* source,int64_t bytes) {
    impl_->ready();if(bytes<0||(!source&&bytes))throw std::invalid_argument("invalid TLS send buffer");
    auto end=impl_->deadline();auto* data=static_cast<const unsigned char*>(source);size_t left=static_cast<size_t>(bytes);
    try {
        while(left) {
            size_t count=std::min(left,impl_->send_buffer.size()-impl_->send_size);
            std::memcpy(impl_->send_buffer.data()+impl_->send_size,data,count);impl_->send_size+=count;data+=count;left-=count;
            if(impl_->send_size==impl_->send_buffer.size()){impl_->drain(end);++flushes_count;}
        }
    }catch(...){impl_->release();throw;}
}
void TLSIO::recv_data_internal(void* destination,int64_t bytes) {
    impl_->ready();if(bytes<0||(!destination&&bytes))throw std::invalid_argument("invalid TLS receive buffer");
    auto end=impl_->deadline();auto* data=static_cast<unsigned char*>(destination);size_t left=static_cast<size_t>(bytes);
    try {
        if(impl_->drain(end))++flushes_count;
        while(left) {
            if(impl_->recv_offset==impl_->recv_size){impl_->recv_size=impl_->read(impl_->recv_buffer.data(),impl_->recv_buffer.size(),end);impl_->recv_offset=0;}
            size_t count=std::min(left,impl_->recv_size-impl_->recv_offset);
            std::memcpy(data,impl_->recv_buffer.data()+impl_->recv_offset,count);data+=count;left-=count;impl_->recv_offset+=count;
        }
    }catch(...){impl_->release();throw;}
}
void TLSIO::flush(){impl_->ready();try{if(impl_->drain(impl_->deadline()))++flushes_count;}catch(...){impl_->release();throw;}}
void TLSIO::sync(){unsigned char marker=0x5a,received=0;if(impl_->server){send_data(&marker,1);recv_data(&received,1);}else{recv_data(&received,1);send_data(&marker,1);flush();}if(received!=marker){impl_->release();fail("TLS synchronization mismatch");}}
void TLSIO::close(){
    if(impl_->closed)return;
    try{
        flush();auto end=impl_->deadline();
        for(;;){ERR_clear_error();int result=SSL_shutdown(impl_->ssl);if(result>=0)break;impl_->retry(result,end);}
        impl_->release();
    }catch(...){impl_->release();throw;}
}
uint64_t TLSIO::tls_write_calls()const{return impl_->write_calls;}
} // namespace matrix_vope

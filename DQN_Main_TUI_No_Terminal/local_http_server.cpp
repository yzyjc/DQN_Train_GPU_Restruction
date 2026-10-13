// Collaboration signature: GPT6-Astra. Win32 transport adapter only.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#include "local_http_server.h"
#include <thread>
#include <atomic>
#include <fstream>
#include <algorithm>
#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib,"bcrypt.lib")
namespace tui_http {
static std::string lower(std::string s){for(char& c:s)if(c>='A' && c<='Z')c+=(char)('a'-'A');return s;}
static std::string trim(std::string s){auto a=s.find_first_not_of(" \t");auto b=s.find_last_not_of(" \t");return a==s.npos?"":s.substr(a,b-a+1);}
struct Server::Impl {
    SOCKET listener=INVALID_SOCKET;std::atomic<bool> running{false};std::thread worker;
    Controller controller;int bound=0;bool wsa=false;std::string token,message="HTTP OFF";
    std::wstring discovery;
    std::string read_discovery(){std::ifstream f(discovery.c_str());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
    void remove_discovery(){const auto old=read_discovery();if(old.find("\"pid\":"+std::to_string(GetCurrentProcessId())+",")!=old.npos)DeleteFileW(discovery.c_str());}
    bool write_discovery(){
        const std::wstring temp=discovery+L".tmp."+std::to_wstring(GetCurrentProcessId());
        std::ofstream f(temp.c_str(),std::ios::trunc);if(!f)return false;
        f<<"{\"service\":\"dqn-tui\",\"api_version\":1,\"pid\":"<<GetCurrentProcessId()
            <<",\"port\":"<<bound<<",\"url\":\"http://127.0.0.1:"<<bound<<"\",\"token\":"<<quote(token)<<"}\n";
        f.close();if(!f)return false;
        if(!MoveFileExW(temp.c_str(),discovery.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){DeleteFileW(temp.c_str());return false;}return true;
    }
    Response receive(SOCKET client){
        DWORD timeout=1000;setsockopt(client,SOL_SOCKET,SO_RCVTIMEO,(char*)&timeout,sizeof(timeout));
        setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,(char*)&timeout,sizeof(timeout));
        std::string data;char buf[2048];ULONGLONG start=GetTickCount64();size_t headers_end;
        while((headers_end=data.find("\r\n\r\n"))==data.npos){
            if(data.size()>8192 || GetTickCount64()-start>3000)return error(413,"Headers too large or request too slow.");
            int n=recv(client,buf,sizeof(buf),0);if(n<=0)return error(400,"Incomplete HTTP headers.");data.append(buf,n);
        }
        if(headers_end>8192)return error(413,"Headers exceed 8192 bytes.");
        Request q;std::istringstream head(data.substr(0,headers_end));std::string line,version,extra;
        if(!std::getline(head,line))return error(400,"Missing request line.");std::istringstream first(line);
        if(!(first>>q.method>>q.path>>version) || (first>>extra) || version!="HTTP/1.1")return error(400,"Use HTTP/1.1 with a valid request line.");
        while(std::getline(head,line)){if(!line.empty() && line.back()=='\r')line.pop_back();size_t colon=line.find(':');
            if(colon==line.npos)return error(400,"Malformed HTTP header.");std::string key=lower(trim(line.substr(0,colon)));
            if(q.headers.count(key))return error(400,"Duplicate HTTP header.");q.headers[key]=trim(line.substr(colon+1));
        }
        const std::string host="127.0.0.1:"+std::to_string(bound),local="localhost:"+std::to_string(bound);
        if(!q.headers.count("host") || (q.headers["host"]!=host && lower(q.headers["host"])!=local))return error(400,"Host must match the local endpoint from the discovery file.");
        if(q.headers.count("transfer-encoding"))return error(400,"Chunked requests are unsupported; use Content-Length.");
        size_t length=0;if(q.headers.count("content-length")){const auto& text=q.headers["content-length"];
            if(text.empty() || text.find_first_not_of("0123456789")!=text.npos || text.size()>5)return error(400,"Invalid Content-Length.");length=strtoul(text.c_str(),nullptr,10);}
        if(length>4096)return error(413,"JSON body exceeds 4096 bytes.");
        q.body=data.substr(headers_end+4);
        while(q.body.size()<length){if(GetTickCount64()-start>3000)return error(408,"Request body timeout.");int n=recv(client,buf,sizeof(buf),0);if(n<=0)return error(400,"Incomplete request body.");q.body.append(buf,n);}
        if(q.body.size()!=length)return error(400,"Body length mismatch or pipelined request; use one request per connection.");
        if(q.method=="GET" && length)return error(400,"GET requests must not carry a body.");
        return route(q,token,controller);
    }
    void run(){
        while(running){fd_set set;FD_ZERO(&set);FD_SET(listener,&set);timeval timeout={0,200000};
            if(select(0,&set,nullptr,nullptr,&timeout)<=0)continue;
            SOCKET client=accept(listener,nullptr,nullptr);if(client==INVALID_SOCKET)continue;
            Response r;try{r=receive(client);}catch(...){r=error(500,"Service request failed; inspect the TUI state and retry.");}
            std::string reason=r.status==200?"OK":r.status==202?"Accepted":r.status==400?"Bad Request":r.status==401?"Unauthorized":r.status==403?"Forbidden":r.status==404?"Not Found":"Error";
            std::string reply="HTTP/1.1 "+std::to_string(r.status)+" "+reason+"\r\nContent-Type: application/json; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: "+std::to_string(r.body.size())+"\r\n\r\n"+r.body;
            size_t sent=0;while(sent<reply.size()){int n=send(client,reply.data()+sent,(int)(reply.size()-sent),0);if(n<=0)break;sent+=n;}
            closesocket(client);
        }
    }
    void stop(){running=false;if(worker.joinable())worker.join();if(listener!=INVALID_SOCKET){closesocket(listener);listener=INVALID_SOCKET;}
        if(bound)remove_discovery();bound=0;if(wsa){WSACleanup();wsa=false;}message="HTTP OFF";
    }
};
Server::Server():impl(new Impl){}Server::~Server(){stop();}
void Server::stop(){impl->stop();}int Server::port()const{return impl->bound;}std::string Server::status()const{return impl->message;}
bool Server::start(const Controller& controller,int preferred_port){
    if(impl->running)return true;
    if(preferred_port<1024 || preferred_port>65526){impl->message="HTTP invalid preferred port";return false;}
    wchar_t exe[32768];DWORD n=GetModuleFileNameW(nullptr,exe,32768);if(!n || n>=32768)return false;
    std::wstring directory(exe,n);auto slash=directory.find_last_of(L"\\/");directory.resize(slash+1);
    impl->discovery=directory+L"tui_http_endpoint.json";
    int previous=0;std::map<std::string,Value> old;if(FlatJson(impl->read_discovery()).parse(old) && old.count("port") && old["port"].number)previous=(int)old["port"].numeric;
    unsigned char bytes[32];if(BCryptGenRandom(nullptr,bytes,sizeof(bytes),BCRYPT_USE_SYSTEM_PREFERRED_RNG)!=0){impl->message="HTTP token generation failed";return false;}
    impl->token.clear();const char hex[]="0123456789abcdef";for(unsigned char b:bytes){impl->token+=hex[b>>4];impl->token+=hex[b&15];}
    WSADATA w;if(WSAStartup(MAKEWORD(2,2),&w)!=0){impl->message="HTTP sockets unavailable";return false;}impl->wsa=true;
    int candidates[11]={preferred_port};int count=1;if(previous>preferred_port && previous<=preferred_port+9)candidates[count++]=previous;
    for(int p=preferred_port+1;p<=preferred_port+9;++p)if(p!=previous)candidates[count++]=p;
    for(int i=0;i<count;++i){SOCKET s=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);if(s==INVALID_SOCKET)continue;
        BOOL exclusive=TRUE;setsockopt(s,SOL_SOCKET,SO_EXCLUSIVEADDRUSE,(char*)&exclusive,sizeof(exclusive));
        sockaddr_in addr={};addr.sin_family=AF_INET;addr.sin_port=htons((u_short)candidates[i]);addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        if(bind(s,(sockaddr*)&addr,sizeof(addr))==0 && listen(s,8)==0){impl->listener=s;impl->bound=candidates[i];break;}closesocket(s);
    }
    if(!impl->bound){impl->stop();impl->message="HTTP port range unavailable";return false;}
    impl->controller=controller;
    if(!impl->write_discovery()){impl->stop();impl->message="HTTP discovery file unavailable";return false;}
    impl->running=true;try{impl->worker=std::thread([p=impl.get()](){p->run();});}
    catch(...){impl->stop();impl->message="HTTP worker unavailable";return false;}
    impl->message="HTTP 127.0.0.1:"+std::to_string(impl->bound);return true;
}
}

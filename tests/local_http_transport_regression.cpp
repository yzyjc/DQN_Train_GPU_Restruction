// Collaboration signature: GPT6-Astra. Real loopback sockets; fake Trainer adapter.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include "../DQN_Main_TUI_No_Terminal/local_http_server.h"
#include <fstream>
#include <cstdio>
#include <atomic>
#pragma comment(lib,"ws2_32.lib")
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
static std::string request(int port,const std::string& extra){
    SOCKET s=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a={};a.sin_family=AF_INET;a.sin_port=htons((u_short)port);a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(connect(s,(sockaddr*)&a,sizeof(a))){printf("CONNECT error %d\n",WSAGetLastError());closesocket(s);return "";}
    std::string q="GET /api/status HTTP/1.1\r\nHost: 127.0.0.1:"+std::to_string(port)+"\r\n"+extra+"\r\n";
    int sent=send(s,q.data(),(int)q.size(),0);std::string out;char b[4096];int n;while((n=recv(s,b,sizeof(b),0))>0)out.append(b,n);if(out.empty())printf("SOCKET sent=%d read=%d error=%d\n",sent,n,WSAGetLastError());closesocket(s);return out;
}
int main(){
    WSADATA w;NEED(!WSAStartup(MAKEWORD(2,2),&w));SOCKET hold=socket(AF_INET,SOCK_STREAM,0);
    BOOL exclusive=TRUE;setsockopt(hold,SOL_SOCKET,SO_EXCLUSIVEADDRUSE,(char*)&exclusive,sizeof(exclusive));
    sockaddr_in a={};a.sin_family=AF_INET;a.sin_port=htons(17865);a.sin_addr.s_addr=htonl(INADDR_LOOPBACK);NEED(!bind(hold,(sockaddr*)&a,sizeof(a)));NEED(!listen(hold,1));
    std::atomic<int> calls{0};tui_http::Server server;NEED(server.start([&](const std::string& c){++calls;return tui_http::Response{200,"{\"command\":"+tui_http::quote(c)+"}"};}));
    NEED(server.port()>17865 && server.port()<=17874);
    wchar_t path[32768];GetModuleFileNameW(nullptr,path,32768);std::wstring f=path;f.resize(f.find_last_of(L"\\/")+1);f+=L"tui_http_endpoint.json";
    std::ifstream in(f.c_str());std::string contents((std::istreambuf_iterator<char>(in)),{});in.close();
    std::map<std::string,tui_http::Value> values;NEED(tui_http::FlatJson(contents).parse(values));NEED(values["token"].text.size()==64);
    const auto unauth=request(server.port(),"");printf("UNAUTH: %s\n",unauth.c_str());NEED(unauth.find("401 Unauthorized")!=std::string::npos && calls==0);
    std::string auth="Authorization: Bearer "+values["token"].text+"\r\n";
    NEED(request(server.port(),auth).find("GET_STATUS")!=std::string::npos && calls==1);
    NEED(request(server.port(),auth+"Origin: http://example.test\r\n").find("403 Forbidden")!=std::string::npos && calls==1);
    server.stop();NEED(GetFileAttributesW(f.c_str())==INVALID_FILE_ATTRIBUTES);closesocket(hold);WSACleanup();
    puts("PASS HTTP real sockets, preferred-port collision fallback, token, Origin rejection, discovery cleanup");return 0;
}

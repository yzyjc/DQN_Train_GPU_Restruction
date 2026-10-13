// Collaboration signature: GPT6-Astra.
#include "../DQN_Main_TUI_No_Terminal/local_http_api.h"
#include <cstdio>
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    int calls=0;std::string cmd;tui_http::Controller controller=[&](const std::string& c){++calls;cmd=c;return tui_http::Response{202,"{\"status\":\"ok\"}"};};
    tui_http::Request r;r.method="GET";r.path="/api/status";NEED(tui_http::route(r,"secret",controller).status==401 && !calls);
    r.headers["authorization"]="Bearer secret";NEED(tui_http::route(r,"secret",controller).status==202 && cmd=="GET_STATUS");
    r.path="/api/config";r.method="PATCH";r.headers["content-type"]="application/json";
    r.body="{\"key\":\"learning_rate\",\"value\":-1}";NEED(tui_http::route(r,"secret",controller).status==202 && cmd=="SET learning_rate -1");
    int before=calls;for(const char* bad:{"{\"key\":\"learning_rate\",\"value\":2}","{\"key\":\"learning_rate\",\"value\":1e999}","{\"key\":\"learning_rate\",\"key\":\"epsilon\",\"value\":0}","{\"key\":\"curriculum_enabled\",\"value\":0.5}"}){
        r.body=bad;auto out=tui_http::route(r,"secret",controller);NEED(out.status==400 && out.body.find("\"help\"")!=out.body.npos && calls==before);}
    r.headers["origin"]="http://evil";NEED(tui_http::route(r,"secret",controller).status==403 && calls==before);r.headers.erase("origin");
    r.path="/api/trainer/action";r.method="POST";r.body="{\"action\":\"save_and_stop\"}";NEED(tui_http::route(r,"secret",controller).status==202 && cmd=="SESSION_STOP");
    NEED(!tui_http::parameter("curriculum_enabled",0.999999999) && !tui_http::parameter("step_outcome_enabled",1.000000001));
    r.path="/wrong";NEED(tui_http::route(r,"secret",controller).status==404);
    puts("PASS local HTTP authentication, strict validation, tutorial, queued actions");return 0;
}

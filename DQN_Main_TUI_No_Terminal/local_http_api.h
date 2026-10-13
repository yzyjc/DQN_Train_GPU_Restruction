#pragma once
// Collaboration signature: GPT6-Astra. Portable API contract; no Win32/UI dependencies.
#include <string>
#include <map>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include "competency_state.h"
#include "step_outcome.h"
namespace tui_http {
inline std::string quote(const std::string& s){std::string r="\"";for(unsigned char c:s){
    if(c=='"' || c=='\\'){r+='\\';r+=(char)c;}else if(c=='\n')r+="\\n";
    else if(c=='\r')r+="\\r";else if(c=='\t')r+="\\t";else if(c>=32)r+=(char)c;
}return r+'"';}
struct Value {bool number=false;std::string text;double numeric=0;};
class FlatJson {
    const std::string& s;size_t i=0;
    void space(){while(i<s.size() && (s[i]==' ' || s[i]=='\t' || s[i]=='\r' || s[i]=='\n'))++i;}
    bool str(std::string& v){if(i>=s.size() || s[i++]!='"')return false;while(i<s.size()){
        unsigned char c=s[i++];if(c=='"')return true;if(c<32)return false;
        if(c=='\\'){if(i>=s.size())return false;c=s[i++];if(c=='n')c='\n';else if(c=='r')c='\r';else if(c=='t')c='\t';else if(c!='"' && c!='\\' && c!='/')return false;}v+=(char)c;
    }return false;}
public:
    explicit FlatJson(const std::string& input):s(input){}
    bool parse(std::map<std::string,Value>& out){space();if(i>=s.size() || s[i++]!='{')return false;space();
        if(i<s.size() && s[i]=='}'){++i;space();return i==s.size();}
        for(int n=0;n<32;++n){std::string key;Value v;if(!str(key) || out.count(key))return false;space();if(i>=s.size() || s[i++]!=':')return false;space();
            if(i<s.size() && s[i]=='"'){if(!str(v.text))return false;}
            else {size_t begin=i;if(i<s.size() && s[i]=='-')++i;if(i>=s.size() || s[i]<'0' || s[i]>'9')return false;
                if(s[i]=='0')++i;else while(i<s.size() && s[i]>='0' && s[i]<='9')++i;
                if(i<s.size() && s[i]=='.'){++i;size_t a=i;while(i<s.size() && s[i]>='0' && s[i]<='9')++i;if(a==i)return false;}
                if(i<s.size() && (s[i]=='e' || s[i]=='E')){++i;if(i<s.size() && (s[i]=='+' || s[i]=='-'))++i;size_t a=i;while(i<s.size() && s[i]>='0' && s[i]<='9')++i;if(a==i)return false;}
                v.text=s.substr(begin,i-begin);v.numeric=strtod(v.text.c_str(),nullptr);v.number=true;if(!std::isfinite(v.numeric))return false;
            }out[key]=v;space();if(i>=s.size())return false;if(s[i]=='}'){++i;space();return i==s.size();}if(s[i++]!=',')return false;space();
        }return false;
    }
};
struct Request {std::string method,path,body;std::map<std::string,std::string> headers;};
struct Response {int status=200;std::string body;};
using Controller=std::function<Response(const std::string&)>;
inline std::string help(){
    return "{\"api_version\":1,\"service\":\"dqn-tui\",\"discovery\":\"tui_http_endpoint.json beside the TUI executable (url + token + pid)\","
        "\"authentication\":\"Authorization: Bearer <token>; /api/help and /api/health are public; browser Origin requests are rejected\","
        "\"routes\":[\"GET /api/status\",\"GET /api/config\",\"GET /api/competency\",\"PATCH /api/config\",\"POST /api/trainer/action\",\"GET /api/action\"],"
        "\"config_example\":{\"key\":\"learning_rate\",\"value\":-1},"
        "\"action_example\":{\"action\":\"save_snapshot\"},"
        "\"actions\":[\"start\",\"save_and_stop\",\"save_snapshot\",\"resume\"],"
        "\"semantics\":\"202 means queued, never applied. Verify actual parameters with GET /api/config; use GET /api/action for TUI action progress.\","
        "\"notes\":\"One configuration key per request; finite numeric values only. learning_rate=-1 and epsilon=-1 restore AUTO. Curriculum/rehearsal startup flags require independent networks.\","
        "\"examples\":[\"curl <url>/api/help\",\"curl -H 'Authorization: Bearer <token>' <url>/api/status\",\"curl -X PATCH -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' -d '{\\\"key\\\":\\\"learning_rate\\\",\\\"value\\\":-1}' <url>/api/config\"]}";
}
inline Response error(int status,const std::string& message){return {status,"{\"status\":\"error\",\"message\":"+quote(message)+",\"help\":"+help()+"}"};}
inline bool parameter(const std::string& key,double v){
    if(!std::isfinite(v) || std::fabs(v)>3.4e38)return false;
    int k=competency_key(key.c_str());if(k>=0){const auto& limit=COMPETENCY_SETTINGS[k];if(v<limit.lo || v>limit.hi || (k<13 && v!=std::floor(v)))return false;CompetencyConfig c;return competency_set(c,k,(float)v);}
    k=step_outcome_key(key.c_str());if(k>=0){if(k==0 && v!=0 && v!=1)return false;StepOutcomeConfig c;return step_outcome_set(c,k,(float)v);}
    if(key=="learning_rate")return v==-1 || (v>0 && v<=1);
    if(key=="epsilon")return v==-1 || (v>=0 && v<=1);
    if(key=="train_repeat")return v>=1 && v<=20 && v==std::floor(v);
    if(key=="memory_enabled")return v==0 || v==1;
    if(key=="memory_confidence")return v>=0 && v<=1;
    if(key=="memory_scale" || key=="block_win_scale" || key=="advanced_scale" || key=="live_two_scale")return v>=0 && v<=10;
    return false;
}
inline Response route(const Request& q,const std::string& token,const Controller& control){
    if(q.headers.count("origin"))return error(403,"Browser-origin requests are not allowed; use a local agent or CLI.");
    if(q.path=="/api/help" && q.method=="GET")return {200,help()};
    if(q.path=="/api/health" && q.method=="GET")return {200,"{\"service\":\"dqn-tui\",\"api_version\":1,\"status\":\"ok\"}"};
    const auto auth=q.headers.find("authorization");
    if(token.empty() || auth==q.headers.end() || auth->second!="Bearer "+token)return error(401,"Read the local discovery file and supply Authorization: Bearer <token>.");
    std::string command;
    if(q.path=="/api/status" || q.path=="/api/config" || q.path=="/api/competency" || q.path=="/api/action"){
        if(q.method=="GET")command=q.path=="/api/status"?"GET_STATUS":q.path=="/api/config"?"GET_CONFIG":q.path=="/api/competency"?"GET_COMPETENCY":"SESSION_ACTION_STATUS";
        else if(q.path!="/api/config" || q.method!="PATCH")return error(405,"Use GET for this endpoint, or PATCH /api/config to adjust a parameter.");
    }else if(q.path=="/api/trainer/action"){
        if(q.method!="POST")return error(405,"Use POST /api/trainer/action with a JSON action.");
    }else return error(404,"Unknown endpoint; consult the included tutorial.");
    if(command.empty()){
        auto ct=q.headers.find("content-type");if(ct==q.headers.end() || ct->second!="application/json")return error(415,"Set Content-Type: application/json.");
        std::map<std::string,Value> body;if(!FlatJson(q.body).parse(body))return error(400,"Malformed JSON, duplicate key, nonfinite number, or unsupported nested value.");
        if(q.path=="/api/config"){
            if(body.size()!=2 || !body.count("key") || !body.count("value") || body["key"].number || !body["value"].number || !parameter(body["key"].text,body["value"].numeric))return error(400,"Expected a supported key and one finite numeric value in its allowed range; GET /api/config shows effective settings.");
            std::ostringstream value;value.precision(9);value<<body["value"].numeric;command="SET "+body["key"].text+" "+value.str();
        }else{
            if(body.size()!=1 || !body.count("action") || body["action"].number)return error(400,"Expected exactly one string action.");
            const std::string a=body["action"].text;
            if(a=="start")command="SESSION_START";else if(a=="save_and_stop")command="SESSION_STOP";
            else if(a=="save_snapshot")command="SAVE_SNAPSHOT";else if(a=="resume")command="RESUME_TRAINING";
            else return error(400,"Unknown action; permitted actions are start, save_and_stop, save_snapshot, resume.");
        }
    }
    Response r=control(command);if(r.status>=400)return error(r.status,r.body);return r;
}
}

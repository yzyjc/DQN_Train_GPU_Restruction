#pragma once
// Collaboration signature: GPT6-Astra. Portable presentation of saved course status.
#include "local_http_api.h"
#include <vector>
#include <cstdio>
namespace competency_view {
inline double number(const std::string& json,const char* key,double fallback=0){
    const std::string name="\""+std::string(key)+"\":";size_t p=json.find(name);if(p==json.npos)return fallback;
    char* end=nullptr;const char* start=json.c_str()+p+name.size();double value=strtod(start,&end);
    return end!=start && std::isfinite(value)?value:fallback;
}
inline std::vector<std::string> lines(const std::string& json){
    std::vector<std::string> out;if(json.empty()){out.push_back("Course: no saved status yet");return out;}
    char b[256];snprintf(b,sizeof(b),"Course Stage %.0f | stage games %.0f/%.0f | passes %.0f/%.0f",number(json,"stage"),number(json,"stage_games"),number(json,"curriculum_min_games"),number(json,"passes"),number(json,"curriculum_required_passes"));out.push_back(b);
    snprintf(b,sizeof(b),"Course %s | rehearsal %s | Minimax interference %.1f%%",number(json,"curriculum_enabled")?"ON":"OFF",number(json,"rehearsal_enabled")?"ON":"OFF",100*number(json,"minimax_ratio"));out.push_back(b);
    size_t pos=json.find("\"skills\":[");if(pos==json.npos)return out;
    double scores[2][3]={};unsigned count[2]={};double regressions=0,rehearsals=0,success=0,cooldown=0;
    for(int i=0;i<6;++i){size_t begin=json.find('{',pos),end=begin==json.npos?json.npos:json.find('}',begin);if(end==json.npos)break;
        std::string object=json.substr(begin,end-begin+1);std::map<std::string,tui_http::Value> values;
        if(!tui_http::FlatJson(object).parse(values))break;
        int side=(int)number(object,"side",-1),skill=(int)number(object,"skill",-1);if(side<0 || side>1 || skill<0 || skill>2)break;
        scores[side][skill]=100*number(object,"score");++count[side];
        regressions+=number(object,"regressions");rehearsals+=number(object,"rehearsals");success+=number(object,"successes");cooldown=std::fmax(cooldown,number(object,"cooldown_remaining"));pos=end+1;
    }
    for(int side=0;side<2;++side)if(count[side]==3){snprintf(b,sizeof(b),"%s EmptyWin %.1f%% | Win1 %.1f%% | Block1 %.1f%%",side?"White":"Black",scores[side][0],scores[side][1],scores[side][2]);out.push_back(b);}
    snprintf(b,sizeof(b),"Regressions %.0f | rehearsals %.0f | success %.1f%% | max cooldown %.0f games",regressions,rehearsals,rehearsals?success*100/rehearsals:0,cooldown);out.push_back(b);
    out.push_back("Saved training skill snapshot; not independent Minimax win rates.");return out;
}
}

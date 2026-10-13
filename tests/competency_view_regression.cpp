// Collaboration signature: GPT6-Astra.
#include "../DQN_Main_TUI_No_Terminal/competency_view.h"
#include <cstdio>
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    NEED(competency_view::lines("").at(0).find("no saved")!=std::string::npos);
    std::string j="{\"stage\":2,\"stage_games\":50,\"curriculum_min_games\":100,\"passes\":1,\"curriculum_required_passes\":3,\"curriculum_enabled\":1,\"rehearsal_enabled\":0,\"minimax_ratio\":0.5,\"skills\":[";
    for(int side=0;side<2;++side)for(int k=0;k<3;++k){if(side||k)j+=",";j+="{\"side\":"+std::to_string(side)+",\"skill\":"+std::to_string(k)+",\"score\":"+(side?"0.6":"0.9")+",\"regressions\":1,\"rehearsals\":2,\"successes\":1,\"cooldown_remaining\":100}";}j+="]}";
    auto lines=competency_view::lines(j);NEED(lines.size()==6);NEED(lines[0].find("Stage 2")!=std::string::npos);
    NEED(lines[2].find("Black EmptyWin 90.0%")!=std::string::npos);NEED(lines[3].find("White EmptyWin 60.0%")!=std::string::npos);
    NEED(lines[4].find("success 50.0%")!=std::string::npos);puts("PASS course UI score split and saved-snapshot labels");return 0;
}

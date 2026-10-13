// Collaboration signature: GPT6-Astra.
#include "competency_state.h"
#include <cstdio>
#define NEED(x) do{if(!(x)){printf("FAIL %d\n",__LINE__);return 1;}}while(0)
int main(){
    CompetencyConfig c;CompetencyState s;s.initialized=1;c.enabled=1;c.min_games=10;c.stage_cooldown=0;c.required_passes=3;
    SideSkillScores v[2];for(auto& q:v){for(int k=0;k<3;++k){q.score[k]=1;q.count[k]=10;}q.empty_steps=4;}
    s.games=s.stage_games=9;NEED(!competency_evaluate(s,c,v));NEED(!competency_evaluate(s,c,v));NEED(s.stage==0);
    s.games=s.stage_games=10;v[1].empty_steps=6;NEED(!competency_evaluate(s,c,v));NEED(s.passes==0);
    v[1].empty_steps=4;NEED(!competency_evaluate(s,c,v));NEED(!competency_evaluate(s,c,v));NEED(competency_evaluate(s,c,v));NEED(s.stage==1 && s.stage_games==0);
    c.rehearsal_enabled=1;auto& w=s.skills[1][2];NEED(w.learned);competency_observe(w,.7f,c);int side,skill;
    NEED(!competency_rehearsal_request(s,c,side,skill));competency_observe(w,.7f,c);
    NEED(competency_rehearsal_request(s,c,side,skill) && side==1 && skill==2);NEED(s.skills[0][2].current==1);
    competency_finish_rehearsal(s,c,side,skill,.98f,2);NEED(w.rehearsal_successes==1 && w.cooldown_until>s.games);
    competency_observe(w,.7f,c);competency_observe(w,.7f,c);NEED(!competency_rehearsal_request(s,c,side,skill));
    NEED(competency_valid(s));NEED(!competency_set(c,0,.5f));NEED(!competency_set(c,15,NAN));
    v[1].count[2]=0;NEED(!competency_evaluate(s,c,v));NEED(s.passes==0);
    puts("PASS competency gates, independent regression, cooldown, validation");return 0;
}

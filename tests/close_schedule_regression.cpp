#define _CRT_SECURE_NO_WARNINGS
#include "../DQN_Main_TUI_No_Terminal/close_schedule.h"
#include <cassert>
#include <cstdio>
int main() {
 int ep=0; assert(close_episode("12345",ep)&&ep==12345);
 assert(!close_episode("0",ep)); assert(!close_episode("-2",ep));
 assert(!close_episode("123x",ep)); assert(!close_episode("2147483648",ep));
 std::tm t={}; t.tm_year=126;t.tm_mon=8;t.tm_mday=21;t.tm_hour=6;t.tm_isdst=-1;
 auto now=std::mktime(&t); std::time_t deadline=0;
 assert(close_time("0730",now,deadline));
 auto v=*std::localtime(&deadline); assert(v.tm_mday==21&&v.tm_hour==7&&v.tm_min==30);
 assert(close_time("0730",deadline,deadline));
 now=deadline+1; assert(close_time("0730",now,deadline));
 v=*std::localtime(&deadline); assert(v.tm_mday==22&&v.tm_hour==7&&v.tm_min==30);
 assert(!close_time("2400",now,deadline));assert(!close_time("1260",now,deadline));
 assert(!close_time("730",now,deadline));assert(!close_time("07x0",now,deadline));
 t={};t.tm_year=126;t.tm_mon=11;t.tm_mday=31;t.tm_hour=23;t.tm_isdst=-1;
 assert(close_time("0000",std::mktime(&t),deadline));
 v=*std::localtime(&deadline);assert(v.tm_year==127&&v.tm_mon==0&&v.tm_mday==1);
 puts("PASS: episode validation, HHMM validation, today, exact minute, tomorrow, year rollover");
}

// GPT6-Astra: strict close command arguments and local wall-clock deadline.
#pragma once
#include <ctime>
#include <climits>
#include <cstring>
inline bool close_episode(const char* text, int& value) {
    if (!text || !*text) return false;
    int n = 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9' || n > (INT_MAX - (*text-'0')) / 10) return false;
        n = n * 10 + (*text-'0');
    }
    if (n <= 0) return false;
    value = n; return true;
}
inline bool close_time(const char* text, std::time_t now, std::time_t& deadline) {
    if (!text || std::strlen(text) != 4) return false;
    for (int i=0; i<4; ++i) if (text[i]<'0' || text[i]>'9') return false;
    int h=(text[0]-'0')*10+text[1]-'0', m=(text[2]-'0')*10+text[3]-'0';
    if (h>23 || m>59) return false;
    std::tm* local = std::localtime(&now);
    if (!local) return false;
    std::tm target = *local;
    target.tm_hour=h; target.tm_min=m; target.tm_sec=0; target.tm_isdst=-1;
    std::time_t result=std::mktime(&target);
    if (result == (std::time_t)-1) return false;
    if (result < now) {
        target = *local;
        ++target.tm_mday;
        target.tm_hour=h; target.tm_min=m; target.tm_sec=0; target.tm_isdst=-1;
        result=std::mktime(&target);
    }
    if (result == (std::time_t)-1) return false;
    deadline=result; return true;
}

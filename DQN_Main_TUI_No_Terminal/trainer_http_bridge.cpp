// Collaboration signature: GPT6-Astra. Own connection, never the UI pipe handle.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "trainer_http_bridge.h"
namespace tui_http {
struct PipeHandle { HANDLE h=INVALID_HANDLE_VALUE; ~PipeHandle(){if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);} };
static bool transfer(HANDLE pipe, bool write, char* bytes, DWORD size, DWORD& done, DWORD timeout){
    OVERLAPPED o={};o.hEvent=CreateEventW(nullptr,TRUE,FALSE,nullptr);if(!o.hEvent)return false;
    BOOL ok=write?WriteFile(pipe,bytes,size,&done,&o):ReadFile(pipe,bytes,size,&done,&o);
    if(!ok && GetLastError()==ERROR_IO_PENDING){
        if(WaitForSingleObject(o.hEvent,timeout)==WAIT_OBJECT_0)ok=GetOverlappedResult(pipe,&o,&done,FALSE);
        else {CancelIoEx(pipe,&o);GetOverlappedResult(pipe,&o,&done,TRUE);ok=FALSE;}
    }
    CloseHandle(o.hEvent);return ok!=FALSE;
}
Response trainer_command(const std::string& command){
    PipeHandle p;p.h=CreateFileW(L"\\\\.\\pipe\\dqn_control_pipe",GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_EXISTING,FILE_FLAG_OVERLAPPED,nullptr);
    if(p.h==INVALID_HANDLE_VALUE)return command=="GET_STATUS"?Response{200,"{\"status\":\"ok\",\"trainer_connected\":false}"}:Response{503,"Trainer is not connected."};
    DWORD mode=PIPE_READMODE_MESSAGE;if(!SetNamedPipeHandleState(p.h,&mode,nullptr,nullptr))return {503,"Cannot configure Trainer connection."};
    DWORD done=0;std::string payload=command;
    if(!transfer(p.h,true,&payload[0],(DWORD)payload.size(),done,1000) || done!=payload.size())return {504,"Command delivery uncertain; query before retrying."};
    const ULONGLONG deadline=GetTickCount64()+3000;
    while(GetTickCount64()<deadline){char bytes[8192];done=0;
        const ULONGLONG now=GetTickCount64();if(now>=deadline)break;
        if(!transfer(p.h,false,bytes,sizeof(bytes),done,(DWORD)(deadline-now)))break;
        std::string reply(bytes,done);
        if(reply.find("\"status\":\"error\"")!=reply.npos)return {400,reply};
        if(reply.find("\"status\":\"ok\"")!=reply.npos)return {command.rfind("GET_",0)==0?200:202,reply};
    }
    return {504,"Trainer acknowledgement timed out; query actual state before retrying."};
}
}

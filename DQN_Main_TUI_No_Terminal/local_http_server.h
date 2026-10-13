#pragma once
#include "local_http_api.h"
#include <memory>
namespace tui_http {
class Server {
    struct Impl;std::unique_ptr<Impl> impl;
public:
    Server();~Server();Server(const Server&)=delete;Server& operator=(const Server&)=delete;
    bool start(const Controller& control,int preferred_port=17865);
    void stop();int port()const;std::string status()const;
};
}

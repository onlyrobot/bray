#include <string>

class Client
{
public:
    virtual void start(std::string = "") = 0;
    virtual void tick(const std::string &data) = 0;
    virtual void stop() = 0;
    virtual void step(const std::string &data) = 0;

    virtual ~Client() = default;
};

Client *create_client(
    std::string host, int port, void (*callback)(std::string data),
    std::string key = "", std::string token = "");
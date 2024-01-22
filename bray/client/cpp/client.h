#include <string>

class Client
{
public:
    virtual std::string start(std::string id = "") = 0;
    virtual std::string tick(const std::string &data) = 0;
    virtual std::string stop() = 0;

    virtual ~Client() = default;
};

Client *create_client(
    std::string host, int port, void (*callback)(std::string) = nullptr,
    std::string key = "", std::string token = "");
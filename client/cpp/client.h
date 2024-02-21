#include <memory>
#include <functional>
#include <string>

// 定义一个抽象基类Client，用于和Bray的Actor通信，支持同步和异步两种模式，
// 一个Client实例对应到游戏中的一个Agent，可以创建多个Client实例，
// 每个Client实例只能在一个线程中使用，多个线程请创建多个Client实例
class Client
{
public:
    // 开始一局游戏，默认ID为随机生成的UUID，
    // 多次开始游戏时，callback默认保持不变，也可以覆盖
    virtual std::string start(
        std::string game = "", std::string agent = "",
        std::function<void(std::string)> callback = nullptr) = 0;
    // 执行一次tick，如果是同步模式，返回值为tick的返回值，如果是异步模式，返回空字符串
    virtual std::string tick(const std::string &data) = 0;
    // 结束一局游戏，结束后可以重新开始新的一局
    virtual std::string stop() = 0;

    // 虚析构函数，保证派生类能够正确析构
    virtual ~Client() = default;
};

// 创建一个Client实例，host和port是Bray Actor的IP和端口，
// callback为异步模式下tick的回调函数，为空时表示同步模式，
// key和token被用来鉴权和调用量统计，训练时可以忽略，部署时从控制台获取
std::shared_ptr<Client> create_client(
    std::string host, int port,
    std::function<void(std::string)> callback = nullptr,
    std::string key = "", std::string token = "");
#include "fake_gamecore.h"
using namespace std;

#define ACTOR_IP "127.0.0.1"
#define ACTOR_PORT 8000
char buffer[1024 * 5];
const string key("fake_key");
const string token("fake_token");

// windows平台通过封装htonll实现htobe64
#ifdef _WIN32
uint64_t htobe64(uint64_t value) {
    return htonll(value);        // 转换为网络字节序（大端字节序）
    //return _byteswap_uint64(value);  // 或者翻转字节序
}

// windows平台通过封装ntohll实现be64toh
uint64_t be64toh(uint64_t value) {
    return ntohll(value);  // 转换为主机字节序
}
#endif

string actor_step(int sock, string game_id, string msg, string kind)
{
    // 构造请求数据包
    int64_t size = htobe64(game_id.size());
    memcpy(buffer, &size, sizeof(int64_t));
    int offset = sizeof(int64_t);
    size = htobe64(kind.size());
    memcpy(buffer + offset, &size, sizeof(int64_t));
    offset += sizeof(int64_t);
    size = htobe64(key.size());
    memcpy(buffer + offset, &size, sizeof(int64_t));
    offset += sizeof(int64_t);
    size = htobe64(token.size());
    memcpy(buffer + offset, &size, sizeof(int64_t));
    offset += sizeof(int64_t);
    size = htobe64(msg.size());
    memcpy(buffer + offset, &size, sizeof(int64_t));
    offset += sizeof(int64_t);
    int64_t time = 0;
    size = htobe64(time);
    memcpy(buffer + offset, &size, sizeof(int64_t));
    offset += sizeof(int64_t);
    memcpy(buffer + offset, game_id.c_str(), game_id.size());
    offset += game_id.size();
    memcpy(buffer + offset, kind.c_str(), kind.size());
    offset += kind.size();
    memcpy(buffer + offset, key.c_str(), key.size());
    offset += key.size();
    memcpy(buffer + offset, token.c_str(), token.size());
    offset += token.size();
    send(sock, buffer, offset, 0);
    send(sock, msg.c_str(), strlen(msg.c_str()), 0);

    // buffer 初始化
    memset(buffer, 0, sizeof(buffer));
    // 接收服务器的响应
    recv(sock, buffer, sizeof(int64_t) * 3, 0);
    memcpy(&size, buffer, sizeof(int64_t));
    int64_t game_id_size_recv = be64toh(size);
    assert(be64toh(size) == game_id.size());
    memcpy(&size, buffer + sizeof(int64_t), sizeof(int64_t));
    int64_t body_size_recv = be64toh(size);
    memcpy(&size, buffer + sizeof(int64_t) * 2, sizeof(int64_t));
    assert(be64toh(size) == time);
    string game_id_recv(game_id_size_recv, ' ');
    recv(sock, &game_id_recv[0], game_id_size_recv, 0);
    string msg_recv(body_size_recv, ' ');
    recv(sock, &msg_recv[0], body_size_recv, 0);
    return move(msg_recv);
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
    // windows平台需要初始化Winsock库
    WSADATA wsaData;
    int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (err != 0) {
        std::cerr << "WSAStartup failed with error: " << err << std::endl;
        return 1;
    }
#endif

    cout << "fake gamecore start" << endl;
    int sock;
    struct sockaddr_in server_address;
    // 创建一个TCP/IP套接字
    sock = socket(AF_INET, SOCK_STREAM, 0);

    // 设置服务器地址和端口号
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(ACTOR_PORT);
#ifdef _WIN32
    // 高版本c++版本ip转换要使用inet_pton函数
    inet_pton(AF_INET, ACTOR_IP, &server_address.sin_addr);
#else
    server_address.sin_addr.s_addr = inet_addr(ACTOR_IP);
#endif


    // 连接服务器
    connect(sock, (struct sockaddr *)&server_address, sizeof(server_address));

    int64_t t = static_cast<int64_t>(time(nullptr));
    string game_id = "game_id_" + to_string(t);
    string res = actor_step(sock, game_id, "start game", "start");
    cout << res << endl;
    for (int i = 0; i < 100; i++)
    {
        res = actor_step(sock, game_id, "fake game state", "tick");
        cout << res << endl;
    }
    res = actor_step(sock, game_id, "stop game", "stop");
    cout << res << endl;
    // 关闭套接字
#ifdef _WIN32
    // windows关闭socket
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif
    cout << "fake gamecore stopped" << endl;
    return 0;
}
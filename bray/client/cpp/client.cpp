#include "client.h"
#include <boost/asio.hpp>
#include <boost/endian.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <thread>
#include <iostream>
#include <functional>

using error_code = boost::system::error_code;
using namespace boost::asio;

class ClientImpl : public Client
{
public:
    ClientImpl(const std::string &host, int port,
               void (*callback)(std::string data));
    ~ClientImpl() = default;

    void start(std::string) override;
    void tick(const std::string &data) override;
    void stop() override;
    void step(const std::string &data) override;

private:
    void _connect_to_server();
    void _build_send_buffer(
        std::string kind, const std::string &data);
    void _handle_write(const error_code &error, size_t size);
    size_t _read_until(const error_code &error, size_t size);
    void _handle_read(const error_code &error, size_t size);

    std::function<void(const error_code &, size_t)>
        handle_write_, handle_read_;
    size_t pending_read_size_ = 0;
    std::function<size_t(const error_code &, size_t)> read_until_;
    std::string host_;
    int port_;

    ip::tcp::socket socket_;
    void (*callback)(std::string data);
    std::string game_id_;
    const std::string key_;
    const std::string token_;
    std::string recv_buffer_;
    std::string send_buffer_;
};

Client *create_client(
    std::string host, int port, void (*callback)(std::string data),
    std::string key, std::string token)
{
    return new ClientImpl(host, port, callback);
}

boost::uuids::random_generator uuid_generator;
io_context ioc;
std::thread io_thread([]
                      {
    io_context::work work(ioc);
    ioc.run(); });

ClientImpl::ClientImpl(const std::string &host, int port,
                       void (*callback)(std::string data))
    : host_(host), port_(port), socket_(ioc), callback(callback)
{
    recv_buffer_.resize(1024 * 2);
    _connect_to_server();
    handle_write_ = std::bind(
        &ClientImpl::_handle_write, this,
        std::placeholders::_1, std::placeholders::_2);
    handle_read_ = std::bind(
        &ClientImpl::_handle_read, this,
        std::placeholders::_1, std::placeholders::_2);
    read_until_ = std::bind(
        &ClientImpl::_read_until, this,
        std::placeholders::_1, std::placeholders::_2);
}

void ClientImpl::_connect_to_server()
{
    ip::tcp::resolver resolver(ioc);
    ip::tcp::resolver::query query(host_, std::to_string(port_));
    auto endpoints = resolver.resolve(query);
    if (!socket_.is_open())
        socket_ = ip::tcp::socket(ioc);
    connect(socket_, endpoints);
}

void ClientImpl::_build_send_buffer(std::string kind, const std::string &data)
{
    int64_t offset = 0;
    char head_buffer[1024 * 2];

    using namespace boost::endian;
    // 报头第一组 gameid长度的b64编码
    int64_t head_size = native_to_big(game_id_.size());
    memcpy(head_buffer, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    // 报头第二组 kind长度的b64编码
    head_size = native_to_big(kind.size());
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    // 报头第三组 key长度的b64编码
    head_size = native_to_big(key_.size());
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    // 报头第四组 token长度的b64编码
    head_size = native_to_big(token_.size());
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    // 报头第五组 正文内容长度的b64编码
    head_size = native_to_big(data.size());
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    // 报头第六组 时间戳长度的b64编码
    int64_t time = 0;
    head_size = htobe64(sizeof(time));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    memcpy(head_buffer + offset, game_id_.c_str(), game_id_.size());
    offset += game_id_.size();
    memcpy(head_buffer + offset, kind.c_str(), kind.size());
    offset += kind.size();
    memcpy(head_buffer + offset, key_.c_str(), key_.size());
    offset += key_.size();
    memcpy(head_buffer + offset, token_.c_str(), token_.size());
    offset += token_.size();

    send_buffer_.resize(offset + data.size());
    memcpy(&send_buffer_[0], head_buffer, offset);
    memcpy(&send_buffer_[offset], data.c_str(), data.size());
}

void ClientImpl::start(std::string game_id)
{
    std::cout << "starting " << game_id << std::endl;
    if (!socket_.is_open())
        _connect_to_server();
    game_id_ = game_id;
    if (game_id_ == "")
    {
        auto uuid = uuid_generator();
        game_id_ = boost::uuids::to_string(uuid);
    }
    _build_send_buffer("start", "");
    write(socket_, buffer(send_buffer_));
    read(socket_, buffer(recv_buffer_), read_until_);
}

void ClientImpl::_handle_read(const error_code &error, size_t size)
{
    if (error)
    {
        std::cout << "read error: " << error << std::endl;
        return socket_.close();
    }
    using namespace boost::endian;
    int64_t game_id_size = 0;
    memcpy(&game_id_size, recv_buffer_.c_str(), sizeof(int64_t));
    game_id_size = big_to_native(game_id_size);
    int64_t body_size = 0;
    memcpy(&body_size, recv_buffer_.c_str() + sizeof(int64_t),
           sizeof(int64_t));
    body_size = big_to_native(body_size);
    size_t total_size = sizeof(int64_t) * 3 + game_id_size + body_size;
    if (total_size != size)
    {
        assert(total_size > recv_buffer_.size());
        recv_buffer_.resize(total_size * 2);
        async_read(socket_, buffer(recv_buffer_),
                   transfer_exactly(total_size - size), handle_read_);
        return;
    }
    char *data = &recv_buffer_[0] + sizeof(int64_t) * 3 + game_id_size;
    callback(std::string(data, body_size));
    if (--pending_read_size_ == 0)
        return;
    async_read(socket_, buffer(recv_buffer_), read_until_, handle_read_);
}

size_t ClientImpl::_read_until(const error_code &error, size_t size)
{
    if (error)
    {
        std::cout << "read error: " << error << std::endl;
        socket_.close();
        return 0;
    }
    size_t total_head_size = sizeof(int64_t) * 3;
    if (size < total_head_size)
    {
        return total_head_size - size;
    }
    using namespace boost::endian;
    int64_t game_id_size = 0;
    memcpy(&game_id_size, recv_buffer_.c_str(), sizeof(int64_t));
    game_id_size = big_to_native(game_id_size);
    int64_t body_size = 0;
    memcpy(&body_size, recv_buffer_.c_str() + sizeof(int64_t),
           sizeof(int64_t));
    body_size = big_to_native(body_size);
    int64_t time = 0;
    memcpy(&time, recv_buffer_.c_str() + sizeof(int64_t) * 2,
           sizeof(int64_t));
    time = big_to_native(time);
    size_t total_size = total_head_size + game_id_size + body_size;
    if (size > total_size)
    {
        std::cout << "read invalid size: " << size << std::endl;
        socket_.close();
        return 0;
    }
    if (recv_buffer_.size() < total_size)
        return 0; // 当前缓冲区不足以容纳一个完整的数据包
    return total_size - size;
}

void ClientImpl::_handle_write(const error_code &error, size_t size)
{
    if (error)
    {
        std::cout << "write error: " << error << std::endl;
        return socket_.close();
    }
    if (++pending_read_size_ != 1)
        return;
    async_read(socket_, buffer(recv_buffer_), read_until_, handle_read_);
}

void ClientImpl::tick(const std::string &data)
{
    _build_send_buffer("tick", data);
    async_write(socket_, buffer(send_buffer_), handle_write_);
}

void ClientImpl::stop()
{
    _build_send_buffer("stop", "");
    write(socket_, buffer(send_buffer_));
    if (pending_read_size_ == 0)
    {
        read(socket_, buffer(recv_buffer_), read_until_);
        return;
    }
    std::cout << "stop before tick finish" << std::endl;
    socket_.close();
}

void ClientImpl::step(const std::string &data)
{
    _build_send_buffer("auto", data);
    async_write(socket_, buffer(send_buffer_), handle_write_);
}
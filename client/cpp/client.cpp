#include "client.h"
#include <boost/asio.hpp>
#include <boost/endian.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <thread>
#include <iostream>
#include <atomic>

using error_code = boost::system::error_code;
using namespace boost::asio;

class ClientImpl : public Client, public std::enable_shared_from_this<ClientImpl>
{
public:
    ClientImpl(const std::string &host, int port,
               std::function<void(std::string)> callback);
    ~ClientImpl() = default;

    std::string start(std::string game, std::string agent,
        std::function<void(std::string)> callback) override;
    std::string tick(const std::string &data) override;
    std::string stop() override;

private:
    void _try_callback(std::string &&data);
    void _handle_write(const error_code &, size_t size);
    size_t _read_until(const error_code &, size_t size);
    void _handle_read(const error_code &, size_t size);
    void _connect_to_server();
    std::string _sync_request(
        std::string kind, const std::string &data);
    void _prepare_send_buffer(
        std::string kind, const std::string &data);
    size_t _parse_head(int64_t &session_size, int64_t &body_size);
    void _async_tick(const std::string &data);

    std::string host_;
    int port_;
    std::atomic<int64_t> pending_read_num_{0};

    ip::tcp::socket socket_;
    std::function<void(std::string)> cur_callback_;
    std::function<void(std::string)> callback_;
    std::string session_;
    const std::string key_;
    const std::string token_;
    std::string recv_buffer_;
    std::string send_buffer_, sending_buffer_;
    // 0: idle, 1: sending w/o pending, 2: sending with pending
    std::atomic<int64_t> sending_state_{0};
};

boost::uuids::random_generator gen_uuid;
io_context ioc;
std::thread io_thread;
std::atomic<bool> is_initialized{false};

std::shared_ptr<Client> create_client(
    std::string host, int port, std::function<void(std::string)> callback,
    std::string key, std::string token)
{
    if (!is_initialized.exchange(true)) {
        io_thread = std::thread([] { io_context::work work(ioc); ioc.run(); });
    }
    return std::make_shared<ClientImpl>(host, port, callback);
}

ClientImpl::ClientImpl(const std::string &host, int port,
                       std::function<void(std::string)> callback)
    : host_(host), port_(port), socket_(ioc), callback_(callback)
{
    recv_buffer_.resize(1024 * 2);
    try { _connect_to_server(); }
    catch (std::exception &e)
    {
        std::cout << "connect error: " << e.what() << std::endl;
    }
}

void ClientImpl::_try_callback(std::string &&data)
{
    try
    {
        return cur_callback_ ? cur_callback_(data) : void();
    }
    catch (std::exception &e)
    {
        std::cout << "callback error: " << e.what() << std::endl;
    }
}

void ClientImpl::_connect_to_server()
{
    ip::tcp::resolver resolver(ioc);
    ip::tcp::resolver::query query(host_, std::to_string(port_));
    auto endpoints = resolver.resolve(query);
    if (!socket_.is_open())
        socket_ = ip::tcp::socket(ioc);
    socket_.open(ip::tcp::v4());
    socket_.set_option(ip::tcp::no_delay(true));
    connect(socket_, endpoints);
}

std::string ClientImpl::_sync_request(std::string kind, 
                                    const std::string &data)
{
    auto interal = std::chrono::milliseconds(10);
    int remain_retry = 20;
    while ((sending_state_.load() || pending_read_num_.load()) &&
           remain_retry-- > 0 && socket_.is_open())
    {
        std::cout << "sync request pending" << std::endl;
        std::this_thread::sleep_for(interal);
    }
    if (sending_state_.load() || pending_read_num_.load())
    {
        std::cout << "sync before async done" << std::endl;
        return "";
    }
    _prepare_send_buffer(kind, data);
    write(socket_, buffer(send_buffer_));
    size_t n = read(socket_, buffer(recv_buffer_), std::bind(
        &ClientImpl::_read_until,
        shared_from_this(),
        std::placeholders::_1, std::placeholders::_2));
    int64_t session_size = 0, body_size = 0;
    size_t total_size = _parse_head(session_size, body_size);
    size_t offset = sizeof(int64_t) * 3 + session_.size();
    if (total_size == n)
        return std::string(&recv_buffer_[0] + offset, body_size);
    recv_buffer_.resize(total_size * 2);
    auto b = buffer(&recv_buffer_[0] + n, total_size - n);
    n = read(socket_, b, transfer_exactly(total_size - n));
    return std::string(&recv_buffer_[0] + offset, body_size);
}

size_t ClientImpl::_parse_head(int64_t &session_size, int64_t &body_size)
{
    using namespace boost::endian;
    memcpy(&session_size, recv_buffer_.c_str(), sizeof(int64_t));
    session_size = big_to_native(session_size);
    memcpy(&body_size, recv_buffer_.c_str() + sizeof(int64_t),
           sizeof(int64_t));
    body_size = big_to_native(body_size);
    return sizeof(int64_t) * 3 + session_size + body_size;
}

void ClientImpl::_prepare_send_buffer(std::string kind,
                                      const std::string &data)
{
    using namespace boost::endian;
    char head_buffer[1024 * 2];

    // 报头第一组 gameid长度的b64编码
    int64_t offset = 0;
    int64_t head_size = native_to_big(int64_t(session_.size()));
    memcpy(head_buffer, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);
    // 报头第二组 kind长度的b64编码
    head_size = native_to_big(int64_t(kind.size()));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);
    // 报头第三组 key长度的b64编码
    head_size = native_to_big(int64_t(key_.size()));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);
    // 报头第四组 token长度的b64编码
    head_size = native_to_big(int64_t(token_.size()));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);
    // 报头第五组 正文内容长度的b64编码
    head_size = native_to_big(int64_t(data.size()));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);
    // 报头第六组 时间戳长度的b64编码
    int64_t time = 0;
    head_size = native_to_big(int64_t(sizeof(time)));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    memcpy(head_buffer + offset, session_.c_str(), session_.size());
    offset += session_.size();
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

void ClientImpl::_handle_read(const error_code &error, size_t size)
{
    if (error || size < sizeof(int64_t) * 3)
    {
        std::cout << "read error: " << error << std::endl;
        _try_callback("");
        return socket_.close();
    }
    int64_t session_size = 0, body_size = 0;
    size_t total_size = _parse_head(session_size, body_size);
    if (session_size != session_.size())
    {
        std::cout << "read invalid session: " << session_size << std::endl;
        _try_callback("");
        return socket_.close();
    }
    auto handle_read = std::bind(
            &ClientImpl::_handle_read,
            shared_from_this(),
            std::placeholders::_1, std::placeholders::_2);
    if (total_size != size)
    {
        recv_buffer_.resize(total_size * 2);
        auto b = buffer(&recv_buffer_[0] + size, total_size - size);
        return async_read(
            socket_, b,
            transfer_exactly(total_size - size),
            handle_read);
    }
    size_t offset = sizeof(int64_t) * 3 + session_.size();
    _try_callback(std::string(&recv_buffer_[0] + offset, body_size));
    if (pending_read_num_.fetch_sub(1) == 1)
        return;
    auto read_until = std::bind(
        &ClientImpl::_read_until,
        shared_from_this(),
        std::placeholders::_1, std::placeholders::_2);
    async_read(socket_, buffer(recv_buffer_), read_until, handle_read);
}

size_t ClientImpl::_read_until(const error_code &error, size_t size)
{
    if (error)
    {
        std::cout << "read error: " << error << std::endl;
        return 0;
    }
    size_t total_head_size = sizeof(int64_t) * 3;
    if (size < total_head_size)
    {
        return total_head_size - size;
    }
    int64_t session_size = 0, body_size = 0;
    size_t total_size = _parse_head(session_size, body_size);
    if (size > total_size)
    {
        std::cout << "read invalid size: " << size << std::endl;
        return 0;
    }
    if (recv_buffer_.size() < total_size)
        return 0; // 当前缓冲区不足以容纳一个完整的数据包
    return total_size - size;
}

void ClientImpl::_handle_write(const error_code &error, size_t size)
{
    if (error || size != sending_buffer_.size())
    {
        std::cout << "write error: " << error << std::endl;
        _try_callback("");
        return socket_.close();
    }
    auto handle_read = std::bind(
        &ClientImpl::_handle_read,
        shared_from_this(),
        std::placeholders::_1, std::placeholders::_2);
    auto read_until = std::bind(
        &ClientImpl::_read_until,
        shared_from_this(),
        std::placeholders::_1, std::placeholders::_2);
    if (pending_read_num_.fetch_add(1) == 0)
    {
        async_read(socket_, buffer(recv_buffer_),
                   read_until, handle_read);
    }
    int64_t before_state = 1;
    if (sending_state_.compare_exchange_strong(before_state, 0))
        return;
    if (before_state != 2)
        return;
    send_buffer_.swap(sending_buffer_);
    sending_state_.store(1);
    auto handle_write = std::bind(
        &ClientImpl::_handle_write,
        shared_from_this(),
        std::placeholders::_1, std::placeholders::_2);
    async_write(socket_, buffer(sending_buffer_), handle_write);
}

void ClientImpl::_async_tick(const std::string &data)
{
    if (sending_state_.load() == 2 || pending_read_num_.load() > 1)
    // if (sending_state_.load() || pending_read_num_.load())
    {
        std::cout << "tick before callback done" << std::endl;
        return _try_callback("");
    }
    _prepare_send_buffer("tick", data);
    if (sending_state_.fetch_add(1) == 1) // sending w/o pending
        return;
    sending_buffer_.swap(send_buffer_);
    auto handle_write = std::bind(
        &ClientImpl::_handle_write,
        shared_from_this(),
        std::placeholders::_1, std::placeholders::_2);
    async_write(socket_, buffer(sending_buffer_), handle_write);
}

std::string ClientImpl::start(std::string game, std::string agent,
    std::function<void(std::string)> callback)
{
    if (game == "")
    {
        game = boost::uuids::to_string(gen_uuid());
    }
    session_ = agent == "" ? game : game + "-agent-" + agent;
    cur_callback_ = callback_ = callback ? callback : callback_;
    std::cout << "starting " << session_ << std::endl;
    try
    {
        return _sync_request("start", "");
    }
    catch (std::exception &e)
    {
        std::cout << "start error: " << e.what() << std::endl;
        socket_.close();
    }
    std::cout << "recovering " << session_ << std::endl;
    try
    {
        _connect_to_server();
        pending_read_num_ = sending_state_ = 0;
        return _sync_request("start", "");
    }
    catch (std::exception &e)
    {
        std::cout << "start error: " << e.what() << std::endl;
        return "";
    }
}

std::string ClientImpl::tick(const std::string &data)
{
    if (cur_callback_) // async mode, return immediately
    {
        _async_tick(data);
        return "";
    }
    try
    {
        return _sync_request("tick", data);
    }
    catch (std::exception &e)
    {
        std::cout << "tick error: " << e.what() << std::endl;
        return "";
    }
}

std::string ClientImpl::stop()
{
    try
    {
        cur_callback_ = nullptr; // stop callback
        return _sync_request("stop", "");
    }
    catch (std::exception &e)
    {
        std::cout << "stop error: " << e.what() << std::endl;
        return "";
    }
}
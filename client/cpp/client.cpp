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

class ClientImpl : public Client
{
public:
    ClientImpl(const std::string &host, int port,
               std::function<void(std::string)> callback);
    ~ClientImpl() = default;

    std::string start(std::string id) override;
    std::string tick(const std::string &data) override;
    std::string stop() override;

private:
    void _try_callback(std::string&& data);
    void _handle_write(const error_code &, size_t size);
    size_t _read_until(const error_code &, size_t size);
    void _handle_read(const error_code &, size_t size);
    void _connect_to_server();
    std::string _sync_request(const std::string &send_buf);
    void _prepare_send_buffer(
        std::string kind, const std::string &data);
    size_t _parse_head(int64_t &id_size, int64_t &body_size);
    void _async_tick(const std::string &data);

    std::function<void(const error_code &, size_t)>
        handle_read_, handle_write_;
    std::function<size_t(const error_code &, size_t)> read_until_;
    std::string host_;
    int port_;
    int64_t pending_reads_ = 0;

    ip::tcp::socket socket_;
    std::function<void(std::string)> callback_;
    std::string id_;
    const std::string key_;
    const std::string token_;
    std::string recv_buffer_;
    std::string send_buffer_, sending_buffer_;
    // 0: idle, 1: sending w/o pending, 2: sending with pending
    std::atomic<int64_t> sending_state_{0};
};

Client *create_client(
    std::string host, int port, std::function<void(std::string)> callback,
    std::string key, std::string token)
{
    return new ClientImpl(host, port, callback);
}

boost::uuids::random_generator gen_uuid;
io_context ioc;
std::thread io_thread([]
                      {
    io_context::work work(ioc);
    ioc.run(); });

ClientImpl::ClientImpl(const std::string &host, int port,
                       std::function<void(std::string)> callback)
    : host_(host), port_(port), socket_(ioc), callback_(callback)
{
    handle_read_ = std::bind(
        &ClientImpl::_handle_read,
        this,
        std::placeholders::_1, std::placeholders::_2);
    handle_write_ = std::bind(
        &ClientImpl::_handle_write,
        this,
        std::placeholders::_1, std::placeholders::_2);
    read_until_ = std::bind(
        &ClientImpl::_read_until,
        this,
        std::placeholders::_1, std::placeholders::_2);
    recv_buffer_.resize(1024 * 2);
    try
    {
        _connect_to_server();
    }
    catch (std::exception &e)
    {
        std::cout << "connect error: " << e.what() << std::endl;
    }
}

void ClientImpl::_try_callback(std::string&& data)
{
    try
    {
        callback_(data);
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
    connect(socket_, endpoints);
}

std::string ClientImpl::_sync_request(const std::string &send_buf)
{
    write(socket_, buffer(send_buf));
    size_t n = read(socket_, buffer(recv_buffer_), read_until_);
    int64_t id_size = 0, body_size = 0;
    size_t total_size = _parse_head(id_size, body_size);
    size_t offset = sizeof(int64_t) * 3 + id_.size();
    if (total_size == n)
        return std::string(&recv_buffer_[0] + offset, body_size);
    recv_buffer_.resize(total_size * 2);
    auto b = buffer(&recv_buffer_[0] + n, total_size - n);
    n = read(socket_, b, transfer_exactly(total_size - n));
    return std::string(&recv_buffer_[0] + offset, body_size);
}

size_t ClientImpl::_parse_head(int64_t &id_size, int64_t &body_size)
{
    using namespace boost::endian;
    memcpy(&id_size, recv_buffer_.c_str(), sizeof(int64_t));
    id_size = big_to_native(id_size);
    memcpy(&body_size, recv_buffer_.c_str() + sizeof(int64_t),
           sizeof(int64_t));
    body_size = big_to_native(body_size);
    return sizeof(int64_t) * 3 + id_size + body_size;
}

void ClientImpl::_prepare_send_buffer(std::string kind,
                                      const std::string &data)
{
    using namespace boost::endian;
    char head_buffer[1024 * 2];

    // 报头第一组 gameid长度的b64编码
    int64_t offset = 0;
    int64_t head_size = native_to_big(id_.size());
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
    head_size = native_to_big(sizeof(time));
    memcpy(head_buffer + offset, &head_size, sizeof(int64_t));
    offset += sizeof(int64_t);

    memcpy(head_buffer + offset, id_.c_str(), id_.size());
    offset += id_.size();
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
    int64_t id_size = 0, body_size = 0;
    size_t total_size = _parse_head(id_size, body_size);
    if (id_size != id_.size())
    {
        std::cout << "read invalid id size: " << id_size << std::endl;
        _try_callback("");
        return socket_.close();
    }
    if (total_size != size)
    {
        recv_buffer_.resize(total_size * 2);
        auto b = buffer(&recv_buffer_[0] + size, total_size - size);
        return async_read(
            socket_, b,
            transfer_exactly(total_size - size),
            handle_read_);
    }
    size_t offset = sizeof(int64_t) * 3 + id_.size();
    _try_callback(std::string(&recv_buffer_[0] + offset, body_size));
    if (--pending_reads_ == 0)
        return;
    async_read(socket_, buffer(recv_buffer_), read_until_, handle_read_);
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
    int64_t id_size = 0, body_size = 0;
    size_t total_size = _parse_head(id_size, body_size);
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
    if (pending_reads_++ == 0)
    {
        async_read(socket_, buffer(recv_buffer_),
                   read_until_, handle_read_);
    }
    int64_t before_state = 1;
    if (sending_state_.compare_exchange_strong(before_state, 0))
        return;
    if (before_state != 2)
        return;
    send_buffer_.swap(sending_buffer_);
    sending_state_.store(1);
    async_write(socket_, buffer(sending_buffer_), handle_write_);
}

void ClientImpl::_async_tick(const std::string &data)
{
    if (sending_state_.load() == 2)
    {
        std::cout << "tick before callback done" << std::endl;
        return _try_callback("");
    }
    _prepare_send_buffer("tick", data);
    if (sending_state_.fetch_add(1) == 1) // sending w/o pending
        return;
    sending_buffer_.swap(send_buffer_);
    async_write(socket_, buffer(sending_buffer_), handle_write_);
}

std::string ClientImpl::start(std::string id)
{
    id_ = id == "" ? boost::uuids::to_string(gen_uuid()) : id;
    std::cout << "starting " << id_ << std::endl;
    if (sending_state_.load() != 0 || pending_reads_ != 0)
    {
        std::cout << "start before callback done" << std::endl;
        socket_.close();
        _connect_to_server();
        sending_state_ = pending_reads_ = 0;
    }
    _prepare_send_buffer("start", "");
    try
    {
        return _sync_request(send_buffer_);
    }
    catch (std::exception &e)
    {
        std::cout << "start error: " << e.what() << std::endl;
        socket_.close();
    }
    std::cout << "recovering " << id_ << std::endl;
    try
    {
        _connect_to_server();
        return _sync_request(send_buffer_);
    }
    catch (std::exception &e)
    {
        std::cout << "start error: " << e.what() << std::endl;
        return "";
    }
}

std::string ClientImpl::tick(const std::string &data)
{
    if (callback_) // async mode, callback in io thread
    {
        _async_tick(data);
        return "";
    }
    _prepare_send_buffer("tick", data);
    try
    {
        return _sync_request(send_buffer_);
    }
    catch (std::exception &e)
    {
        std::cout << "tick error: " << e.what() << std::endl;
        return "";
    }
}

std::string ClientImpl::stop()
{
    if (sending_state_.load() != 0 || pending_reads_ != 0)
    {
        std::cout << "stop before callback done" << std::endl;
        return "";
    }
    _prepare_send_buffer("stop", "");
    try
    {
        return _sync_request(send_buffer_);
    }
    catch (std::exception &e)
    {
        std::cout << "stop error: " << e.what() << std::endl;
        return "";
    }
}
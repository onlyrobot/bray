#include "client.h"
#include <iostream>
#include <thread>

#define HOST "0.0.0.0"
#define PORT 8000

void callback(std::string data)
{
    std::cout << "callback " << data << std::endl;
}

void test_client(int i)
{
    std::cout << "test_client " << i << std::endl;
    auto client = create_client(HOST, PORT, callback);
    for (int j = 0; j < 10; j++)
    {
        client->start();
        for (int k = 0; k < 10; k++)
        {
            client->tick("hello world");
            client->tick("world------------------------------");
        }
        client->stop();
    }
    client = create_client(HOST, PORT);
    client->start();
    for (int j = 0; j < 10; j++)
    {
        std::cout << client->tick("hello") << std::endl;
        std::cout << client->tick("world") << std::endl;
    }
    std::cout << client->stop() << std::endl;
}

int main()
{
    size_t parallelism = 10;
    std::thread threads[parallelism];

    for (int i = 0; i < parallelism; i++)
    {
        threads[i] = std::thread(test_client, i);
    }

    for (int i = 0; i < parallelism; i++)
    {
        threads[i].join();
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "All client finished!" << std::endl;

    return 0;
}
#pragma once
#include<iostream>
#include<string>
#include <cassert>

#ifdef _WIN32
#include <Winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#pragma comment(lib, "wsock32.lib")
#pragma comment(lib, "Ws2_32.lib")
#include <intrin.h>
#elif __linux__
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#endif

#ifdef _WIN32
uint64_t htobe64(uint64_t value);
uint64_t be64toh(uint64_t value);
#endif


#ifndef COREAI3D_MAIN_HPP
#define COREAI3D_MAIN_HPP

// Platform-specific includes and macros
#ifdef _WIN32
// Windows-specific macros to prevent WinSock.h conflicts and optimize windows.h
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

// Standard C++ Library Includes
#include <thread>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <memory>
#include <functional>
#include <chrono>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <future>
#include <optional>
#include <filesystem>
#include <unordered_map>
#include <map>
#include <iomanip>

// External library headers
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <mysqlx/xdevapi.h>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>

// Namespace aliases
namespace net = boost::asio;
namespace beast = boost::beast;
namespace po = boost::program_options;

// Project-specific headers
#include "Core.hpp"
#include "Train.hpp"
#include "Database.hpp"
#include "Language.hpp"

#endif // COOREAI3D_MAIN_HPP
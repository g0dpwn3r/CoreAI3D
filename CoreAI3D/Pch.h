#ifndef MAIN_H
#define MAIN_H
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/signal_set.hpp>
#include <thread> // For running the network server in a separate thread
#include <iterator> 
#include <string>
#include <memory>       // Crucial: Add this line for std::enable_shared_from_this and std::shared_ptr
#include <functional>   // For std::function
#include <chrono> 
#include <memory>
#include <iostream>
#include <unordered_set>
#include <algorithm> 
#include <fstream>
#include <sstream>
#include <regex>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>
#include <future>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <mysqlx/xdevapi.h>
#include <filesystem>
#include <unordered_map>
#include <map>
#include <chrono> // Added for std::chrono::milliseconds
#include <thread> // Added for std::this_thread::sleep_for
#include <iomanip> // Added for std::fixed, std::setprecision, std::setw

#include <argparse/argparse.hpp>
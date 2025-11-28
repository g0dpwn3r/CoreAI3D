// This file will be used to generate the Precompiled Header.
// It should contain stable, frequently included headers.

// System/Standard Library Headers
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>

// Boost Headers (The major time consumers)
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

// Third-Party Headers
#include <nlohmann/json.hpp>

// Local Core Headers
// You might include your main common header here:
#include "CoreAI3DCommon.hpp"
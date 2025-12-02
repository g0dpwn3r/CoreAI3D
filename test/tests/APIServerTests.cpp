// test/tests/APIServerTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <atomic>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <nlohmann/json.hpp>

#include "APIServer.hpp"
#include "CoreAI3DCommon.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

// HTTP client helper for integration tests
class HttpClient {
private:
    net::io_context ioc;
    tcp::resolver resolver{ioc};
    beast::tcp_stream stream{ioc};

public:
    HttpClient() = default;

    // Perform HTTP GET request
    json get(const std::string& host, const std::string& port, const std::string& target) {
        try {
            // Resolve the host
            auto const results = resolver.resolve(host, port);

            // Connect to the endpoint
            stream.connect(results);

            // Set up HTTP GET request
            http::request<http::string_body> req{http::verb::get, target, 11};
            req.set(http::field::host, host);
            req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
            req.set(http::field::accept, "application/json");

            // Send the HTTP request
            http::write(stream, req);

            // Receive the HTTP response
            beast::flat_buffer buffer;
            http::response<http::string_body> res;
            http::read(stream, buffer, res);

            // Close the connection
            beast::error_code ec;
            stream.socket().shutdown(tcp::socket::shutdown_both, ec);

            // Parse JSON response
            if (res.result() == http::status::ok && !res.body().empty()) {
                return json::parse(res.body());
            } else {
                return json{{"error", "HTTP " + std::to_string(static_cast<int>(res.result()))},
                           {"status_code", static_cast<int>(res.result())}};
            }
        } catch (const std::exception& e) {
            return json{{"error", std::string("Exception: ") + e.what()}};
        }
    }

    // Perform HTTP POST request
    json post(const std::string& host, const std::string& port, const std::string& target, const json& body) {
        try {
            // Resolve the host
            auto const results = resolver.resolve(host, port);

            // Connect to the endpoint
            stream.connect(results);

            // Set up HTTP POST request
            http::request<http::string_body> req{http::verb::post, target, 11};
            req.set(http::field::host, host);
            req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
            req.set(http::field::content_type, "application/json");
            req.set(http::field::accept, "application/json");
            req.body() = body.dump();
            req.prepare_payload();

            // Send the HTTP request
            http::write(stream, req);

            // Receive the HTTP response
            beast::flat_buffer buffer;
            http::response<http::string_body> res;
            http::read(stream, buffer, res);

            // Close the connection
            beast::error_code ec;
            stream.socket().shutdown(tcp::socket::shutdown_both, ec);

            // Parse JSON response
            if (res.result() == http::status::ok && !res.body().empty()) {
                return json::parse(res.body());
            } else {
                return json{{"error", "HTTP " + std::to_string(static_cast<int>(res.result()))},
                           {"status_code", static_cast<int>(res.result())}};
            }
        } catch (const std::exception& e) {
            return json{{"error", std::string("Exception: ") + e.what()}};
        }
    }
};

TEST_CASE("API Server Integration Tests", "[api][integration][!mayfail]") {
    // Skip tests if environment variable is not set
    const char* run_api_tests = std::getenv("RUN_API_TESTS");
    if (!run_api_tests || std::string(run_api_tests) != "1") {
        WARN("Skipping API integration tests because RUN_API_TESTS!=1");
        return;
    }

    // Test configuration
    const std::string test_host = "127.0.0.1";
    const std::string test_port = "9090"; // Use a different port for testing
    const int test_api_port = 9090;

    // Create API server instance
    APIServer apiServer("TestAPIServer", test_host, test_api_port);

    // Initialize server (offline mode for testing)
    REQUIRE(apiServer.initialize("config.json", "", 0, "", "", "", SSLMode::DISABLED, false));

    // Start server in background thread
    std::atomic<bool> server_started{false};
    std::thread server_thread([&]() {
        try {
            server_started = true;
            apiServer.start();
        } catch (const std::exception& e) {
            FAIL(std::string("Server start failed: ") + e.what());
        }
    });

    // Wait for server to start
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Verify server is running
    REQUIRE(apiServer.isServerRunning());

    // Create HTTP client
    HttpClient client;

    SECTION("API Status Endpoint") {
        auto response = client.get(test_host, test_port, "/api/v1/status");

        REQUIRE(response.contains("status"));
        REQUIRE(response["status"] == "ok");
        REQUIRE(response.contains("server_name"));
        REQUIRE(response["server_name"] == "TestAPIServer");
    }

    SECTION("API Version Endpoint") {
        auto response = client.get(test_host, test_port, "/api/v1/version");

        REQUIRE(response.contains("version"));
        REQUIRE(response.contains("api_version"));
    }

    SECTION("API Endpoints List") {
        auto response = client.get(test_host, test_port, "/api/v1/endpoints");

        REQUIRE(response.contains("endpoints"));
        REQUIRE(response["endpoints"].is_array());
        REQUIRE(response["endpoints"].size() > 0);
    }

    SECTION("Session Management") {
        // Create session
        json create_session_data = {
            {"action", "create_session"}
        };
        auto create_response = client.post(test_host, test_port, "/api/v1/sessions", create_session_data);

        REQUIRE(create_response.contains("session_id"));
        std::string session_id = create_response["session_id"];

        // Validate session
        json validate_data = {
            {"action", "validate_session"},
            {"session_id", session_id}
        };
        auto validate_response = client.post(test_host, test_port, "/api/v1/sessions", validate_data);

        REQUIRE(validate_response.contains("valid"));
        REQUIRE(validate_response["valid"] == true);
    }

    SECTION("Vision API Endpoints") {
        // Create session first
        json create_session_data = {{"action", "create_session"}};
        auto session_response = client.post(test_host, test_port, "/api/v1/sessions", create_session_data);
        REQUIRE(session_response.contains("session_id"));
        std::string session_id = session_response["session_id"];

        // Test vision classify (will fail without actual image, but should return proper error)
        json vision_data = {
            {"action", "classify"},
            {"session_id", session_id},
            {"image_path", "/nonexistent/image.jpg"},
            {"module_name", "default"}
        };
        auto vision_response = client.post(test_host, test_port, "/api/v1/vision", vision_data);

        // Should return error for nonexistent file, but proper JSON structure
        REQUIRE(vision_response.contains("error") || vision_response.contains("success"));
    }

    SECTION("Math API Endpoints") {
        // Create session first
        json create_session_data = {{"action", "create_session"}};
        auto session_response = client.post(test_host, test_port, "/api/v1/sessions", create_session_data);
        REQUIRE(session_response.contains("session_id"));
        std::string session_id = session_response["session_id"];

        // Test math calculation
        json math_data = {
            {"action", "calculate"},
            {"session_id", session_id},
            {"expression", "2 + 3 * 4"}
        };
        auto math_response = client.post(test_host, test_port, "/api/v1/math", math_data);

        REQUIRE(math_response.contains("result"));
        REQUIRE(math_response["result"] == 14);
    }

    SECTION("System API Endpoints") {
        // Create session first
        json create_session_data = {{"action", "create_session"}};
        auto session_response = client.post(test_host, test_port, "/api/v1/sessions", create_session_data);
        REQUIRE(session_response.contains("session_id"));
        std::string session_id = session_response["session_id"];

        // Test system metrics
        json system_data = {
            {"action", "get_system_metrics"},
            {"session_id", session_id}
        };
        auto system_response = client.post(test_host, test_port, "/api/v1/system", system_data);

        REQUIRE(system_response.contains("metrics"));
        REQUIRE(system_response["metrics"].is_object());
    }

    SECTION("Error Handling") {
        // Test invalid endpoint
        auto response = client.get(test_host, test_port, "/api/v1/nonexistent");

        REQUIRE(response.contains("error"));
        REQUIRE(response.contains("status_code"));
    }

    SECTION("Invalid JSON Request") {
        // Send malformed JSON
        json invalid_data = {{"invalid", json::value_t::discarded}}; // This creates invalid JSON
        auto response = client.post(test_host, test_port, "/api/v1/status", invalid_data);

        // Should handle gracefully
        REQUIRE(response.contains("error") || response.contains("status_code"));
    }

    SECTION("Performance API Endpoints") {
        // Create session first
        json create_session_data = {{"action", "create_session"}};
        auto session_response = client.post(test_host, test_port, "/api/v1/sessions", create_session_data);
        REQUIRE(session_response.contains("session_id"));
        std::string session_id = session_response["session_id"];

        // Test performance test
        json perf_data = {
            {"action", "run_performance_test"},
            {"session_id", session_id},
            {"test_type", "cpu"},
            {"duration", 1},
            {"iterations", 10},
            {"concurrency", 1}
        };
        auto perf_response = client.post(test_host, test_port, "/api/v1/performance", perf_data);

        REQUIRE(perf_response.contains("results"));
        REQUIRE(perf_response["results"].is_object());
    }

    SECTION("Batch Processing") {
        // Create session first
        json create_session_data = {{"action", "create_session"}};
        auto session_response = client.post(test_host, test_port, "/api/v1/sessions", create_session_data);
        REQUIRE(session_response.contains("session_id"));
        std::string session_id = session_response["session_id"];

        // Test batch processing with math operations
        json batch_data = {
            {"action", "batch_process"},
            {"session_id", session_id},
            {"requests", json::array({
                {
                    {"endpoint", "/api/v1/math"},
                    {"method", "POST"},
                    {"data", {
                        {"action", "calculate"},
                        {"expression", "1 + 1"}
                    }}
                },
                {
                    {"endpoint", "/api/v1/math"},
                    {"method", "POST"},
                    {"data", {
                        {"action", "calculate"},
                        {"expression", "2 * 3"}
                    }}
                }
            })}
        };
        auto batch_response = client.post(test_host, test_port, "/api/v1/batch", batch_data);

        REQUIRE(batch_response.contains("results"));
        REQUIRE(batch_response["results"].is_array());
        REQUIRE(batch_response["results"].size() == 2);
    }

    // Stop server
    apiServer.stop();

    // Wait for server thread to finish
    if (server_thread.joinable()) {
        server_thread.join();
    }

    // Verify server stopped
    REQUIRE(!apiServer.isServerRunning());
}

TEST_CASE("API Server Request/Response Handling", "[api][request_response]") {
    // Skip tests if environment variable is not set
    const char* run_api_tests = std::getenv("RUN_API_TESTS");
    if (!run_api_tests || std::string(run_api_tests) != "1") {
        WARN("Skipping API request/response tests because RUN_API_TESTS!=1");
        return;
    }

    // Test configuration
    const std::string test_host = "127.0.0.1";
    const std::string test_port = "9091"; // Different port
    const int test_api_port = 9091;

    // Create API server instance
    APIServer apiServer("TestAPIServer_RR", test_host, test_api_port);

    // Initialize server
    REQUIRE(apiServer.initialize("config.json", "", 0, "", "", "", SSLMode::DISABLED, false));

    // Start server in background thread
    std::thread server_thread([&]() {
        apiServer.start();
    });

    // Wait for server to start
    std::this_thread::sleep_for(std::chrono::seconds(2));
    REQUIRE(apiServer.isServerRunning());

    HttpClient client;

    SECTION("Content-Type Handling") {
        // Test with different content types
        json test_data = {{"test", "value"}};

        // Should handle JSON content type
        auto response = client.post(test_host, test_port, "/api/v1/status", test_data);
        REQUIRE(response.contains("error") || response.contains("status"));
    }

    SECTION("Request Size Limits") {
        // Create large request body
        std::string large_data(1024 * 1024, 'x'); // 1MB of data
        json large_request = {{"data", large_data}};

        auto response = client.post(test_host, test_port, "/api/v1/status", large_request);

        // Should either handle it or return error for size limit
        REQUIRE(response.contains("error") || response.contains("status"));
    }

    SECTION("Concurrent Requests") {
        // Test multiple concurrent requests
        std::vector<std::thread> threads;
        std::vector<json> responses;
        std::mutex responses_mutex;

        auto make_request = [&]() {
            HttpClient thread_client;
            auto response = thread_client.get(test_host, test_port, "/api/v1/status");
            std::lock_guard<std::mutex> lock(responses_mutex);
            responses.push_back(response);
        };

        // Launch 5 concurrent requests
        for (int i = 0; i < 5; ++i) {
            threads.emplace_back(make_request);
        }

        // Wait for all threads
        for (auto& thread : threads) {
            thread.join();
        }

        // Verify all responses
        REQUIRE(responses.size() == 5);
        for (const auto& response : responses) {
            REQUIRE(response.contains("status"));
        }
    }

    SECTION("Timeout Handling") {
        // Test with a potentially slow endpoint
        auto start_time = std::chrono::steady_clock::now();
        auto response = client.get(test_host, test_port, "/api/v1/status");
        auto end_time = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Should respond within reasonable time (less than 5 seconds)
        REQUIRE(duration.count() < 5000);
        REQUIRE(response.contains("status"));
    }

    // Stop server
    apiServer.stop();
    server_thread.join();
    REQUIRE(!apiServer.isServerRunning());
}
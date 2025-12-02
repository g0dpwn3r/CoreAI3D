// test/tests/WebSocketTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <atomic>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <nlohmann/json.hpp>

#include "WebSocketServer.hpp"
#include "CoreAI3DCommon.hpp"

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

// WebSocket client helper for integration tests
class WebSocketClient {
private:
    net::io_context ioc;
    tcp::resolver resolver{ioc};
    websocket::stream<tcp::socket> ws{ioc};
    beast::multi_buffer buffer;
    std::atomic<bool> connected{false};
    std::string last_message;

public:
    WebSocketClient() = default;

    ~WebSocketClient() {
        if (connected) {
            close();
        }
    }

    bool connect(const std::string& host, const std::string& port, const std::string& target = "/") {
        try {
            // Resolve the host
            auto const results = resolver.resolve(host, port);

            // Connect to the endpoint
            net::connect(ws.next_layer(), results.begin(), results.end());

            // Perform WebSocket handshake
            ws.handshake(host, target);

            connected = true;
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }

    void close() {
        if (connected) {
            try {
                ws.close(websocket::close_code::normal);
            } catch (...) {}
            connected = false;
        }
    }

    bool send(const std::string& message) {
        if (!connected) return false;

        try {
            ws.write(net::buffer(message));
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }

    bool send_json(const json& data) {
        return send(data.dump());
    }

    std::string receive() {
        if (!connected) return "";

        try {
            beast::multi_buffer buffer;
            ws.read(buffer);
            std::string message = beast::make_printable(buffer.data());
            return message;
        } catch (const std::exception& e) {
            return "";
        }
    }

    json receive_json() {
        std::string msg = receive();
        if (msg.empty()) return json{};
        try {
            return json::parse(msg);
        } catch (...) {
            return json{{"raw_message", msg}};
        }
    }

    bool is_connected() const {
        return connected;
    }
};

TEST_CASE("WebSocket Server Integration Tests", "[websocket][integration][!mayfail]") {
    // Skip tests if environment variable is not set
    const char* run_ws_tests = std::getenv("RUN_WS_TESTS");
    if (!run_ws_tests || std::string(run_ws_tests) != "1") {
        WARN("Skipping WebSocket integration tests because RUN_WS_TESTS!=1");
        return;
    }

    // Test configuration
    const std::string test_host = "127.0.0.1";
    const std::string test_port = "9092"; // Use a different port for testing
    const int test_ws_port = 9092;

    // Create WebSocket server instance
    WebSocketServer wsServer("TestWebSocketServer", test_host, test_ws_port);

    // Initialize server
    REQUIRE(wsServer.initialize("config.json"));

    // Start server in background thread
    std::atomic<bool> server_started{false};
    std::thread server_thread([&]() {
        try {
            server_started = true;
            wsServer.start();
        } catch (const std::exception& e) {
            FAIL(std::string("WebSocket server start failed: ") + e.what());
        }
    });

    // Wait for server to start
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Verify server is running
    REQUIRE(wsServer.isServerRunning());

    SECTION("WebSocket Connection Management") {
        WebSocketClient client;

        // Test successful connection
        REQUIRE(client.connect(test_host, test_port, "/"));

        // Verify connection
        REQUIRE(client.is_connected());

        // Test disconnection
        client.close();
        REQUIRE(!client.is_connected());
    }

    SECTION("WebSocket Message Exchange") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Send a ping message
        json ping_msg = {
            {"type", "ping"},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
        };

        REQUIRE(client.send_json(ping_msg));

        // Receive response (should be pong or some acknowledgment)
        auto response = client.receive_json();

        // Response should be valid JSON
        REQUIRE(!response.empty());
        REQUIRE(response.contains("type") || response.contains("status"));

        client.close();
    }

    SECTION("Session Management over WebSocket") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Send session creation request
        json session_msg = {
            {"action", "create_session"},
            {"client_id", "test_client_123"}
        };

        REQUIRE(client.send_json(session_msg));

        // Receive session response
        auto response = client.receive_json();

        REQUIRE(response.contains("session_id"));
        std::string session_id = response["session_id"];

        // Test session validation
        json validate_msg = {
            {"action", "validate_session"},
            {"session_id", session_id}
        };

        REQUIRE(client.send_json(validate_msg));

        auto validate_response = client.receive_json();
        REQUIRE(validate_response.contains("valid"));
        REQUIRE(validate_response["valid"] == true);

        client.close();
    }

    SECTION("Chat Functionality over WebSocket") {
        WebSocketClient client1, client2;

        REQUIRE(client1.connect(test_host, test_port, "/"));
        REQUIRE(client2.connect(test_host, test_port, "/"));

        // Create sessions for both clients
        json session1_msg = {{"action", "create_session"}, {"client_id", "client1"}};
        json session2_msg = {{"action", "create_session"}, {"client_id", "client2"}};

        REQUIRE(client1.send_json(session1_msg));
        REQUIRE(client2.send_json(session2_msg));

        auto resp1 = client1.receive_json();
        auto resp2 = client2.receive_json();

        std::string session1_id = resp1["session_id"];
        std::string session2_id = resp2["session_id"];

        // Send chat message from client1
        json chat_msg = {
            {"action", "send_chat_message"},
            {"session_id", session1_id},
            {"message", "Hello from client1!"},
            {"message_type", "text"}
        };

        REQUIRE(client1.send_json(chat_msg));

        // Both clients should receive some response
        auto chat_resp1 = client1.receive_json();
        auto chat_resp2 = client2.receive_json();

        REQUIRE(!chat_resp1.empty());
        REQUIRE(!chat_resp2.empty());

        client1.close();
        client2.close();
    }

    SECTION("Real-time Data Streaming") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Create session
        json session_msg = {{"action", "create_session"}, {"client_id", "streaming_client"}};
        REQUIRE(client.send_json(session_msg));

        auto session_resp = client.receive_json();
        std::string session_id = session_resp["session_id"];

        // Start system monitoring stream
        json stream_msg = {
            {"action", "start_system_monitoring"},
            {"session_id", session_id},
            {"update_interval", 1000}
        };

        REQUIRE(client.send_json(stream_msg));

        // Receive initial response
        auto stream_resp = client.receive_json();
        REQUIRE(stream_resp.contains("status") || stream_resp.contains("stream_id"));

        // Wait a bit for streaming data
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));

        // Try to receive streaming data
        auto stream_data = client.receive_json();
        // May or may not receive data depending on timing, but should not error

        // Stop streaming
        json stop_msg = {
            {"action", "stop_stream"},
            {"session_id", session_id},
            {"stream_type", "system_monitoring"}
        };

        REQUIRE(client.send_json(stop_msg));

        client.close();
    }

    SECTION("Broadcasting Functionality") {
        WebSocketClient client1, client2, client3;

        REQUIRE(client1.connect(test_host, test_port, "/"));
        REQUIRE(client2.connect(test_host, test_port, "/"));
        REQUIRE(client3.connect(test_host, test_port, "/"));

        // Create sessions
        json session1 = {{"action", "create_session"}, {"client_id", "broadcast_client1"}};
        json session2 = {{"action", "create_session"}, {"client_id", "broadcast_client2"}};
        json session3 = {{"action", "create_session"}, {"client_id", "broadcast_client3"}};

        REQUIRE(client1.send_json(session1));
        REQUIRE(client2.send_json(session2));
        REQUIRE(client3.send_json(session3));

        auto resp1 = client1.receive_json();
        auto resp2 = client2.receive_json();
        auto resp3 = client3.receive_json();

        std::string session1_id = resp1["session_id"];

        // Create a group
        json group_msg = {
            {"action", "create_group"},
            {"session_id", session1_id},
            {"group_name", "test_group"},
            {"members", json::array({"broadcast_client1", "broadcast_client2", "broadcast_client3"})}
        };

        REQUIRE(client1.send_json(group_msg));

        // Wait for group creation response
        auto group_resp = client1.receive_json();

        // Send group message
        json broadcast_msg = {
            {"action", "send_group_message"},
            {"session_id", session1_id},
            {"group_name", "test_group"},
            {"message", "Broadcast test message"}
        };

        REQUIRE(client1.send_json(broadcast_msg));

        // All clients should receive the message
        auto msg1 = client1.receive_json();
        auto msg2 = client2.receive_json();
        auto msg3 = client3.receive_json();

        // At least some clients should receive the broadcast
        bool received_broadcast = (!msg1.empty() && msg1.contains("message")) ||
                                 (!msg2.empty() && msg2.contains("message")) ||
                                 (!msg3.empty() && msg3.contains("message"));

        REQUIRE(received_broadcast);

        client1.close();
        client2.close();
        client3.close();
    }

    SECTION("Connection Limits and Error Handling") {
        // Test multiple connections
        std::vector<std::unique_ptr<WebSocketClient>> clients;

        // Create 10 connections
        for (int i = 0; i < 10; ++i) {
            auto client = std::make_unique<WebSocketClient>();
            if (client->connect(test_host, test_port, "/")) {
                clients.push_back(std::move(client));
            }
        }

        // Should have at least some successful connections
        REQUIRE(clients.size() > 0);

        // Send messages from all connected clients
        for (size_t i = 0; i < clients.size(); ++i) {
            json msg = {
                {"action", "ping"},
                {"client_id", std::to_string(i)}
            };
            clients[i]->send_json(msg);
        }

        // Clean up connections
        for (auto& client : clients) {
            client->close();
        }
    }

    SECTION("WebSocket Server Status") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Request server status
        json status_msg = {{"action", "get_server_status"}};
        REQUIRE(client.send_json(status_msg));

        auto status_resp = client.receive_json();

        REQUIRE(status_resp.contains("status"));
        REQUIRE(status_resp.contains("active_connections"));
        REQUIRE(status_resp["active_connections"].is_number());

        client.close();
    }

    // Stop server
    wsServer.stop();

    // Wait for server thread to finish
    if (server_thread.joinable()) {
        server_thread.join();
    }

    // Verify server stopped
    REQUIRE(!wsServer.isServerRunning());
}

TEST_CASE("WebSocket Real-time Communication", "[websocket][realtime]") {
    // Skip tests if environment variable is not set
    const char* run_ws_tests = std::getenv("RUN_WS_TESTS");
    if (!run_ws_tests || std::string(run_ws_tests) != "1") {
        WARN("Skipping WebSocket real-time tests because RUN_WS_TESTS!=1");
        return;
    }

    // Test configuration
    const std::string test_host = "127.0.0.1";
    const std::string test_port = "9093"; // Different port
    const int test_ws_port = 9093;

    // Create WebSocket server instance
    WebSocketServer wsServer("TestWebSocketServer_RT", test_host, test_ws_port);

    // Initialize server
    REQUIRE(wsServer.initialize("config.json"));

    // Start server in background thread
    std::thread server_thread([&]() {
        wsServer.start();
    });

    // Wait for server to start
    std::this_thread::sleep_for(std::chrono::seconds(2));
    REQUIRE(wsServer.isServerRunning());

    SECTION("Command Execution over WebSocket") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Create session
        json session_msg = {{"action", "create_session"}, {"client_id", "command_client"}};
        REQUIRE(client.send_json(session_msg));

        auto session_resp = client.receive_json();
        std::string session_id = session_resp["session_id"];

        // Execute a system command (safe command for testing)
        json cmd_msg = {
            {"action", "execute_command"},
            {"session_id", session_id},
            {"command", "echo"},
            {"parameters", json::array({"Hello WebSocket!"})}
        };

        REQUIRE(client.send_json(cmd_msg));

        // Receive command response
        auto cmd_resp = client.receive_json();

        REQUIRE(cmd_resp.contains("result") || cmd_resp.contains("output"));

        client.close();
    }

    SECTION("File Upload Simulation") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Create session
        json session_msg = {{"action", "create_session"}, {"client_id", "upload_client"}};
        REQUIRE(client.send_json(session_msg));

        auto session_resp = client.receive_json();
        std::string session_id = session_resp["session_id"];

        // Simulate file upload
        json upload_msg = {
            {"action", "upload_file"},
            {"session_id", session_id},
            {"file_name", "test.txt"},
            {"file_type", "text/plain"},
            {"file_size", 13},
            {"file_content", "Hello World!\n"}
        };

        REQUIRE(client.send_json(upload_msg));

        // Receive upload response
        auto upload_resp = client.receive_json();

        REQUIRE(upload_resp.contains("status") || upload_resp.contains("file_id"));

        client.close();
    }

    SECTION("Multi-modal Processing") {
        WebSocketClient client;

        REQUIRE(client.connect(test_host, test_port, "/"));

        // Create session
        json session_msg = {{"action", "create_session"}, {"client_id", "multimodal_client"}};
        REQUIRE(client.send_json(session_msg));

        auto session_resp = client.receive_json();
        std::string session_id = session_resp["session_id"];

        // Request multi-modal analysis
        json multimodal_msg = {
            {"action", "process_multi_modal"},
            {"session_id", session_id},
            {"content_type", "text"},
            {"content", "This is a test message for analysis"},
            {"analysis_types", json::array({"sentiment", "keywords", "summary"})}
        };

        REQUIRE(client.send_json(multimodal_msg));

        // Receive analysis response
        auto analysis_resp = client.receive_json();

        REQUIRE(analysis_resp.contains("results") || analysis_resp.contains("analysis"));

        client.close();
    }

    // Stop server
    wsServer.stop();
    server_thread.join();
    REQUIRE(!wsServer.isServerRunning());
}
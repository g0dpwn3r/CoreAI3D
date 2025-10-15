#ifndef WEBSOCKET_SERVER_HPP
#define WEBSOCKET_SERVER_HPP

#include "CoreAI3DCommon.hpp"

#include "ModuleOrchestrator.hpp"
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
private:
    websocket::stream<tcp::socket> ws_;
    beast::flat_buffer buffer_;
    std::string sessionId_;
    std::string clientId_;
    std::atomic<bool> isActive_;

    // Message handling
    std::queue<json> messageQueue_;
    std::mutex queueMutex_;
    std::condition_variable messageCondition_;

    // Stream processing
    std::atomic<bool> isStreaming_;
    std::string currentStreamId_;
    std::vector<json> streamBuffer_;

public:
    WebSocketSession(tcp::socket socket, const std::string& sessionId);
    ~WebSocketSession();

    void run();
    void doAccept();
    void onAccept(beast::error_code ec);
    void doRead();
    void onRead(beast::error_code ec, std::size_t bytes_transferred);
    void onWrite(beast::error_code ec, std::size_t bytes_transferred);

    // Message handling
    void sendMessage(const json& message);
    void sendError(const std::string& error, int errorCode = 400);
    void startStream(const std::string& streamId);
    void sendStreamData(const json& data);
    void endStream();

    // Session management
    std::string getSessionId() const { return sessionId_; }
    std::string getClientId() const { return clientId_; }
    bool isActive() const { return isActive_; }
    void close();

    // Utility functions
    static std::string getTimestampString();

private:
    void handleMessage(const json& message);
    void processChatMessage(const json& chatData);
    void processCommand(const json& commandData);
    void processFileUpload(const json& fileData);
    void processRealTimeRequest(const json& requestData);
};

class WebSocketServer {
private:
    std::unique_ptr<ModuleOrchestrator> orchestrator_;
    std::string serverName_;
    bool isInitialized_;
    std::atomic<bool> isRunning_;

    // Server configuration
    std::string host_;
    int port_;
    int numThreads_;
    net::io_context ioc_;
    tcp::acceptor acceptor_;

    // Session management
    std::map<std::string, std::shared_ptr<WebSocketSession>> activeSessions_;
    std::map<std::string, std::string> clientToSession_;
    std::mutex sessionMutex_;

    // Server threads
    std::vector<std::unique_ptr<std::thread>> serverThreads_;

    // Message broadcasting
    std::map<std::string, std::vector<std::string>> broadcastGroups_;
    std::map<std::string, std::vector<json>> broadcastQueues_;
    std::mutex broadcastMutex_;

protected:
    // Core WebSocket processing
    virtual void processWebSocketMessage(const std::string& sessionId, const json& message);
    virtual json handleChatMessage(const std::string& sessionId, const json& chatData);
    virtual json handleCommand(const std::string& sessionId, const json& commandData);
    virtual json handleFileUpload(const std::string& sessionId, const json& fileData);
    virtual json handleRealTimeRequest(const std::string& sessionId, const json& requestData);

    // Session management
    virtual std::string createWebSocketSession(const std::string& clientId = "");
    virtual void destroyWebSocketSession(const std::string& sessionId);
    virtual std::shared_ptr<WebSocketSession> getWebSocketSession(const std::string& sessionId);

    // Message broadcasting
    virtual void broadcastToGroup(const std::string& groupName, const json& message);
    virtual void broadcastToAll(const json& message);
    virtual void broadcastToClient(const std::string& clientId, const json& message);

    // Real-time processing
    virtual void startRealTimeUpdates(const std::string& sessionId, const std::string& updateType, int intervalMs = 1000);
    virtual void stopRealTimeUpdates(const std::string& sessionId, const std::string& updateType);
    virtual void sendRealTimeUpdate(const std::string& sessionId, const std::string& updateType, const json& data);

public:
    // Constructor
    WebSocketServer(const std::string& name, const std::string& host = "localhost", int port = 8081);
    virtual ~WebSocketServer();

    // Initialization
    bool initialize(const std::string& configPath = "");
    bool start();
    void stop();
    bool isServerRunning() const { return isRunning_; }

    // Server connection handling
    void doAccept();
    void onAccept(beast::error_code ec, tcp::socket socket);

    // Configuration
    void setHost(const std::string& host);
    void setPort(int port);
    void setNumThreads(int threads);

    // Module integration
    bool addVisionModule(const std::string& name, std::unique_ptr<VisionModule> module);
    bool addAudioModule(const std::string& name, std::unique_ptr<AudioModule> module);
    bool addSystemModule(const std::string& name, std::unique_ptr<SystemModule> module);
    bool addWebModule(const std::string& name, std::unique_ptr<WebModule> module);
    bool addMathModule(const std::string& name, std::unique_ptr<MathModule> module);

    // WebSocket endpoints
    struct WSEndpoint {
        std::string path;
        std::string description;
        json messageSchema;
        json responseSchema;
        std::string authentication;
    };

    // Chat functionality
    json sendChatMessage(const std::string& sessionId, const std::string& message, const std::string& messageType = "text");
    json sendChatMessageWithFile(const std::string& sessionId, const std::string& message, const std::string& filePath, const std::string& fileType);
    json getChatHistory(const std::string& sessionId, int limit = 50);
    json clearChatHistory(const std::string& sessionId);

    // Real-time processing
    json startVisionStream(const std::string& sessionId, const std::string& source, int frameRate = 30);
    json startAudioStream(const std::string& sessionId, const std::string& source, int sampleRate = 44100);
    json startSystemMonitoring(const std::string& sessionId, int updateInterval = 1000);
    json startWebMonitoring(const std::string& sessionId, const std::string& url, int updateInterval = 5000);

    json stopStream(const std::string& sessionId, const std::string& streamType);
    json getStreamStatus(const std::string& sessionId, const std::string& streamType);

    // Command execution
    json executeCommand(const std::string& sessionId, const std::string& command, const json& parameters = json::object());
    json executeVisionCommand(const std::string& sessionId, const std::string& command, const json& parameters);
    json executeAudioCommand(const std::string& sessionId, const std::string& command, const json& parameters);
    json executeSystemCommand(const std::string& sessionId, const std::string& command, const json& parameters);
    json executeWebCommand(const std::string& sessionId, const std::string& command, const json& parameters);
    json executeMathCommand(const std::string& sessionId, const std::string& command, const json& parameters);

    // File operations
    json uploadFile(const std::string& sessionId, const std::string& filePath, const std::string& fileType, const std::string& description = "");
    json downloadFile(const std::string& sessionId, const std::string& fileId, const std::string& downloadPath);
    json listFiles(const std::string& sessionId, const std::string& fileType = "");
    json deleteFile(const std::string& sessionId, const std::string& fileId);

    // Multi-modal processing
    json processMultiModal(const std::string& sessionId, const std::string& contentType, const std::string& content, const std::vector<std::string>& analysisTypes);
    json analyzeImageWithAI(const std::string& sessionId, const std::string& imagePath, const std::vector<std::string>& analysisTypes);
    json analyzeAudioWithAI(const std::string& sessionId, const std::string& audioPath, const std::vector<std::string>& analysisTypes);
    json analyzeTextWithAI(const std::string& sessionId, const std::string& text, const std::vector<std::string>& analysisTypes);

    // Collaboration features
    json createGroup(const std::string& sessionId, const std::string& groupName, const std::vector<std::string>& members);
    json joinGroup(const std::string& sessionId, const std::string& groupName);
    json leaveGroup(const std::string& sessionId, const std::string& groupName);
    json sendGroupMessage(const std::string& sessionId, const std::string& groupName, const std::string& message);

    // Status and monitoring
    json getWebSocketStatus();
    json getActiveSessions();
    json getSessionInfo(const std::string& sessionId);
    json getServerMetrics();
    json getRealTimeStats();

    // Event handling
    void registerEventHandler(const std::string& eventType, std::function<void(const std::string&, const json&)> handler);
    void unregisterEventHandler(const std::string& eventType);
    void triggerEvent(const std::string& eventType, const std::string& sessionId, const json& eventData);

    // Utility functions
    std::string generateSessionId();
    static std::string generateClientId();
    static std::string getTimestampString();
    json createSuccessResponse(const std::string& message, const json& data = json::object());
    json createErrorResponse(const std::string& error, int errorCode = 400, const json& details = json::object());

    // Server management
    void restartServer();
    void reloadConfiguration();
    bool isHealthy();
    size_t getActiveConnections();
    size_t getTotalMessages();
};

#endif // WEBSOCKET_SERVER_HPP
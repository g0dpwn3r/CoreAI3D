#ifndef API_SERVER_HPP
#define API_SERVER_HPP

#include "CoreAI3DCommon.hpp"
#include "ModuleOrchestrator.hpp"
#include "VisionModule.hpp"
#include "AudioModule.hpp"
#include "SystemModule.hpp"
#include "WebModule.hpp"
#include "MathModule.hpp"
#include "Train.hpp"
#include "Database.hpp"
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <atomic>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

class APIServer {
private:
    std::unique_ptr<ModuleOrchestrator> orchestrator;
    std::unique_ptr<Training> trainingModule;
    std::string serverName;
    bool isInitialized;
    std::atomic<bool> isRunning;

    // Server configuration
    std::string host;
    int port;
    int numThreads;
    net::io_context ioc;
    tcp::acceptor acceptor{ioc};

    // Server methods
    void do_accept();

    // HTTP session management
    std::vector<std::unique_ptr<std::thread>> serverThreads;
    std::map<std::string, std::string> activeSessions;
    std::map<std::string, json> sessionData;

    // API configuration
    std::string apiVersion;
    std::string corsOrigin;
    int requestTimeout;
    size_t maxRequestSize;
    bool enableLogging;
    std::string logFilePath;

    // Database configuration
    std::string dbHost;
    unsigned int dbPort;
    std::string dbUser;
    std::string dbPassword;
    std::string dbSchema;
    SSLMode dbSSLMode;
    bool createTables;
    std::unique_ptr<Database> database;

protected:
    // Core API processing
    virtual json processAPIRequest(const std::string& endpoint, const std::string& method, const json& requestData);
    virtual json handleVisionAPI(const std::string& action, const json& parameters);
    virtual json handleAudioAPI(const std::string& action, const json& parameters);
    virtual json handleSystemAPI(const std::string& action, const json& parameters);
    virtual json handleWebAPI(const std::string& action, const json& parameters);
    virtual json handleMathAPI(const std::string& action, const json& parameters);
    virtual json handleNeuralAPI(const std::string& action, const json& parameters);
    virtual json handleOrchestratorAPI(const std::string& action, const json& parameters);

    // HTTP request handling
    virtual void handleRequest(http::request<http::string_body>&& req, tcp::socket socket);
    virtual json parseRequestBody(const std::string& body);
    virtual std::string createResponse(const json& responseData, unsigned int statusCode = 200);
    virtual void sendErrorResponse(tcp::socket& socket, const std::string& error, unsigned int statusCode = 400);

public:
    void handleRequestInternal(http::request<http::string_body>&& req, tcp::socket socket);

    // Session management
    virtual std::string createSession(const std::string& clientId = "");
    virtual void destroySession(const std::string& sessionId);
    virtual bool validateSession(const std::string& sessionId);
    virtual json getSessionData(const std::string& sessionId);
    virtual void updateSessionData(const std::string& sessionId, const json& data);

    // Authentication
    virtual bool authenticateRequest(const std::string& sessionId, const json& request);
    virtual std::string generateAPIKey();
    virtual bool validateAPIKey(const std::string& apiKey);

public:
    // Constructor
    APIServer(const std::string& name, const std::string& host = "localhost", int port = 8080);
    virtual ~APIServer();

    // Initialization
    bool initialize(const std::string& configPath = "config.json",
                   const std::string& dbHost = "localhost",
                   unsigned int dbPort = 33060,
                   const std::string& dbUser = "user",
                   const std::string& dbPassword = "password",
                   const std::string& dbSchema = "coreai_db",
                   SSLMode dbSSLMode = SSLMode::DISABLED,
                   bool createTables = false);
    bool start();
    void stop();
    bool isServerRunning() const { return isRunning; }

    // Configuration
    void setHost(const std::string& host);
    void setPort(int port);
    void setNumThreads(int threads);
    void setAPIVersion(const std::string& version);
    void setCORSOrigin(const std::string& origin);
    void setRequestTimeout(int seconds);
    void setMaxRequestSize(size_t maxSize);
    void enableRequestLogging(bool enable);
    void setLogFile(const std::string& logPath);

    // Module integration
    bool addVisionModule(const std::string& name, std::unique_ptr<VisionModule> module);
    bool addAudioModule(const std::string& name, std::unique_ptr<AudioModule> module);
    bool addSystemModule(const std::string& name, std::unique_ptr<SystemModule> module);
    bool addWebModule(const std::string& name, std::unique_ptr<WebModule> module);
    bool addMathModule(const std::string& name, std::unique_ptr<MathModule> module);
    void setTrainingModule(std::unique_ptr<Training> training);

    // Database configuration getters
    const std::string& getDBHost() const { return dbHost; }
    unsigned int getDBPort() const { return dbPort; }
    const std::string& getDBUser() const { return dbUser; }
    const std::string& getDBPassword() const { return dbPassword; }
    const std::string& getDBSchema() const { return dbSchema; }
    SSLMode getDBSSLMode() const { return dbSSLMode; }
    bool getCreateTables() const { return createTables; }
    Database* getDatabase() const { return database.get(); }

    // API endpoints
    struct APIEndpoint {
        std::string path;
        std::string method;
        std::string description;
        json parameters;
        json responseSchema;
        std::string module;
        std::string action;
    };

    // Vision API endpoints
    json visionClassify(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName = "default");
    json visionDetectObjects(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName = "default");
    json visionOCR(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName = "default");
    json visionAnalyzeFaces(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName = "default");
    json visionProcessVideo(const std::string& sessionId, const std::string& videoPath, const std::string& moduleName = "default");

    // Audio API endpoints
    json audioSpeechToText(const std::string& sessionId, const std::string& audioPath, const std::string& moduleName = "default");
    json audioTextToSpeech(const std::string& sessionId, const std::string& text, const std::string& voice = "default", const std::string& moduleName = "default");
    json audioAnalyze(const std::string& sessionId, const std::string& audioPath, const std::string& moduleName = "default");
    json audioProcess(const std::string& sessionId, const std::string& audioPath, const std::string& effects, const std::string& moduleName = "default");

    // System API endpoints
    json systemGetProcesses(const std::string& sessionId);
    json systemStartApplication(const std::string& sessionId, const std::string& appName, const std::vector<std::string>& args = {});
    json systemAutomateTask(const std::string& sessionId, const std::string& taskType, const json& parameters);
    json systemGetSystemMetrics(const std::string& sessionId);
    json systemCaptureScreen(const std::string& sessionId, const std::string& outputPath = "");

    // Web API endpoints
    json webSearch(const std::string& sessionId, const std::string& query, int maxResults = 10);
    json webExtractContent(const std::string& sessionId, const std::string& url);
    json webGetNews(const std::string& sessionId, const std::string& topic, int maxArticles = 10);
    json webAnalyzeSentiment(const std::string& sessionId, const std::string& content);

    // Math API endpoints
    json mathCalculate(const std::string& sessionId, const std::string& expression);
    json mathOptimize(const std::string& sessionId, const std::string& objective, const std::vector<float>& initialGuess);
    json mathStatistics(const std::string& sessionId, const std::vector<float>& data);
    json mathMatrixOperation(const std::string& sessionId, const std::string& operation, const std::string& matrixName, const json& parameters);

    // Neural API endpoints
    json neuralGetTopology(const std::string& sessionId);
    json neuralGetActivity(const std::string& sessionId);

    // Neural API helper methods
    json getNeuralTopology();
    json getNeuralActivity();

    // Orchestrator API endpoints
    json orchestratorSubmitTask(const std::string& sessionId, const std::string& taskType, const std::string& description, const json& parameters);
    json orchestratorGetTaskStatus(const std::string& sessionId, const std::string& taskId);
    json orchestratorGetSystemState(const std::string& sessionId);
    json orchestratorProcessMultiModal(const std::string& sessionId, const std::string& contentType, const std::string& content, const std::vector<std::string>& analysisTypes);

    // Batch processing
    json batchProcess(const std::string& sessionId, const std::vector<json>& requests);
    json batchVisionProcess(const std::string& sessionId, const std::vector<std::string>& imagePaths, const std::vector<std::string>& operations);
    json batchAudioProcess(const std::string& sessionId, const std::vector<std::string>& audioPaths, const std::vector<std::string>& operations);

    // Real-time processing
    json startRealTimeProcessing(const std::string& sessionId, const std::string& processingType, const json& parameters);
    json stopRealTimeProcessing(const std::string& sessionId, const std::string& processingType);
    json getRealTimeResults(const std::string& sessionId, const std::string& processingType);

    // File upload/download
    json uploadFile(const std::string& sessionId, const std::string& filePath, const std::string& fileType);
    json downloadFile(const std::string& sessionId, const std::string& fileId, const std::string& downloadPath);
    json listUploadedFiles(const std::string& sessionId);

    // Status and information
    json getAPIStatus();
    json getModuleStatus(const std::string& moduleName);
    json getAllModulesStatus();
    json getAPIEndpoints();
    json getAPIVersion();

    // Logging and monitoring
    json getRequestLogs(const std::string& sessionId, int limit = 100);
    json getPerformanceMetrics(const std::string& sessionId);
    json getErrorLogs(const std::string& sessionId, int limit = 50);

    // Utility functions
    std::string generateUniqueId();
    std::string getTimestampString();
    json createSuccessResponse(const std::string& message, const json& data = json::object());
    json createErrorResponse(const std::string& error, int errorCode = 400, const json& details = json::object());

    // Server management
    void restartServer();
    void reloadConfiguration();
    bool isHealthy();
    size_t getActiveConnections();
    size_t getTotalRequests();
};

#endif // API_SERVER_HPP
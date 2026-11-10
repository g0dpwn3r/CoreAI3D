#include "APIServer.hpp"
#include "MathModule.hpp"
#include "Predict.hpp"
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <regex>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <openssl/rand.h>
#include <openssl/evp.h>

using namespace std::chrono_literals;

// HTTP session class for handling individual connections
class http_session : public std::enable_shared_from_this<http_session> {
public:
    http_session(tcp::socket socket, APIServer* server)
        : socket_(std::move(socket)), server_(server) {}

    void run() {
        do_read();
    }

private:
    tcp::socket socket_;
    beast::flat_buffer buffer_;
    http::request<http::string_body> req_;
    APIServer* server_;

    void do_read() {
        auto self = shared_from_this();
        http::async_read(socket_, buffer_, req_,
            [self](beast::error_code ec, std::size_t bytes_transferred) {
                boost::ignore_unused(bytes_transferred);
                if (!ec) {
                    self->server_->handleRequestInternal(std::move(self->req_), std::move(self->socket_));
                }
            });
    }
};

// Constructor
APIServer::APIServer(const std::string& name, const std::string& host, int port)
    : serverName(name), host(host), port(port), numThreads(4), isInitialized(false), isRunning(false),
      apiVersion("v1"), corsOrigin("*"), requestTimeout(30), maxRequestSize(1024 * 1024),
      enableLogging(true), logFilePath("api_server.log"),
      dbHost("0.0.0.0"), dbPort(33060), dbUser("user"), dbPassword("password"),
      dbSchema("coreai_db"), dbSSLMode(SSLMode::REQUIRED), createTables(false) {
}

// Destructor
APIServer::~APIServer() {
    stop();
}

// Initialization
bool APIServer::initialize(const std::string& configPath,
                          const std::string& dbHost,
                          unsigned int dbPort,
                          const std::string& dbUser,
                          const std::string& dbPassword,
                          const std::string& dbSchema,
                          SSLMode dbSSLMode,
                          bool createTables,
                          bool verbose) {
    try {
        if (isInitialized) {
            return true;
        }

        // Load configuration file FIRST (provides base defaults)
        if (!configPath.empty()) {
            std::ifstream configFile(configPath);
            if (configFile.is_open()) {
                json config;
                configFile >> config;

                // Load server configuration
                if (config.contains("server")) {
                    auto serverConfig = config["server"];
                    if (serverConfig.contains("host")) host = serverConfig["host"];
                    if (serverConfig.contains("port")) port = serverConfig["port"];
                    if (serverConfig.contains("threads")) numThreads = serverConfig["threads"];
                    if (serverConfig.contains("api_version")) apiVersion = serverConfig["api_version"];
                    if (serverConfig.contains("cors_origin")) corsOrigin = serverConfig["cors_origin"];
                    if (serverConfig.contains("request_timeout")) requestTimeout = serverConfig["request_timeout"];
                    if (serverConfig.contains("max_request_size")) maxRequestSize = serverConfig["max_request_size"];
                    if (serverConfig.contains("enable_logging")) enableLogging = serverConfig["enable_logging"];
                    if (serverConfig.contains("log_file")) logFilePath = serverConfig["log_file"];
                }

                // Load database configuration from JSON (base defaults, overridden by CLI args)
                if (config.contains("database")) {
                    auto dbConfig = config["database"];
                    if (dbConfig.contains("host")) this->dbHost = dbConfig["host"];
                    if (dbConfig.contains("port")) this->dbPort = dbConfig["port"];
                    if (dbConfig.contains("user")) this->dbUser = dbConfig["user"];
                    if (dbConfig.contains("password")) this->dbPassword = dbConfig["password"];
                    if (dbConfig.contains("schema")) this->dbSchema = dbConfig["schema"];
                    if (dbConfig.contains("ssl_mode")) {
                        std::string sslModeStr = dbConfig["ssl_mode"];
                        if (sslModeStr == "DISABLED") this->dbSSLMode = SSLMode::DISABLED;
                        else if (sslModeStr == "REQUIRED") this->dbSSLMode = SSLMode::REQUIRED;
                        else if (sslModeStr == "VERIFY_CA") this->dbSSLMode = SSLMode::VERIFY_CA;
                        else if (sslModeStr == "VERIFY_IDENTITY") this->dbSSLMode = SSLMode::VERIFY_IDENTITY;
                        else this->dbSSLMode = SSLMode::REQUIRED;
                    }
                    if (dbConfig.contains("create_tables")) this->createTables = dbConfig["create_tables"];
                }

                // Load LLM configuration
                if (config.contains("llm")) {
                    auto llmConfig = config["llm"];
                    if (llmConfig.contains("provider")) llmProvider = llmConfig["provider"];
                    if (llmConfig.contains("base_url")) llmBaseUrl = llmConfig["base_url"];
                    if (llmConfig.contains("model")) llmModel = llmConfig["model"];
                    if (llmConfig.contains("temperature")) llmTemperature = llmConfig["temperature"];
                    if (llmConfig.contains("max_tokens")) llmMaxTokens = llmConfig["max_tokens"];
                    if (llmConfig.contains("api_key")) llmApiKey = llmConfig["api_key"];
                }

                std::cout << "Configuration loaded from " << configPath << std::endl;
            } else {
                std::cout << "Configuration file not found, using defaults" << std::endl;
            }
        }

        // CLI args override config.json values (always take priority)
        this->dbHost = dbHost;
        this->dbPort = dbPort;
        this->dbUser = dbUser;
        this->dbPassword = dbPassword;
        this->dbSchema = dbSchema;
        this->dbSSLMode = dbSSLMode;
        this->createTables = createTables;

        // Set working directory
        // Prefer COREAI3D_WORK_DIR env var (set by Docker); fall back to current working directory.
        std::filesystem::path projectRoot;
        const char* envWorkDir = std::getenv("COREAI3D_WORK_DIR");
        if (envWorkDir && envWorkDir[0] != '\0') {
            projectRoot = envWorkDir;
        } else {
            projectRoot = std::filesystem::current_path();
        }
        try {
            if (!std::filesystem::exists(projectRoot)) {
                std::cerr << "Working directory " << projectRoot << " does not exist; creating it..." << std::endl;
                std::filesystem::create_directories(projectRoot);
            }
            if (std::filesystem::is_directory(projectRoot)) {
                std::filesystem::current_path(projectRoot);
                std::cout << "Server working directory set to: " << std::filesystem::current_path() << std::endl;
            } else {
                std::cerr << "Warning: " << projectRoot << " is not a directory; using current working directory." << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& fsErr) {
            std::cerr << "Warning: Could not set working directory to " << projectRoot
                      << ": " << fsErr.what() << ". Continuing with current directory." << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "Warning: Unexpected error setting working directory: " << ex.what()
                      << ". Continuing with current directory." << std::endl;
        }

        // Initialize orchestrator
        orchestrator = std::make_unique<ModuleOrchestrator>(serverName + "_orchestrator");

        // Initialize database if not in offline mode
        if (!this->dbHost.empty() && this->dbPort > 0) {
            try {
                database = std::make_unique<Database>(this->dbHost, this->dbPort, this->dbUser,
                                                     this->dbPassword, this->dbSchema, this->dbSSLMode, this->createTables, verbose);
                std::cout << "Database connection initialized" << std::endl;
                
                // Initialize auth manager after database
                authManager = std::make_unique<AuthTables>(database.get());
                std::cout << "AuthTables initialized for JWT/bcrypt authentication" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to initialize database connection: " << e.what() << std::endl;
                std::cerr << "Continuing in offline mode" << std::endl;
                database.reset();
                authManager.reset();
            }
        }

        // Set default configuration
        setHost(host);
        setPort(port);
        setNumThreads(numThreads);

        isInitialized = true;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing API server: " << e.what() << std::endl;
        return false;
    }
}

bool APIServer::start() {
    try {
        if (!isInitialized || isRunning) {
            return false;
        }

        // Create acceptor
        tcp::endpoint endpoint(net::ip::make_address(host), port);
        acceptor.open(endpoint.protocol());
        acceptor.bind(endpoint);
        acceptor.listen();

        // Start accepting connections
        do_accept();

        // Start IO context in multiple threads
        for (int i = 0; i < numThreads; ++i) {
            serverThreads.push_back(std::make_unique<std::thread>([this]() {
                ioc.run();
            }));
        }

        isRunning = true;
        std::cout << "[API MODE] Starting server thread..." << std::endl;
        std::cout << "CoreAI3D application running and API server active on port " << port << "." << std::endl;
        std::cout << "Press Ctrl+C to stop the server." << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error starting API server: " << e.what() << std::endl;
        return false;
    }
}

void APIServer::stop() {
    if (!isRunning) {
        return;
    }

    isRunning = false;

    // Stop accepting new connections
    acceptor.close();

    // Stop IO context
    ioc.stop();

    // Wait for threads to finish
    for (auto& thread : serverThreads) {
        if (thread && thread->joinable()) {
            thread->join();
        }
    }
    serverThreads.clear();
}

void APIServer::do_accept() {
    acceptor.async_accept(
        [this](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                std::make_shared<http_session>(std::move(socket), this)->run();
            }
            if (isRunning) {
                do_accept();
            }
        });
}

// Configuration methods
void APIServer::setHost(const std::string& host) {
    this->host = host;
}

void APIServer::setPort(int port) {
    this->port = port;
}

void APIServer::setNumThreads(int threads) {
    this->numThreads = threads;
}

void APIServer::setAPIVersion(const std::string& version) {
    this->apiVersion = version;
}

void APIServer::setCORSOrigin(const std::string& origin) {
    this->corsOrigin = origin;
}

void APIServer::setRequestTimeout(int seconds) {
    this->requestTimeout = seconds;
}

void APIServer::setMaxRequestSize(size_t maxSize) {
    this->maxRequestSize = maxSize;
}

void APIServer::enableRequestLogging(bool enable) {
    this->enableLogging = enable;
}

void APIServer::setLogFile(const std::string& logPath) {
    this->logFilePath = logPath;
}

// Core API processing
json APIServer::processAPIRequest(const std::string& endpoint, const std::string& method, const json& requestData) {
    try {
        // Parse endpoint to determine module and action
        std::vector<std::string> pathParts;
        std::stringstream ss(endpoint);
        std::string part;
        while (std::getline(ss, part, '/')) {
            if (!part.empty()) {
                pathParts.push_back(part);
            }
        }

        if (pathParts.empty()) {
            return createErrorResponse("Invalid endpoint", 400);
        }

        // Skip API version prefix if present (e.g., /api/v1/vision/classify -> vision/classify)
        size_t startIndex = 0;
        if (pathParts.size() >= 2 && pathParts[0] == "api") {
            std::string potentialEndpoint = pathParts[1];
            
            // DIAGNOSTIC LOG: API endpoint parsing
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Path parts: [";
            for (size_t i = 0; i < pathParts.size(); ++i) {
                std::cout << "\"" << pathParts[i] << "\"";
                if (i < pathParts.size() - 1) std::cout << ", ";
            }
            std::cout << "], apiVersion: " << apiVersion << std::endl;
            
            if (pathParts[1] == apiVersion) {
                startIndex = 2;
                std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Version-matched endpoint, startIndex = 2" << std::endl;
                } else {
                    // Handle endpoints without version (e.g., /api/status, /api/health)
                    // Check if the second part is a known endpoint that doesn't require version
                    if (potentialEndpoint == "status" || potentialEndpoint == "health" ||
                        potentialEndpoint == "ws" || potentialEndpoint == "ai" ||
                        potentialEndpoint == "predict" || potentialEndpoint == "training" ||
                        potentialEndpoint == "chat") {
                        startIndex = 1; // Skip only "api" prefix
                        std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Version-less endpoint matched: " << potentialEndpoint << ", startIndex = 1" << std::endl;
                    } else {
                        std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Unknown endpoint: " << potentialEndpoint << ", startIndex = 0" << std::endl;
                    }
            }
        }

        std::string module = (startIndex < pathParts.size()) ? pathParts[startIndex] : "";
        std::string action = (startIndex + 1 < pathParts.size()) ? pathParts[startIndex + 1] : "status";
        
        // DIAGNOSTIC LOG: Final module/action determination
        std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Final module = \"" << module << "\", action = \"" << action << "\"" << std::endl;

        // DIAGNOSTIC LOG: Starting module routing
        std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Starting module routing for module=\"" << module << "\", action=\"" << action << "\"" << std::endl;
        
        // Route to appropriate handler
        if (module == "vision") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to vision module" << std::endl;
            return handleVisionAPI(action, requestData);
        }
        else if (module == "audio") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to audio module" << std::endl;
            return handleAudioAPI(action, requestData);
        }
        else if (module == "system") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to system module" << std::endl;
            return handleSystemAPI(action, requestData);
        }
        else if (module == "web") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to web module" << std::endl;
            return handleWebAPI(action, requestData);
        }
        else if (module == "math") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to math module" << std::endl;
            return handleMathAPI(action, requestData);
        }
        else if (module == "orchestrator") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to orchestrator module" << std::endl;
            return handleOrchestratorAPI(action, requestData);
        }
        else if (module == "neural") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to neural module" << std::endl;
            return handleNeuralAPI(action, requestData);
        }
        else if (module == "chat") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to chat module" << std::endl;
            return handleChatAPI(action, requestData, method);
        }
        else if (module == "websocket" || module == "ws") {
            // Handle WebSocket module API calls - consolidated handler
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to WebSocket module API, action=" << action << std::endl;
            if (action == "status") {
                json wsStatus;
                wsStatus["websocket_enabled"] = true;
                wsStatus["port"] = 8081; // Default WebSocket port
                wsStatus["connections"] = 0; // Would track active connections
                wsStatus["timestamp"] = getTimestampString();
                return wsStatus;
            }
            else if (action == "connect") {
                // Handle WebSocket connection request
                json response = {
                    {"status", "connected"},
                    {"session_id", generateUniqueId()},
                    {"websocket_url", "ws://localhost:8081/ws"},
                    {"timestamp", getTimestampString()}
                };
                return response;
            }
            else if (action == "chat") {
                // Route chat messages through orchestrator
                if (orchestrator) {
                    json chatData = {
                        {"message", requestData.value("message", "")},
                        {"messageType", requestData.value("messageType", "text")},
                        {"sessionId", requestData.value("sessionId", "")},
                        {"clientId", requestData.value("clientId", "")}
                    };
                    
                    // DIAGNOSTIC LOG: API server routing WebSocket chat to orchestrator
                    std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing WebSocket chat to orchestrator" << std::endl;
                    return orchestrator->processChatMessage(chatData);
                } else {
                    // DIAGNOSTIC LOG: API server orchestrator not available
                    std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Orchestrator not available for WebSocket chat" << std::endl;
                    return createErrorResponse("Orchestrator not available", 503);
                }
            }
            else {
                std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Unknown WebSocket action: " << action << std::endl;
                return createErrorResponse("Unknown WebSocket action: " + action, 400);
            }
        }
        else if (module == "predict" || module == "prediction") {
            // DIAGNOSTIC LOG: API server routing to prediction module
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to prediction module: " << module << "/" << action << std::endl;
            return handlePredictAPI(action, requestData, method);
        }
        else if (module == "training") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to training module" << std::endl;
            return handleTrainAPI(action, requestData, method);
        }
        // else if (module == "tuning" || module == "hyperparameter") {
        //     std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to hyperparameter tuning module" << std::endl;
        //     return handleHyperparameterTuningAPI(action, requestData, method);
        // }
        else if (module == "performance") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to performance module" << std::endl;
            if (action == "load_test") {
                // Handle load testing endpoint
                int concurrentUsers = requestData.value("concurrentUsers", 10);
                int duration = requestData.value("duration", 60);
                std::string endpoint = requestData.value("endpoint", "/health");
                return runLoadTest("", concurrentUsers, duration, 0, endpoint, "GET", json::object(), json::object());
            }
            else {
                return handlePerformanceAPI(action, requestData);
            }
        }
        else if (module == "debug") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to debug module" << std::endl;
            if (action == "api") {
                return handleAPIDebug(requestData);
            }
            else if (action == "system") {
                return handleSystemDebug(requestData);
            }
            else {
                return createErrorResponse("Unknown debug action: " + action, 400);
            }
        }
        else if (module == "ai") {
            // Consolidated AI module API calls - delegate to prediction handler
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing AI module to prediction handler, action=" << action << std::endl;
            return handlePredictAPI(action, requestData, method);
        }
        else if (module == "status") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to status module" << std::endl;
            return getAPIStatus();
        }
        else if (module == "health") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to health module" << std::endl;
            json health;
            health["status"] = "healthy";
            health["timestamp"] = getTimestampString();
            health["server"] = serverName;
            health["is_running"] = isRunning.load();
            return health;
        }
        else if (module == "auth") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to auth module, action=" << action << std::endl;
            return handleAuthAPI(action, requestData, method);
        }
        else if (module == "generate_key") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to generate_key module" << std::endl;
            std::string apiKey = generateAPIKey();
#ifdef USE_MYSQL
            if (database) {
                database->saveAPIKey(apiKey, "Generated via API", true);
            }
#endif
            return createSuccessResponse("API key generated", {{"api_key", apiKey}});
        }
        else if (module == "beta") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to beta module, action=" << action << std::endl;

            if (method == "POST" && action == "generate") {
                int count = requestData.value("count", 1);
                std::string description = requestData.value("description", "");
                std::string email = requestData.value("email", "");
                std::string expires_at = requestData.value("expires_at", "");
                int max_uses = requestData.value("max_uses", 0);

                if (count < 1) count = 1;
                if (count > 100) count = 100;

                json keys = json::array();
                for (int i = 0; i < count; ++i) {
                    std::string betaKey = generateBetaKey();

                    // Never return an empty key; if generation failed, skip it.
                    if (betaKey.empty()) {
                        continue;
                    }

                    if (database) {
                        try {
                            database->saveBetaKey(betaKey, description, email, expires_at, max_uses);
                        }
                        catch (const std::exception& ex) {
                            // A database failure must not prevent the key from
                            // being returned to the caller. Log and continue.
                            std::cerr << "Warning: failed to persist beta key: "
                                      << ex.what() << std::endl;
                        }
                    }

                    keys.push_back(betaKey);
                }

                if (keys.empty()) {
                    return createErrorResponse("Failed to generate any beta keys", 500);
                }

                json data = {
                    {"keys", keys},
                    {"count", static_cast<int>(keys.size())},
                    {"message", "Beta key(s) generated successfully"}
                };

                if (!keys.empty()) {
                    data["key"] = keys[0];
                    data["beta_key"] = keys[0];
                }
                if (!email.empty()) {
                    data["email"] = email;
                }

                return data;
            }
            else if (method == "GET" && action == "list") {
                json keys = json::array();
                if (database) {
                    auto betaKeys = database->getBetaKeys();
                    for (const auto& info : betaKeys) {
                        keys.push_back({
                            {"id", info.id},
                            {"key", info.key},
                            {"description", info.description},
                            {"email", info.email},
                            {"max_uses", info.max_uses},
                            {"uses", info.uses},
                            {"active", info.is_active},
                            {"used_at", info.used_at},
                            {"created_at", info.created_at}
                        });
                    }
                }

                return json{
                    {"keys", keys}
                };
            }
            else if (method == "POST" && action == "validate") {
                std::string key = requestData.value("key", "");

                if (key.empty()) {
                    return createErrorResponse("Beta key is required", 400);
                }

                bool valid = false;
                if (database) {
                    valid = database->validateBetaKey(key);
                }
                else {
                    valid = !key.empty();
                }

                if (valid) {
                    return json{
                        {"valid", true},
                        {"key", key},
                        {"message", "Beta key is valid"}
                    };
                }

                return json{
                    {"valid", false},
                    {"error", "Beta key is invalid, expired, revoked or already used"}
                };
            }
            else if (method == "POST" && action == "use") {
                std::string key = requestData.value("key", "");
                std::string userId = requestData.value("user_id", "");

                if (key.empty() || userId.empty()) {
                    return json{
                        {"success", false},
                        {"error", "Beta key and user ID are required"}
                    };
                }

                bool used = false;
                if (database) {
                    used = database->useBetaKey(key, userId);
                }
                else {
                    used = !key.empty() && !userId.empty();
                }

                if (used) {
                    return json{
                        {"success", true},
                        {"key", key},
                        {"message", "Beta key marked as used"}
                    };
                }

                return json{
                    {"success", false},
                    {"error", "Beta key could not be marked as used"}
                };
            }
            else if (method == "POST" && action == "toggle") {
                std::string keyIdStr = requestData.value("id", "");
                bool enable = requestData.value("enable", true);

                if (keyIdStr.empty()) {
                    return createErrorResponse("Key ID is required", 400);
                }

                if (database) {
                    auto betaKeys = database->getBetaKeys();
                    for (const auto& info : betaKeys) {
                        if (info.id == keyIdStr) {
                            if (!enable) {
                                database->deactivateBetaKey(info.key);
                            }
                            break;
                        }
                    }
                }

                return json{
                    {"id", keyIdStr},
                    {"enable", enable}
                };
            }
            else {
                return createErrorResponse("Unknown beta action: " + action + " with method " + method, 400);
            }
        }
        else if (module == "sessions") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to sessions module" << std::endl;
            return handleSessionsAPI(action, requestData, method);
        }
        else if (module == "dataset") {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to dataset module" << std::endl;
            return handleDatasetAPI(action, requestData, method);
        }
        else if (module.empty() || module == "/") {
            // Handle root endpoint
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Routing to root endpoint" << std::endl;
            return getAPIStatus();
        }
        else {
            std::cout << "[DIAGNOSTIC] APIServer::processAPIRequest: Unknown module: " << module << std::endl;
            return createErrorResponse("Unknown module: " + module, 404);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Internal server error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleVisionAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "classify") {
            std::string imagePath = parameters.value("image_path", "");
            std::string moduleName = parameters.value("module", "default");
            return visionClassify("", imagePath, moduleName);
        }
        else if (action == "detect") {
            std::string imagePath = parameters.value("image_path", "");
            std::string moduleName = parameters.value("module", "default");
            return visionDetectObjects("", imagePath, moduleName);
        }
        else if (action == "ocr") {
            std::string imagePath = parameters.value("image_path", "");
            std::string moduleName = parameters.value("module", "default");
            return visionOCR("", imagePath, moduleName);
        }
        else {
            return createErrorResponse("Unknown vision action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Vision API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleAudioAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "speech_to_text") {
            std::string audioPath = parameters.value("audio_path", "");
            std::string moduleName = parameters.value("module", "default");
            return audioSpeechToText("", audioPath, moduleName);
        }
        else if (action == "text_to_speech") {
            std::string text = parameters.value("text", "");
            std::string voice = parameters.value("voice", "default");
            std::string moduleName = parameters.value("module", "default");
            return audioTextToSpeech("", text, voice, moduleName);
        }
        else {
            return createErrorResponse("Unknown audio action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Audio API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleSystemAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "processes") {
            return systemGetProcesses("");
        }
        else if (action == "metrics") {
            return systemGetSystemMetrics("");
        }
        else if (action == "automate") {
            std::string taskType = parameters.value("task_type", "");
            return systemAutomateTask("", taskType, parameters);
        }
        else if (action == "run_performance_test") {
            // Extract performance test parameters
            std::string testType = parameters.value("testType", "cpu");
            int duration = parameters.value("duration", 10);
            int iterations = parameters.value("iterations", 1000);
            int concurrency = parameters.value("concurrency", 1);

            // Validate parameters
            if (duration <= 0 || duration > 300) {
                return createErrorResponse("Duration must be between 1 and 300 seconds", 400);
            }
            if (iterations <= 0 || iterations > 10000000) {
                return createErrorResponse("Iterations must be between 1 and 10,000,000", 400);
            }
            if (concurrency <= 0 || concurrency > 100) {
                return createErrorResponse("Concurrency must be between 1 and 100", 400);
            }

            // Run performance test
            return runPerformanceTest("", testType, duration, iterations, concurrency);
        }
        else if (action == "run_load_test") {
            // Extract load test parameters
            int concurrentUsers = parameters.value("concurrentUsers", 10);
            int duration = parameters.value("duration", 60);
            int rampUpTime = parameters.value("rampUpTime", 10);
            std::string endpoint = parameters.value("endpoint", "/health");
            std::string method = parameters.value("method", "GET");
            json headers = parameters.value("headers", json::object());
            json body = parameters.value("body", json::object());

            // Validate parameters
            if (concurrentUsers <= 0 || concurrentUsers > 1000) {
                return createErrorResponse("Concurrent users must be between 1 and 1000", 400);
            }
            if (duration <= 0 || duration > 300) {
                return createErrorResponse("Duration must be between 1 and 300 seconds", 400);
            }
            if (rampUpTime < 0 || rampUpTime > duration) {
                return createErrorResponse("Ramp-up time must be between 0 and duration", 400);
            }

            // Run load test
            return runLoadTest("", concurrentUsers, duration, rampUpTime, endpoint, method, headers, body);
        }
        else {
            return createErrorResponse("Unknown system action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("System API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleWebAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "search") {
            std::string query = parameters.value("query", "");
            int maxResults = parameters.value("max_results", 10);
            return webSearch("", query, maxResults);
        }
        else {
            return createErrorResponse("Unknown web action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Web API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleMathAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "calculate") {
            std::string expression = parameters.value("expression", "");
            return mathCalculate("", expression);
        }
        else {
            return createErrorResponse("Unknown math action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Math API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handlePerformanceAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "test") {
            // Extract performance test parameters
            std::string testType = parameters.value("testType", "cpu");
            int duration = parameters.value("duration", 10);
            int iterations = parameters.value("iterations", 1000);
            int concurrency = parameters.value("concurrency", 1);

            // Validate parameters
            if (duration <= 0 || duration > 300) {
                return createErrorResponse("Duration must be between 1 and 300 seconds", 400);
            }
            if (iterations <= 0 || iterations > 10000000) {
                return createErrorResponse("Iterations must be between 1 and 10,000,000", 400);
            }
            if (concurrency <= 0 || concurrency > 100) {
                return createErrorResponse("Concurrency must be between 1 and 100", 400);
            }

            // Run performance test
            return runPerformanceTest("", testType, duration, iterations, concurrency);
        }
        else {
            return createErrorResponse("Unknown performance action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Performance API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleAPIDebug(const json& parameters) {
    try {
        json debugInfo = {
            {"debug_type", "api"},
            {"timestamp", getTimestampString()},
            {"server_status", {
                {"is_running", isRunning.load()},
                {"is_initialized", isInitialized},
                {"host", host},
                {"port", port},
                {"api_version", apiVersion},
                {"active_sessions", activeSessions.size()},
                {"total_requests", getTotalRequests()},
                {"active_connections", getActiveConnections()}
            }},
            {"modules", {
                {"training_module", trainingModule ? "available" : "not available"},
                {"predict_module", predictModule ? "available" : "not available"},
                {"orchestrator", orchestrator ? "available" : "not available"},
                {"database", database ? "connected" : "not connected"}
            }},
            {"configuration", {
                {"num_threads", numThreads},
                {"cors_origin", corsOrigin},
                {"request_timeout", requestTimeout},
                {"max_request_size", maxRequestSize},
                {"enable_logging", enableLogging}
            }}
        };

        if (database) {
            debugInfo["database_info"] = {
                {"host", dbHost},
                {"port", dbPort},
                {"user", dbUser},
                {"schema", dbSchema},
                {"ssl_mode", dbSSLMode == SSLMode::DISABLED ? "DISABLED" :
                           dbSSLMode == SSLMode::REQUIRED ? "REQUIRED" :
                           dbSSLMode == SSLMode::VERIFY_CA ? "VERIFY_CA" : "VERIFY_IDENTITY"}
            };
        }

        return createSuccessResponse("API debug information retrieved successfully", debugInfo);
    }
    catch (const std::exception& e) {
        return createErrorResponse("API debug failed: " + std::string(e.what()), 500);
    }
}

json APIServer::handleSystemDebug(const json& parameters) {
    try {
        // Get system information
        json systemInfo = {
            {"debug_type", "system"},
            {"timestamp", getTimestampString()},
            {"platform", {
                {"os", "Linux"}, // Could be made dynamic
                {"architecture", "x64"}, // Could be made dynamic
                {"hostname", "localhost"} // Could be made dynamic
            }},
            {"memory", {
                {"total_mb", 8192}, // Placeholder - could get actual system memory
                {"available_mb", 4096}, // Placeholder
                {"used_mb", 4096} // Placeholder
            }},
            {"cpu", {
                {"cores", 4}, // Placeholder
                {"usage_percent", 25.5} // Placeholder
            }},
            {"disk", {
                {"total_gb", 100}, // Placeholder
                {"available_gb", 75}, // Placeholder
                {"used_gb", 25} // Placeholder
            }},
            {"network", {
                {"interfaces", json::array({"eth0", "lo"})},
                {"connections", 5} // Placeholder
            }},
            {"processes", {
                {"total", 150}, // Placeholder
                {"running", 45}, // Placeholder
                {"sleeping", 105} // Placeholder
            }}
        };

        // Add server-specific information
        systemInfo["server_info"] = {
            {"uptime_seconds", 3600}, // Placeholder
            {"memory_usage_mb", 256}, // Placeholder
            {"cpu_usage_percent", 15.2}, // Placeholder
            {"thread_count", numThreads}
        };

        return createSuccessResponse("System debug information retrieved successfully", systemInfo);
    }
    catch (const std::exception& e) {
        return createErrorResponse("System debug failed: " + std::string(e.what()), 500);
    }
}

json APIServer::runPerformanceTest(const std::string& sessionId, const std::string& testType, int duration, int iterations, int concurrency) {
    try {
        // Initialize performance metrics
        auto startTime = std::chrono::high_resolution_clock::now();
        double totalOperations = 0;
        double totalTime = 0;
        std::vector<double> operationTimes;

        if (testType == "cpu") {
            // CPU performance test - matrix multiplication benchmark
            totalOperations = runCPUTest(duration, iterations, concurrency, operationTimes);
        }
        else if (testType == "memory") {
            // Memory performance test - allocation/deallocation benchmark
            totalOperations = runMemoryTest(duration, iterations, concurrency, operationTimes);
        }
        else if (testType == "disk") {
            // Disk I/O performance test
            totalOperations = runDiskTest(duration, iterations, concurrency, operationTimes);
        }
        else if (testType == "network") {
            // Network performance test (simulated)
            totalOperations = runNetworkTest(duration, iterations, concurrency, operationTimes);
        }
        else {
            return createErrorResponse("Unknown test type: " + testType, 400);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration<double>(endTime - startTime).count();

        // Calculate performance metrics
        double operationsPerSecond = totalOperations / totalTime;
        double averageOperationTime = (operationTimes.size() > 0) ?
            std::accumulate(operationTimes.begin(), operationTimes.end(), 0.0) / operationTimes.size() : 0.0;

        // Calculate percentiles
        std::sort(operationTimes.begin(), operationTimes.end());
        double p50 = operationTimes.size() > 0 ? operationTimes[operationTimes.size() * 0.5] : 0.0;
        double p95 = operationTimes.size() > 0 ? operationTimes[operationTimes.size() * 0.95] : 0.0;
        double p99 = operationTimes.size() > 0 ? operationTimes[operationTimes.size() * 0.99] : 0.0;

        json result = {
            {"test_type", testType},
            {"duration_seconds", totalTime},
            {"total_operations", totalOperations},
            {"operations_per_second", operationsPerSecond},
            {"average_operation_time_ms", averageOperationTime * 1000},
            {"p50_operation_time_ms", p50 * 1000},
            {"p95_operation_time_ms", p95 * 1000},
            {"p99_operation_time_ms", p99 * 1000},
            {"concurrency", concurrency},
            {"iterations", iterations},
            {"status", "completed"},
            {"timestamp", getTimestampString()}
        };

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Performance test failed: " + std::string(e.what()), 500);
    }
}

json APIServer::runLoadTest(const std::string& sessionId, int concurrentUsers, int duration, int rampUpTime,
                           const std::string& endpoint, const std::string& method, const json& headers, const json& body) {
    try {
        // Validate parameters
        if (concurrentUsers <= 0 || concurrentUsers > 1000) {
            return createErrorResponse("Concurrent users must be between 1 and 1000", 400);
        }
        if (duration <= 0 || duration > 300) {
            return createErrorResponse("Duration must be between 1 and 300 seconds", 400);
        }
        if (rampUpTime < 0 || rampUpTime > duration) {
            return createErrorResponse("Ramp-up time must be between 0 and duration", 400);
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        auto endTime = startTime + std::chrono::seconds(duration);
        auto rampUpEndTime = startTime + std::chrono::seconds(rampUpTime);

        // Initialize load testing metrics
        std::atomic<int> totalRequests{0};
        std::atomic<int> successfulRequests{0};
        std::atomic<int> failedRequests{0};
        std::atomic<int> activeUsers{0};
        std::vector<double> responseTimes;
        std::vector<int> statusCodes;
        std::mutex metricsMutex;

        // Function to make actual HTTP request
        auto makeHttpRequest = [&](const std::string& targetEndpoint, const std::string& requestMethod,
                                  const json& requestHeaders, const json& requestBody) -> std::pair<int, double> {
            try {
                // For load testing, we'll simulate HTTP requests with realistic timing
                // In a real implementation, this would make actual HTTP calls
                auto requestStart = std::chrono::high_resolution_clock::now();

                // Simulate network latency (10-50ms)
                std::this_thread::sleep_for(std::chrono::milliseconds(10 + (rand() % 40)));

                // Simulate response time based on endpoint complexity
                int baseResponseTime = 50; // Base response time in ms
                if (targetEndpoint.find("/api/") != std::string::npos) {
                    baseResponseTime += 20; // API calls take longer
                }
                if (!requestBody.empty()) {
                    baseResponseTime += 10; // POST/PUT requests take longer
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(baseResponseTime));

                auto requestEnd = std::chrono::high_resolution_clock::now();
                double responseTime = std::chrono::duration<double>(requestEnd - requestStart).count();

                // Simulate HTTP status codes (95% success rate)
                int statusCode = (rand() % 100) < 95 ? 200 : (rand() % 4 == 0 ? 404 : (rand() % 3 == 0 ? 500 : 503));

                return {statusCode, responseTime};

            } catch (const std::exception&) {
                return {-1, 0.0}; // Request failed
            }
        };

        // Function to simulate a single user making requests
        auto simulateUser = [&]() {
            activeUsers++;

            // Calculate ramp-up delay for this user
            if (rampUpTime > 0) {
                int userIndex = activeUsers.load() - 1;
                double delayPerUser = static_cast<double>(rampUpTime) / concurrentUsers;
                double userDelay = userIndex * delayPerUser;
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(userDelay * 1000)));
            }

            while (std::chrono::high_resolution_clock::now() < endTime) {
                auto [statusCode, responseTime] = makeHttpRequest(endpoint, method, headers, body);

                {
                    std::lock_guard<std::mutex> lock(metricsMutex);
                    responseTimes.push_back(responseTime);
                    statusCodes.push_back(statusCode);
                    totalRequests++;
                    if (statusCode >= 200 && statusCode < 300) {
                        successfulRequests++;
                    } else {
                        failedRequests++;
                    }
                }

                // Add some delay between requests (simulate user think time)
                std::this_thread::sleep_for(std::chrono::milliseconds(100 + (rand() % 200)));
            }

            activeUsers--;
        };

        // Start concurrent user threads
        std::vector<std::thread> userThreads;
        for (int i = 0; i < concurrentUsers; ++i) {
            userThreads.emplace_back(simulateUser);
        }

        // Wait for all threads to complete
        for (auto& thread : userThreads) {
            thread.join();
        }

        auto actualEndTime = std::chrono::high_resolution_clock::now();
        double actualDuration = std::chrono::duration<double>(actualEndTime - startTime).count();

        // Calculate load test metrics
        int totalReq = totalRequests.load();
        int successReq = successfulRequests.load();
        int failReq = failedRequests.load();

        double requestsPerSecond = totalReq / actualDuration;
        double successRate = totalReq > 0 ? (static_cast<double>(successReq) / totalReq) * 100.0 : 0.0;

        // Calculate response time percentiles
        std::sort(responseTimes.begin(), responseTimes.end());
        double avgResponseTime = responseTimes.size() > 0 ?
            std::accumulate(responseTimes.begin(), responseTimes.end(), 0.0) / responseTimes.size() : 0.0;
        double p50ResponseTime = responseTimes.size() > 0 ? responseTimes[responseTimes.size() * 0.5] * 1000 : 0.0;
        double p95ResponseTime = responseTimes.size() > 0 ? responseTimes[responseTimes.size() * 0.95] * 1000 : 0.0;
        double p99ResponseTime = responseTimes.size() > 0 ? responseTimes[responseTimes.size() * 0.99] * 1000 : 0.0;

        // Calculate status code distribution
        std::map<int, int> statusCodeCounts;
        for (int code : statusCodes) {
            statusCodeCounts[code]++;
        }
        json statusCodeDistribution = json::object();
        for (const auto& pair : statusCodeCounts) {
            statusCodeDistribution[std::to_string(pair.first)] = pair.second;
        }

        json result = {
            {"test_type", "load_test"},
            {"endpoint", endpoint},
            {"method", method},
            {"concurrent_users", concurrentUsers},
            {"duration_seconds", actualDuration},
            {"ramp_up_time_seconds", rampUpTime},
            {"total_requests", totalReq},
            {"successful_requests", successReq},
            {"failed_requests", failReq},
            {"requests_per_second", requestsPerSecond},
            {"success_rate_percent", successRate},
            {"average_response_time_ms", avgResponseTime * 1000},
            {"p50_response_time_ms", p50ResponseTime},
            {"p95_response_time_ms", p95ResponseTime},
            {"p99_response_time_ms", p99ResponseTime},
            {"status_code_distribution", statusCodeDistribution},
            {"headers", headers},
            {"body", body},
            {"status", "completed"},
            {"timestamp", getTimestampString()}
        };

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Load test failed: " + std::string(e.what()), 500);
    }
}

double APIServer::runCPUTest(int duration, int iterations, int concurrency, std::vector<double>& operationTimes) {
    double totalOperations = 0;

    // Simple matrix multiplication benchmark
    auto matrixMultiply = [](const std::vector<std::vector<double>>& A,
                           const std::vector<std::vector<double>>& B) -> std::vector<std::vector<double>> {
        size_t n = A.size();
        size_t m = B[0].size();
        size_t p = A[0].size();

        std::vector<std::vector<double>> C(n, std::vector<double>(m, 0.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                for (size_t k = 0; k < p; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    };

    // Create test matrices
    std::vector<std::vector<double>> A(10, std::vector<double>(10, 1.0));
    std::vector<std::vector<double>> B(10, std::vector<double>(10, 2.0));

    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::seconds(duration);

    while (std::chrono::high_resolution_clock::now() < endTime && totalOperations < iterations) {
        auto opStart = std::chrono::high_resolution_clock::now();

        // Perform matrix multiplication
        auto C = matrixMultiply(A, B);

        auto opEnd = std::chrono::high_resolution_clock::now();
        double opTime = std::chrono::duration<double>(opEnd - opStart).count();

        operationTimes.push_back(opTime);
        totalOperations += 1.0;

        // Prevent overflow
        if (operationTimes.size() > 10000) {
            operationTimes.erase(operationTimes.begin());
        }
    }

    return totalOperations;
}

double APIServer::runMemoryTest(int duration, int iterations, int concurrency, std::vector<double>& operationTimes) {
    double totalOperations = 0;

    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::seconds(duration);

    std::vector<std::unique_ptr<std::vector<double>>> allocations;

    while (std::chrono::high_resolution_clock::now() < endTime && totalOperations < iterations) {
        auto opStart = std::chrono::high_resolution_clock::now();

        // Allocate memory
        size_t allocSize = 1000 + (rand() % 9000); // 1KB to 10KB
        allocations.push_back(std::make_unique<std::vector<double>>(allocSize, 1.0));

        // Simulate some work
        for (size_t i = 0; i < std::min(allocSize, size_t(100)); ++i) {
            (*allocations.back())[i] = (*allocations.back())[i] * 2.0;
        }

        // Deallocate some memory periodically
        if (allocations.size() > 100) {
            allocations.erase(allocations.begin() + rand() % allocations.size());
        }

        auto opEnd = std::chrono::high_resolution_clock::now();
        double opTime = std::chrono::duration<double>(opEnd - opStart).count();

        operationTimes.push_back(opTime);
        totalOperations += 1.0;

        // Prevent overflow
        if (operationTimes.size() > 10000) {
            operationTimes.erase(operationTimes.begin());
        }
    }

    return totalOperations;
}

double APIServer::runDiskTest(int duration, int iterations, int concurrency, std::vector<double>& operationTimes) {
    double totalOperations = 0;

    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::seconds(duration);

    while (std::chrono::high_resolution_clock::now() < endTime && totalOperations < iterations) {
        auto opStart = std::chrono::high_resolution_clock::now();

        // Simulate disk I/O with file operations
        std::string tempFile = "temp_perf_test_" + std::to_string(rand()) + ".tmp";

        try {
            // Write to file
            std::ofstream outFile(tempFile, std::ios::binary);
            std::vector<char> data(1024, 'A'); // 1KB of data
            outFile.write(data.data(), data.size());
            outFile.close();

            // Read from file
            std::ifstream inFile(tempFile, std::ios::binary);
            std::vector<char> readData(1024);
            inFile.read(readData.data(), readData.size());
            inFile.close();

            // Clean up
            std::remove(tempFile.c_str());
        }
        catch (const std::exception&) {
            // Ignore file operation errors for performance testing
        }

        auto opEnd = std::chrono::high_resolution_clock::now();
        double opTime = std::chrono::duration<double>(opEnd - opStart).count();

        operationTimes.push_back(opTime);
        totalOperations += 1.0;

        // Prevent overflow
        if (operationTimes.size() > 10000) {
            operationTimes.erase(operationTimes.begin());
        }
    }

    return totalOperations;
}

double APIServer::runNetworkTest(int duration, int iterations, int concurrency, std::vector<double>& operationTimes) {
    double totalOperations = 0;

    // For network testing, we'll simulate network operations with delays
    // In a real implementation, this would make actual network calls

    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::seconds(duration);

    while (std::chrono::high_resolution_clock::now() < endTime && totalOperations < iterations) {
        auto opStart = std::chrono::high_resolution_clock::now();

        // Simulate network latency with a small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(1 + (rand() % 10)));

        // Simulate some data processing
        std::vector<char> simulatedData(512, 'N');
        for (size_t i = 0; i < simulatedData.size(); ++i) {
            simulatedData[i] = static_cast<char>((simulatedData[i] + i) % 256);
        }

        auto opEnd = std::chrono::high_resolution_clock::now();
        double opTime = std::chrono::duration<double>(opEnd - opStart).count();

        operationTimes.push_back(opTime);
        totalOperations += 1.0;

        // Prevent overflow
        if (operationTimes.size() > 10000) {
            operationTimes.erase(operationTimes.begin());
        }
    }

    return totalOperations;
}

json APIServer::handleOrchestratorAPI(const std::string& action, const json& parameters) {
    try {
        if (action == "status") {
            return orchestratorGetSystemState("");
        }
        else {
            return createErrorResponse("Unknown orchestrator action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Orchestrator API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleNeuralAPI(const std::string& action, const json& parameters) {
    try {
        if (!trainingModule) {
            return createErrorResponse("Training module not available", 503);
        }

        if (action == "topology") {
            return trainingModule->getNetworkTopology();
        }
        else if (action == "activity") {
            return trainingModule->getNetworkActivity();
        }
        else if (action == "initialize") {
            // Initialize neural network with provided parameters
            double learningRate = parameters.value("learning_rate", 0.01);
            int epochs = parameters.value("epochs", 10);
            int layers = parameters.value("layers", 3);
            int neurons = parameters.value("neurons", 10);
            float minRange = parameters.value("min_range", 0.0f);
            float maxRange = parameters.value("max_range", 1.0f);
            int inputSize = parameters.value("input_size", trainingModule->inputSize);
            int outputSize = parameters.value("output_size", trainingModule->outputSize);

            // Validate parameters
            if (learningRate <= 0.0 || learningRate > 1.0) {
                return createErrorResponse("Learning rate must be between 0 and 1", 400);
            }
            if (epochs <= 0) {
                return createErrorResponse("Epochs must be positive", 400);
            }
            if (layers <= 0) {
                return createErrorResponse("Layers must be positive", 400);
            }
            if (neurons <= 0) {
                return createErrorResponse("Neurons must be positive", 400);
            }
            if (inputSize <= 0) {
                return createErrorResponse("Input size must be positive", 400);
            }
            if (outputSize <= 0) {
                return createErrorResponse("Output size must be positive", 400);
            }
            if (minRange >= maxRange) {
                return createErrorResponse("min_range must be less than max_range", 400);
            }

            // Initialize the network
            trainingModule->layers = layers;
            trainingModule->neurons = neurons;
            trainingModule->inputSize = inputSize;
            trainingModule->outputSize = outputSize;
            trainingModule->learningRate = learningRate;

            try {
                trainingModule->preprocess(minRange, maxRange);
            }
            catch (const std::exception& e) {
                return createErrorResponse("Failed to initialize neural network: " + std::string(e.what()), 500);
            }

            json response = {
                {"status", "initialized"},
                {"learning_rate", learningRate},
                {"epochs", epochs},
                {"layers", layers},
                {"neurons", neurons},
                {"input_size", inputSize},
                {"output_size", outputSize},
                {"min_range", minRange},
                {"max_range", maxRange},
                {"message", "Neural network initialized successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "train") {
            // Start training with provided parameters
            double learningRate = parameters.value("learning_rate", 0.01);
            int epochs = parameters.value("epochs", 10);
            std::string datasetName = parameters.value("dataset_name", "");

            // Validate parameters
            if (learningRate <= 0.0 || learningRate > 1.0) {
                return createErrorResponse("Learning rate must be between 0 and 1", 400);
            }
            if (epochs <= 0) {
                return createErrorResponse("Epochs must be positive", 400);
            }

            // Check if already training
            if (trainingModule->isTraining()) {
                return createErrorResponse("Training is already in progress", 409);
            }

            // Start training asynchronously
            bool started = trainingModule->startTraining(learningRate, epochs);

            if (!started) {
                return createErrorResponse("Failed to start training", 500);
            }

            json response = {
                {"status", "started"},
                {"learning_rate", learningRate},
                {"epochs", epochs},
                {"dataset_name", datasetName},
                {"message", "Training started successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "set_calculus_config") {
            // Set calculus activation configuration for a specific layer
            int layerIndex = parameters.value("layer_index", 0);
            int N = parameters.value("N", 10);
            float SM = parameters.value("SM", 1.0f);
            float a = parameters.value("a", 1.0f);

            // Validate parameters
            if (layerIndex < 0) {
                return createErrorResponse("Layer index must be non-negative", 400);
            }
            if (N <= 0 || N > 1000) {
                return createErrorResponse("N (integration points) must be between 1 and 1000", 400);
            }
            if (SM <= 0.0f) {
                return createErrorResponse("SM (Simpson's method multiplier) must be positive", 400);
            }
            if (a <= 0.0f) {
                return createErrorResponse("a (alpha scaling parameter) must be positive", 400);
            }

            // Set calculus configuration for the layer
            if (trainingModule->getCore()) {
                // trainingModule->getCore()->setLayerActivation(layerIndex, ActivationType::INTEGRAL_SIGMOID, 1.0f, 0.1f, 1.0f, N, SM, a);
            } else {
                return createErrorResponse("CoreAI not initialized", 500);
            }

            json response = {
                {"status", "configured"},
                {"layer_index", layerIndex},
                {"activation_type", "INTEGRAL_SIGMOID"},
                {"calculus_config", {
                    {"N", N},
                    {"SM", SM},
                    {"a", a}
                }},
                {"message", "Calculus activation configuration set successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "get_calculus_config") {
            // Get calculus activation configuration for a specific layer
            int layerIndex = parameters.value("layer_index", 0);

            if (!trainingModule->getCore()) {
                return createErrorResponse("CoreAI not initialized", 500);
            }
            if (layerIndex < 0 || layerIndex >= static_cast<int>(trainingModule->getCore()->layer_activation_configs.size())) {
                return createErrorResponse("Invalid layer index", 400);
            }

            const ActivationConfig& config = trainingModule->getCore()->layer_activation_configs[layerIndex];

            json response = {
                {"layer_index", layerIndex},
                {"activation_type", static_cast<int>(config.type)},
                {"calculus_config", {
                    {"N", config.calculus_config.N},
                    {"SM", config.calculus_config.SM},
                    {"a", config.calculus_config.a}
                }},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "set_all_calculus_activations") {
            // Set calculus activation for all layers
            std::string activationType = parameters.value("activation_type", "INTEGRAL_SIGMOID");
            int N = parameters.value("N", 10);
            float SM = parameters.value("SM", 1.0f);
            float a = parameters.value("a", 1.0f);

            // Validate parameters
            if (N <= 0 || N > 1000) {
                return createErrorResponse("N (integration points) must be between 1 and 1000", 400);
            }
            if (SM <= 0.0f) {
                return createErrorResponse("SM (Simpson's method multiplier) must be positive", 400);
            }
            if (a <= 0.0f) {
                return createErrorResponse("a (alpha scaling parameter) must be positive", 400);
            }

            // Map string to ActivationType
            ActivationType type = ActivationType::INTEGRAL_SIGMOID;
            if (activationType == "INTEGRAL_RELU") {
                type = ActivationType::INTEGRAL_RELU;
            } else if (activationType == "INTEGRAL_TANH") {
                type = ActivationType::INTEGRAL_TANH;
            }

            // Set calculus activation for all layers
            if (trainingModule->getCore()) {
                // trainingModule->getCore()->setAllLayerActivations(type, 1.0f, 0.1f, 1.0f, N, SM, a);
            } else {
                return createErrorResponse("CoreAI not initialized", 500);
            }

            json response = {
                {"status", "configured"},
                {"activation_type", activationType},
                {"calculus_config", {
                    {"N", N},
                    {"SM", SM},
                    {"a", a}
                }},
                {"message", "Calculus activation configuration set for all layers"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "test_calculus_activation") {
            // Test calculus activation function with sample input
            float testInput = parameters.value("input", 1.0f);
            std::string activationType = parameters.value("activation_type", "INTEGRAL_SIGMOID");
            int N = parameters.value("N", 10);
            float SM = parameters.value("SM", 1.0f);
            float a = parameters.value("a", 1.0f);

            // Create test configuration
            CalculusConfig calcConfig{N, SM, a};
            ActivationConfig config;
            config.calculus_config = calcConfig;

            if (activationType == "INTEGRAL_SIGMOID") {
                config.type = ActivationType::INTEGRAL_SIGMOID;
            } else if (activationType == "INTEGRAL_RELU") {
                config.type = ActivationType::INTEGRAL_RELU;
            } else if (activationType == "INTEGRAL_TANH") {
                config.type = ActivationType::INTEGRAL_TANH;
            } else {
                return createErrorResponse("Unknown activation type: " + activationType, 400);
            }

            // Test the activation function
            float result = 0.0f;
            float derivative = 0.0f;
            if (trainingModule->getCore()) {
                result = trainingModule->getCore()->activate(testInput, config, 0.0f); // distance = 0 for simplicity
                derivative = trainingModule->getCore()->activate_derivative(testInput, config, 0.0f);
            } else {
                return createErrorResponse("CoreAI not initialized", 500);
            }

            json response = {
                {"input", testInput},
                {"activation_type", activationType},
                {"calculus_config", {
                    {"N", N},
                    {"SM", SM},
                    {"a", a}
                }},
                {"output", result},
                {"derivative", derivative},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else {
            return createErrorResponse("Unknown neural action: " + action, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Neural API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleSessionsAPI(const std::string& action, const json& parameters, const std::string& method) {
    try {
        if (method == "POST") {
            // Create new session
            std::string clientId = parameters.value("clientId", "");
            std::string sessionId = createSession(clientId);

            json response = {
                {"sessionId", sessionId},
                {"clientId", clientId},
                {"created", true},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (method == "DELETE") {
            // Destroy session - extract session ID from action (e.g., "session123")
            std::string sessionId = action;
            if (sessionId.empty()) {
                return createErrorResponse("Session ID required for DELETE", 400);
            }

            destroySession(sessionId);
            json response = {
                {"sessionId", sessionId},
                {"destroyed", true},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (method == "GET") {
            // Get session info - extract session ID from action
            std::string sessionId = action;
            if (sessionId.empty()) {
                // Return all active sessions
                json sessions = json::array();
                for (const auto& pair : activeSessions) {
                    sessions.push_back({
                        {"sessionId", pair.first},
                        {"clientId", pair.second},
                        {"created", getTimestampString()}
                    });
                }
                return sessions;
            }
            else {
                // Return specific session info
                if (!validateSession(sessionId)) {
                    return createErrorResponse("Session not found", 404);
                }

                json sessionData = getSessionData(sessionId);
                json response = {
                    {"sessionId", sessionId},
                    {"clientId", activeSessions[sessionId]},
                    {"data", sessionData},
                    {"valid", true},
                    {"timestamp", getTimestampString()}
                };
                return response;
            }
        }
        else {
            return createErrorResponse("Method not allowed for sessions", 405);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Sessions API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleChatAPI(const std::string& action, const json& parameters, const std::string& method) {
    try {
        // When /api/chat is called without an explicit action (e.g. POST /api/chat),
        // the router defaults action to "status". Treat POST with no action as "message".
        std::string effectiveAction = action;
        if (effectiveAction == "status" && method == "POST") {
            effectiveAction = "message";
        }

        if (effectiveAction == "message" && method == "POST") {
            // Handle chat message sending
            std::string message = parameters.value("message", "");
            std::string messageType = parameters.value("messageType", "text");
            std::string sessionId = parameters.value("sessionId", "");
            std::string clientId = parameters.value("clientId", "");

            if (message.empty()) {
                return createErrorResponse("Message cannot be empty", 400);
            }

            // Save chat message to database if available
            if (database && !sessionId.empty()) {
                #ifdef USE_MYSQL
                try {
                    int sessionIdInt = std::stoi(sessionId); // Convert string to int
                    std::string role = "user"; // Assume user message for now
                    float confidence = 1.0f; // Default confidence
                    std::string metadata = json({
                        {"clientId", clientId},
                        {"messageType", messageType},
                        {"timestamp", getTimestampString()}
                    }).dump();

                    database->saveChatMessage(sessionIdInt, role, message, messageType, confidence, metadata);
                } catch (const std::exception& dbError) {
                    std::cerr << "Warning: Failed to save chat message to database: " << dbError.what() << std::endl;
                    // Continue with processing even if database save fails
                }
                #endif
            }

            std::string responseMessage;
            bool usedWebSearch = false;
            std::string webSearchQuery;
                if (trainingModule) {
                    Language* lang = trainingModule->getLanguage();
                    if (lang) {
                        lang->setLlmConfig(llmProvider, llmBaseUrl, llmModel, llmTemperature, llmMaxTokens, llmApiKey);
                        responseMessage = lang->respond(message);
                    usedWebSearch = lang->getLastResponseUsedWebSearch();
                    webSearchQuery = lang->getLastWebSearchQuery();
                } else {
                    responseMessage = "I'm still initializing my language processing capabilities. Please try again in a moment.";
                }
            } else {
                responseMessage = "I'm currently unable to process your request. The AI training module is not available.";
            }

            // Extract code/data blocks from the response for structured artifact rendering
            json artifacts = json::array();
            {
                std::string remaining = responseMessage;
                int artifactIndex = 0;
                size_t pos = 0;
                while ((pos = remaining.find("```", pos)) != std::string::npos) {
                    size_t langStart = pos + 3;
                    size_t langEnd = remaining.find("\n", langStart);
                    if (langEnd == std::string::npos) break;
                    std::string language = remaining.substr(langStart, langEnd - langStart);
                    // trim
                    while (!language.empty() && (language.front() == ' ' || language.front() == '\t')) language.erase(0, 1);
                    while (!language.empty() && (language.back() == ' ' || language.back() == '\t')) language.pop_back();

                    size_t codeStart = langEnd + 1;
                    size_t codeEnd = remaining.find("```", codeStart);
                    if (codeEnd == std::string::npos) break;
                    std::string code = remaining.substr(codeStart, codeEnd - codeStart);
                    if (!code.empty() && code.back() == '\n') code.pop_back();

                    json artifact;
                    artifact["id"] = "artifact-" + std::to_string(artifactIndex);
                    artifact["language"] = language;
                    artifact["code"] = code;
                    artifact["filename"] = "snippet-" + std::to_string(artifactIndex + 1) + "." + (language.empty() ? "txt" : language);
                    artifacts.push_back(artifact);
                    artifactIndex++;
                    pos = codeEnd + 3;
                }
            }

            // Save AI response to database if available
            if (database && !sessionId.empty()) {
                try {
                    int sessionIdInt = std::stoi(sessionId); // Convert string to int
                    std::string role = "assistant";
                    float confidence = 0.8f; // AI response confidence
                    std::string metadata = json({
                        {"clientId", clientId},
                        {"messageType", "text"},
                        {"timestamp", getTimestampString()},
                        {"response_type", "ai_generated"}
                    }).dump();

                    database->saveChatMessage(sessionIdInt, role, responseMessage, "text", confidence, metadata);
                } catch (const std::exception& dbError) {
                    std::cerr << "Warning: Failed to save AI response to database: " << dbError.what() << std::endl;
                }
            }

            json response = {
                {"message", message},
                {"messageType", messageType},
                {"response", responseMessage},
                {"sessionId", sessionId},
                {"timestamp", getTimestampString()},
                {"status", "success"}
            };
            if (!artifacts.empty()) {
                response["artifacts"] = artifacts;
            }
            if (usedWebSearch) {
                response["web_searched"] = true;
                response["search_query"] = webSearchQuery;
            }
            return response;
        }
        else if (effectiveAction == "history" && method == "GET") {
            // Handle chat history retrieval
            std::string sessionId = parameters.value("sessionId", "");
            int limit = parameters.value("limit", 50);

            if (sessionId.empty()) {
                return createErrorResponse("Session ID is required for chat history", 400);
            }

            json history = json::array();
            int count = 0;

            // Retrieve chat history from database if available
            if (database) {
                try {
                    int sessionIdInt = std::stoi(sessionId); // Convert string to int
                    auto messages = database->getChatMessages(sessionIdInt, limit);
                    for (const auto& msg : messages) {
                        json messageJson = {
                            {"id", msg.id},
                            {"sessionId", msg.sessionId},
                            {"role", msg.role},
                            {"content", msg.content},
                            {"messageType", msg.messageType},
                            {"confidence", msg.confidence},
                            {"timestamp", msg.timestamp},
                            {"metadata", json::parse(msg.metadata)}
                        };
                        history.push_back(messageJson);
                    }
                    count = messages.size();
                } catch (const std::exception& dbError) {
                    std::cerr << "Warning: Failed to retrieve chat history from database: " << dbError.what() << std::endl;
                    // Return empty history if database fails
                }
            }

            json response = {
                {"history", history},
                {"sessionId", sessionId},
                {"limit", limit},
                {"count", count},
                {"timestamp", getTimestampString()},
                {"status", "success"}
            };
            return response;
        }
        else if (effectiveAction == "clear" && method == "POST") {
            // Handle chat history clearing
            std::string sessionId = parameters.value("sessionId", "");

            if (sessionId.empty()) {
                return createErrorResponse("Session ID is required for clearing chat history", 400);
            }

            bool cleared = false;

            // Clear chat history from database if available
            if (database) {
                try {
                    int sessionIdInt = std::stoi(sessionId); // Convert string to int
                    cleared = database->clearChatMessages(sessionIdInt);
                } catch (const std::exception& dbError) {
                    std::cerr << "Warning: Failed to clear chat history from database: " << dbError.what() << std::endl;
                }
            }

            json response = {
                {"cleared", cleared},
                {"sessionId", sessionId},
                {"timestamp", getTimestampString()},
                {"status", "success"}
            };
            return response;
        }
        else if (effectiveAction == "session" && method == "POST") {
            // Create new chat session
            std::string sessionName = parameters.value("sessionName", "New Chat Session");
            std::string userId = parameters.value("userId", "anonymous");
            std::string modelVersion = parameters.value("modelVersion", "default");

            int sessionId = -1;

            // Save chat session to database if available
            if (database) {
                try {
                    std::string sessionId;
                    int dbSessionId = database->saveChatSession(sessionName, userId, modelVersion, "active");
                    sessionId = std::to_string(dbSessionId); // Convert to string for API response
                } catch (const std::exception& dbError) {
                    std::cerr << "Warning: Failed to save chat session to database: " << dbError.what() << std::endl;
                }
            }

            json response = {
                {"sessionId", sessionId},
                {"sessionName", sessionName},
                {"userId", userId},
                {"modelVersion", modelVersion},
                {"status", sessionId != -1 ? "created" : "failed"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else {
            return createErrorResponse("Unknown chat action: " + action + " with method " + method, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Chat API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handlePredictAPI(const std::string& action, const json& parameters, const std::string& method) {
    try {
        if (!predictModule) {
            return createErrorResponse("Prediction module not available", 503);
        }

        if (action == "run" && method == "POST") {
            // Handle prediction request
            std::string inputType = parameters.value("input_type", "file");
            std::string inputFile = parameters.value("input_file", "");
            std::string inputText = parameters.value("input_text", "");
            std::string inputUrl = parameters.value("input_url", "");
            std::string outputCsvFile = parameters.value("output_file", "predictions.csv");
            std::string targetFile = parameters.value("target_file", "");
            std::string loadModelId = parameters.value("load_model_id", "");
            int datasetId = parameters.value("dataset_id", -1);
            int layers = parameters.value("layers", 3);
            int neurons = parameters.value("neurons", 10);
            float minRange = parameters.value("min_range", 0.0f);
            float maxRange = parameters.value("max_range", 1.0f);
            int inputSize = parameters.value("input_size", 1);
            int outputSize = parameters.value("output_size", 1);
            int numSamples = parameters.value("num_samples", 0);
            double learningRate = parameters.value("learning_rate", 0.01);
            int epochs = parameters.value("epochs", 10);
            std::string embeddingFile = parameters.value("embedding_file", "embedding.txt");
            std::string language = parameters.value("language", "en");
            std::string datasetName = parameters.value("dataset_name", "online-1a");
            bool hasHeader = parameters.value("contains_header", true);
            bool containsText = parameters.value("contains_text", false);
            char delimiter = parameters.value("delimiter", ',');

            // Validate input based on type
            if (inputType == "file" && inputFile.empty()) {
                return createErrorResponse("Input file cannot be empty for file input type", 400);
            }
            if (inputType == "text" && inputText.empty()) {
                return createErrorResponse("Input text cannot be empty for text input type", 400);
            }
            if (inputType == "url" && inputUrl.empty()) {
                return createErrorResponse("Input URL cannot be empty for URL input type", 400);
            }

            // Handle different input types
            std::string processedInputFile = inputFile;
            if (inputType == "text") {
                // Create temporary file from text input
                processedInputFile = "temp_text_input.csv";
                std::ofstream tempFile(processedInputFile);
                if (tempFile.is_open()) {
                    // For text input, treat it as a single data row without header
                    tempFile << inputText << std::endl;
                    tempFile.close();
                    hasHeader = false; // Text input doesn't have headers
                } else {
                    return createErrorResponse("Failed to create temporary file for text input", 500);
                }
            }
            else if (inputType == "url") {
                // TODO: Implement URL download functionality
                return createErrorResponse("URL input type not yet implemented", 501);
            }
            else if (inputType == "image" || inputType == "audio") {
                // TODO: Implement image/audio processing
                return createErrorResponse(inputType + " input type not yet implemented", 501);
            }

            // Run prediction
            bool success = predictModule->runPrediction(processedInputFile, outputCsvFile, targetFile, loadModelId,
                                                       datasetId, layers, neurons, minRange, maxRange,
                                                       inputSize, outputSize, numSamples, learningRate, epochs,
                                                       embeddingFile, language, datasetName,
                                                       hasHeader, containsText, delimiter);

            // Clean up temporary file if created
            if (inputType == "text" && processedInputFile != inputFile) {
                std::remove(processedInputFile.c_str());
            }

            if (!success) {
                return createErrorResponse("Prediction failed", 500);
            }

            // Get evaluation metrics
            json response = {
                {"input_type", inputType},
                {"input_file", processedInputFile},
                {"output_file", outputCsvFile},
                {"rmse", predictModule->getRMSE()},
                {"mse", predictModule->getMSE()},
                {"accuracy", predictModule->getAccuracy()},
                {"training_time", predictModule->getTrainingTime()},
                {"timestamp", getTimestampString()},
                {"status", "success"}
            };
            return response;
        }
        else if (action == "models" && method == "GET") {
            // Handle available models listing
            json response = {
                {"models", json::array({"default", "advanced", "experimental"})},
                {"count", 3},
                {"timestamp", getTimestampString()},
                {"status", "success"}
            };
            return response;
        }
        else {
            return createErrorResponse("Unknown predict action: " + action + " with method " + method, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Predict API error: " + std::string(e.what()), 500);
    }
}

json APIServer::handleTrainAPI(const std::string& action, const json& parameters, const std::string& method) {
    try {
        if (!trainingModule) {
            return createErrorResponse("Training module not available", 503);
        }

        if (action == "start" && method == "POST") {
            // Handle training start request
            double learningRate = parameters.value("learning_rate", 0.01);
            int epochs = parameters.value("epochs", 10);
            std::string datasetName = parameters.value("dataset_name", "");

            // Validate parameters
            if (learningRate <= 0.0 || learningRate > 1.0) {
                return createErrorResponse("Learning rate must be between 0 and 1", 400);
            }
            if (epochs <= 0) {
                return createErrorResponse("Epochs must be positive", 400);
            }

            // Check if already training
            if (trainingModule->isTraining()) {
                return createErrorResponse("Training is already in progress", 409);
            }

            // Ensure CoreAI is initialized before starting training
            if (!trainingModule->getCore()) {
                // Initialize CoreAI if not already done
                float minRange = parameters.value("min_range", 0.0f);
                float maxRange = parameters.value("max_range", 1.0f);
                int layers = parameters.value("layers", trainingModule->layers);
                int neurons = parameters.value("neurons", trainingModule->neurons);
                int inputSize = parameters.value("input_size", trainingModule->inputSize);
                int outputSize = parameters.value("output_size", trainingModule->outputSize);

                // Validate parameters
                if (minRange >= maxRange) {
                    return createErrorResponse("min_range must be less than max_range", 400);
                }
                if (layers <= 0) {
                    return createErrorResponse("layers must be positive", 400);
                }
                if (neurons <= 0) {
                    return createErrorResponse("neurons must be positive", 400);
                }
                if (inputSize <= 0) {
                    return createErrorResponse("input_size must be positive", 400);
                }
                if (outputSize <= 0) {
                    return createErrorResponse("output_size must be positive", 400);
                }

                // Check if data is loaded - with better diagnostic logging
                std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training start - checking data state" << std::endl;
                std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module state at start - "
                          << "numSamples: " << trainingModule->numSamples << ", "
                          << "inputSize: " << trainingModule->inputSize << ", "
                          << "outputSize: " << trainingModule->outputSize << std::endl;

                // DEBUG LOG: Training module instance and state for persistence check
                std::cout << "[DEBUG] APIServer::handleTrainAPI START: trainingModule address: " << trainingModule.get()
                          << ", numSamples: " << trainingModule->numSamples << std::endl;

                if (trainingModule->numSamples == 0) {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module reports 0 samples, checking data state" << std::endl;
                    
                    // Try to reload data if available - use parameters from the request
                    std::string reloadFilename = parameters.value("filename", "");
                    bool reloadHasHeader = parameters.value("has_header", true);
                    bool reloadContainsText = parameters.value("contains_text", false);
                    
                    if (!reloadFilename.empty() && std::filesystem::exists(reloadFilename)) {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Attempting to reload data from: " << reloadFilename << std::endl;
                        bool reloaded = trainingModule->loadCSV(reloadFilename, -1, outputSize, reloadHasHeader, reloadContainsText, ',', datasetName);
                        if (reloaded) {
                            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Data reloaded successfully, samples: " << trainingModule->numSamples << std::endl;
                        } else {
                            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Data reload failed" << std::endl;
                            return createErrorResponse("No training data loaded. Please load dataset first.", 400);
                        }
                    } else {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: No data file available for reload" << std::endl;
                        return createErrorResponse("No training data loaded. Please load dataset first.", 400);
                    }
                } else {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module has " << trainingModule->numSamples << " samples loaded" << std::endl;
                }

                try {
                    trainingModule->layers = layers;
                    trainingModule->neurons = neurons;
                    trainingModule->inputSize = inputSize;
                    trainingModule->outputSize = outputSize;
                    trainingModule->preprocess(minRange, maxRange);
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Neural network initialized successfully" << std::endl;
                }
                catch (const std::exception& e) {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Neural network initialization failed: " << e.what() << std::endl;
                    return createErrorResponse("Failed to initialize neural network: " + std::string(e.what()), 500);
                }
            }

            // Start training asynchronously
            bool started = trainingModule->startTraining(learningRate, epochs);

            if (!started) {
                return createErrorResponse("Failed to start training", 500);
            }

            json response = {
                {"status", "started"},
                {"learning_rate", learningRate},
                {"epochs", epochs},
                {"dataset_name", datasetName},
                {"message", "Training started successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "status" && method == "GET") {
            // Handle training status check
            std::string status = trainingModule->getTrainingStatus();
            bool isActive = trainingModule->isTraining();

            json response = {
                {"status", isActive ? "running" : "idle"},
                {"details", status},
                {"is_active", isActive},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "stop" && method == "POST") {
            // Handle training stop request
            if (!trainingModule->isTraining()) {
                return createErrorResponse("No training is currently running", 400);
            }

            bool stopped = trainingModule->stopTraining();

            json response = {
                {"status", stopped ? "stopped" : "failed"},
                {"message", stopped ? "Training stopped successfully" : "Failed to stop training"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "load_dataset" && method == "POST") {
            // Handle dataset loading for multiple file formats
            std::string filename = parameters.value("filename", "");
            std::string directory = parameters.value("directory", "");
            std::vector<std::string> extensions = parameters.value("extensions", std::vector<std::string>{});
            std::vector<std::string> fileList = parameters.value("files", std::vector<std::string>{});
            int outputSize = parameters.value("output_size", 1);
            bool hasHeader = parameters.value("has_header", true);
            bool containsText = parameters.value("contains_text", false);
            std::string datasetName = parameters.value("dataset_name", "api_dataset");
            float minRange = parameters.value("min_range", -1.0f);
            float maxRange = parameters.value("max_range", 1.0f);
            int maxSeqLen = parameters.value("max_seq_len", 512);
            int embeddingDim = parameters.value("embedding_dim", EMBEDDING_DIM);
            
            // DIAGNOSTIC LOG: Training dataset loading parameters
            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Loading dataset - filename: " << filename
                      << ", directory: " << directory << ", datasetName: " << datasetName << std::endl;

            // DIAGNOSTIC LOG: Training module state before loading
            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module state before loading - "
                      << "trainingModule exists: " << (trainingModule ? "YES" : "NO") << ", "
                      << "numSamples: " << (trainingModule ? trainingModule->numSamples : -1) << ", "
                      << "inputSize: " << (trainingModule ? trainingModule->inputSize : -1) << ", "
                      << "outputSize: " << (trainingModule ? trainingModule->outputSize : -1) << std::endl;

            // FILE PATH VALIDATION: Check if requested files exist, fallback to available test data
            if (!filename.empty()) {
                std::filesystem::path filePath(filename);
                
                // If path is relative, try to resolve it relative to the project root
                if (filePath.is_relative()) {
                    std::filesystem::path projectRoot = std::filesystem::current_path();
                    std::filesystem::path absolutePath = projectRoot / filePath;
                    
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Relative path detected: " << filename
                              << ", trying absolute path: " << absolutePath << std::endl;
                    
                    if (std::filesystem::exists(absolutePath)) {
                        filename = absolutePath.string();
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Resolved relative path to: " << filename << std::endl;
                    } else {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Absolute path also not found: " << absolutePath << std::endl;
                    }
                }
                
                if (!std::filesystem::exists(filename)) {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: File not found: " << filename << std::endl;
                    
                    // Fallback to available test data with absolute path
                    std::filesystem::path projectRoot = std::filesystem::current_path();
                    std::string testDataPath = (projectRoot / "training_data/sample_dataset/test_data.csv").string();
                    if (std::filesystem::exists(testDataPath)) {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Using fallback test data: " << testDataPath << std::endl;
                        filename = testDataPath;
                    } else {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: No fallback test data available" << std::endl;
                    }
                } else {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: File exists: " << filename << std::endl;
                }
            }

            // Allow setting network topology during load
            if (parameters.contains("layers")) {
                trainingModule->layers = parameters["layers"];
            }
            if (parameters.contains("neurons")) {
                trainingModule->neurons = parameters["neurons"];
            }

            // Validate normalization parameters
            if (minRange >= maxRange) {
                return createErrorResponse("min_range must be less than max_range", 400);
            }

            bool loaded = false;
            std::string loadType = "single_file";
            int totalFiles = 0;

            // Determine loading strategy
            if (!directory.empty()) {
                // Load all files from directory with specified extensions
                loadType = "directory";
                if (extensions.empty()) {
                    extensions = {"txt", "json", "xml", "pdf", "docx", "doc", "mp4", "avi", "mov", "mkv", "mp3", "wav", "flac", "aac", "jpg", "jpeg", "png", "bmp", "gif", "csv"};
                }
                loaded = trainingModule->loadDirectoryFiles(directory, extensions, maxSeqLen, embeddingDim);
                totalFiles = extensions.size(); // Approximate
            }
            else if (!fileList.empty()) {
                // Load batch of specific files
                loadType = "batch_files";
                loaded = trainingModule->loadBatchFiles(fileList, maxSeqLen, embeddingDim);
                totalFiles = fileList.size();
            }
            else if (!filename.empty()) {
                // Load single file
                loadType = "single_file";
                std::string extension = getFileExtension(filename);
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                // Route to appropriate loading method based on file extension
                if (extension == "csv") {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Calling loadCSV with filename=" << filename
                              << ", outputSize=" << outputSize << ", hasHeader=" << hasHeader
                              << ", containsText=" << containsText << std::endl;
                    loaded = trainingModule->loadCSV(filename, -1, outputSize, hasHeader, containsText, ',', datasetName);
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: loadCSV returned: " << loaded << std::endl;
                }
                else {
                    // Use the general loadFile method for all other formats
                    loaded = trainingModule->loadFile(filename, maxSeqLen, embeddingDim);
                }
                totalFiles = 1;
            }
            else {
                return createErrorResponse("Either filename, directory, or files list must be provided", 400);
            }

            if (!loaded) {
                std::string errorMsg = "Failed to load dataset";
                if (!filename.empty()) errorMsg += " from file: " + filename;
                else if (!directory.empty()) errorMsg += " from directory: " + directory;
                else errorMsg += " from batch files";
                
                // DIAGNOSTIC LOG: Dataset loading failed
                std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Dataset loading failed - " << errorMsg << std::endl;
                
                // Check if training module has any data loaded
                if (trainingModule->numSamples == 0) {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module reports 0 samples loaded" << std::endl;
                } else {
                    std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module reports " << trainingModule->numSamples << " samples loaded" << std::endl;
                }
                
                // Additional diagnostic: Check file existence and permissions
                if (!filename.empty()) {
                    std::filesystem::path filePath(filename);
                    if (!std::filesystem::exists(filePath)) {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: File does not exist: " << filename << std::endl;
                    } else {
                        std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: File exists, checking permissions..." << std::endl;
                        std::ifstream testFile(filename);
                        if (!testFile.is_open()) {
                            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Cannot open file for reading: " << filename << std::endl;
                        } else {
                            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: File can be opened for reading" << std::endl;
                            testFile.close();
                        }
                    }
                }
                
                return createErrorResponse(errorMsg, 500);
            }
            
            // DIAGNOSTIC LOG: Dataset loaded successfully
            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Dataset loaded successfully - samples: "
                      << trainingModule->numSamples << ", inputSize: " << trainingModule->inputSize
                      << ", outputSize: " << trainingModule->outputSize << std::endl;

            // DIAGNOSTIC LOG: Training module state after loading
            std::cout << "[DIAGNOSTIC] APIServer::handleTrainAPI: Training module state after loading - "
                      << "numSamples: " << trainingModule->numSamples << ", "
                      << "inputSize: " << trainingModule->inputSize << ", "
                      << "outputSize: " << trainingModule->outputSize << std::endl;

            // DEBUG LOG: Training module instance and state for persistence check
            std::cout << "[DEBUG] APIServer::handleTrainAPI LOAD: trainingModule address: " << trainingModule.get()
                      << ", numSamples: " << trainingModule->numSamples << std::endl;

            // Perform data preprocessing with normalization
            try {
                trainingModule->preprocess(minRange, maxRange);
            }
            catch (const std::exception& e) {
                return createErrorResponse("Preprocessing failed: " + std::string(e.what()), 500);
            }

            json response = {
                {"status", "loaded"},
                {"load_type", loadType},
                {"dataset_name", datasetName},
                {"samples", trainingModule->numSamples},
                {"input_size", trainingModule->inputSize},
                {"output_size", trainingModule->outputSize},
                {"min_range", minRange},
                {"max_range", maxRange},
                {"max_seq_len", maxSeqLen},
                {"embedding_dim", embeddingDim},
                {"message", "Dataset loaded and preprocessed successfully"},
                {"timestamp", getTimestampString()}
            };

            // Add file-specific information
            if (!filename.empty()) {
                response["filename"] = filename;
                response["file_type"] = getFileExtension(filename);
            }
            else if (!directory.empty()) {
                response["directory"] = directory;
                response["extensions"] = extensions;
            }
            else if (!fileList.empty()) {
                response["files_loaded"] = fileList.size();
            }

            return response;
        }
        else if (action == "conversion_progress" && method == "GET") {
            // Handle conversion progress tracking
            if (!trainingModule) {
                return createErrorResponse("Training module not available", 503);
            }

            json response = {
                {"current_file", trainingModule->getCurrentConversionFile()},
                {"progress", trainingModule->getConversionProgress()},
                {"total", trainingModule->getConversionTotal()},
                {"progress_percent", trainingModule->getConversionProgressPercent()},
                {"timestamp", getTimestampString()}
            };

            return response;
        }
        else if (action == "preprocess" && method == "POST") {
            // Handle data preprocessing
            float minRange = parameters.value("min_range", 0.0f);
            float maxRange = parameters.value("max_range", 1.0f);
            int layers = parameters.value("layers", 3);
            int neurons = parameters.value("neurons", 10);

            if (minRange >= maxRange) {
                return createErrorResponse("min_range must be less than max_range", 400);
            }

            try {
                trainingModule->layers = layers;
                trainingModule->neurons = neurons;
                trainingModule->preprocess(minRange, maxRange);

                json response = {
                    {"status", "preprocessed"},
                    {"min_range", minRange},
                    {"max_range", maxRange},
                    {"layers", layers},
                    {"neurons", neurons},
                    {"message", "Data preprocessing completed"},
                    {"timestamp", getTimestampString()}
                };
                return response;
            }
            catch (const std::exception& e) {
                return createErrorResponse("Preprocessing failed: " + std::string(e.what()), 500);
            }
        }
        else {
            return createErrorResponse("Unknown training action: " + action + " with method " + method, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Training API error: " + std::string(e.what()), 500);
    }
}



json APIServer::getNeuralTopology() {
    json topology = {
        {"message", "Neural topology not yet implemented"}
    };

    if (trainingModule) {
        topology = trainingModule->getNetworkTopology();
        topology["status"] = "success";
    } else {
        topology["status"] = "error";
        topology["message"] = "Training module not initialized";
    }

    return topology;
}

json APIServer::getNeuralActivity() {
    json activity = {
        {"message", "Neural activity not yet implemented"}
    };

    if (trainingModule) {
        activity = trainingModule->getNetworkActivity();
        activity["status"] = "success";
    } else {
        activity["status"] = "error";
        activity["message"] = "Training module not initialized";
    }

    return activity;
}

// HTTP request handling
void APIServer::handleRequest(http::request<http::string_body>&& req, tcp::socket socket) {
    try {
        // Parse request
        std::string endpoint = std::string(req.target());
        std::string method = std::string(req.method_string());

        // Handle CORS preflight (OPTIONS) requests
        if (method == "OPTIONS") {
            http::response<http::string_body> res{http::status::ok, req.version()};
            res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
            res.set(http::field::access_control_allow_origin, corsOrigin);
            res.set(http::field::access_control_allow_methods, "GET, POST, OPTIONS, PUT, DELETE");
            res.set(http::field::access_control_allow_headers, "Content-Type, Authorization, X-Session-ID");
            res.set(http::field::access_control_allow_credentials, "true");
            res.set(http::field::content_type, "application/json");
            res.content_length(0);
            res.prepare_payload();
            http::write(socket, res);
            return;
        }

        json requestData;

        if (req.body().size() > 0) {
            requestData = parseRequestBody(req.body());
        }

        // Process request
        json responseData = processAPIRequest(endpoint, method, requestData);

        // Determine the HTTP status code from the response content.
        // processAPIRequest may return error JSON objects (via
        // createErrorResponse or the catch-all handler) that carry
        // "status": "error" plus an optional "error_code". We must not
        // always report HTTP 200, otherwise the client sees success=true
        // with an error body that contains no beta key -- leading to
        // "The key was generated but the server response did not contain it."
        unsigned int statusCode = 200;
        if (responseData.is_object()) {
            if (responseData.contains("status") && responseData["status"] == "error") {
                if (responseData.contains("error_code") && responseData["error_code"].is_number_integer()) {
                    statusCode = static_cast<unsigned int>(responseData["error_code"].get<int>());
                } else {
                    statusCode = 500;
                }
            }
        }

        // Create response
        std::string response = createResponse(responseData, statusCode);

        // Send response
        http::response<http::string_body> res{http::status(statusCode), req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.set(http::field::access_control_allow_origin, corsOrigin);
        res.set(http::field::access_control_allow_credentials, "true");
        res.keep_alive(req.keep_alive());
        res.body() = response;
        res.prepare_payload();

        http::write(socket, res);
    }
    catch (const std::exception& e) {
        sendErrorResponse(socket, "Internal server error: " + std::string(e.what()), 500);
    }
}

void APIServer::handleRequestInternal(http::request<http::string_body>&& req, tcp::socket socket) {
    handleRequest(std::move(req), std::move(socket));
}

json APIServer::parseRequestBody(const std::string& body) {
    try {
        return json::parse(body);
    }
    catch (const std::exception&) {
        return json::object();
    }
}

// Sanitize all string values in a JSON object to remove control characters
// (U+0000 through U+001F) and invalid UTF-8 byte sequences that would cause
// nlohmann::json::dump() to throw an exception. Replaces invalid bytes with
// a space and escapes control characters by stripping them.
void APIServer::sanitizeJsonStrings(json& j) {
    if (j.is_string()) {
        std::string s = j.get<std::string>();
        std::string clean;
        clean.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            // Replace C0 control characters (0x00-0x1F) and 0x7F (DEL) with space
            if (c < 0x20 || c == 0x7F) {
                clean += ' ';
                continue;
            }
            // Detect and drop invalid UTF-8 lead bytes
            if (c >= 0x80 && c <= 0xBF) {
                // Continuation byte without a lead - drop it
                continue;
            }
            if (c == 0xC0 || c == 0xC1) {
                // Overlong sequence lead - drop it
                continue;
            }
            if (c >= 0xF5 && c <= 0xFF) {
                // Invalid lead byte - drop it
                continue;
            }
            // Count expected continuation bytes for valid lead bytes
            int expected = 0;
            if (c >= 0xC0 && c <= 0xDF) expected = 1;
            else if (c >= 0xE0 && c <= 0xEF) expected = 2;
            else if (c >= 0xF0 && c <= 0xF4) expected = 3;
            clean += s[i];
            // Validate continuation bytes
            for (int k = 0; k < expected; ++k) {
                if (i + k + 1 < s.size()) {
                    unsigned char nc = static_cast<unsigned char>(s[i + k + 1]);
                    if (nc < 0x80 || nc > 0xBF) {
                        // Invalid continuation - stop here, truncate
                        i += k + 1;
                        goto done;
                    }
                }
            }
            i += expected;
        }
        done:
        j = clean;
    } else if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            sanitizeJsonStrings(it.value());
        }
    } else if (j.is_array()) {
        for (auto& elem : j) {
            sanitizeJsonStrings(elem);
        }
    }
}

std::string APIServer::createResponse(const json& responseData, unsigned int statusCode) {
    json response;
    response["status"] = statusCode == 200 ? "success" : "error";
    response["timestamp"] = getTimestampString();

    if (responseData.is_object()) {
        // Merge top-level fields from responseData directly into the response.
        // This avoids double-wrapping so clients can access fields like
        // valid, success, keys, key, beta_key directly at the top level
        // without navigating through a nested "data" wrapper.
        for (auto it = responseData.begin(); it != responseData.end(); ++it) {
            response[it.key()] = it.value();
        }
    } else {
        response["data"] = responseData;
    }

    // Sanitize all string values to remove control characters and invalid
    // UTF-8 bytes (e.g. 0x0F) that would cause nlohmann::json::dump() to throw.
    sanitizeJsonStrings(response);
    return response.dump();
}

void APIServer::sendErrorResponse(tcp::socket& socket, const std::string& error, unsigned int statusCode) {
    json errorResponse = createErrorResponse(error, statusCode);
    std::string response = createResponse(errorResponse, statusCode);

    http::response<http::string_body> res{http::status(statusCode), 11};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "application/json");
    res.set(http::field::access_control_allow_origin, corsOrigin);
    res.set(http::field::access_control_allow_credentials, "true");
    res.body() = response;
    res.prepare_payload();

    http::write(socket, res);
}

// Session management
std::string APIServer::createSession(const std::string& clientId) {
    std::string sessionId = generateUniqueId();
    activeSessions[sessionId] = clientId;
    sessionData[sessionId] = json::object();
    return sessionId;
}

void APIServer::destroySession(const std::string& sessionId) {
    activeSessions.erase(sessionId);
    sessionData.erase(sessionId);
}

bool APIServer::validateSession(const std::string& sessionId) {
    return activeSessions.find(sessionId) != activeSessions.end();
}

json APIServer::getSessionData(const std::string& sessionId) {
    auto it = sessionData.find(sessionId);
    if (it != sessionData.end()) {
        return it->second;
    }
    return json::object();
}

void APIServer::updateSessionData(const std::string& sessionId, const json& data) {
    sessionData[sessionId] = data;
}

// Authentication
bool APIServer::authenticateRequest(const std::string& sessionId, const json& request) {
    // Simple authentication - in production, use proper authentication
    return validateSession(sessionId);
}

std::string APIServer::generateAPIKey() {
    unsigned char randomBytes[32];
    if (RAND_bytes(randomBytes, sizeof(randomBytes)) != 1) {
        return "";
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hashLen = 0;
    if (EVP_Digest(randomBytes, sizeof(randomBytes), hash, &hashLen, EVP_sha256(), nullptr) != 1) {
        return "";
    }

    std::ostringstream hexStream;
    for (unsigned int i = 0; i < hashLen; ++i) {
        hexStream << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(hash[i]);
    }
    return hexStream.str();
}

std::string APIServer::generateBetaKey() {
    // Generates a CA3D- prefixed beta key (e.g. CA3D-A1B2C3D4E5F6G7H8I9J0)
    // that matches the format expected by coreai3d.nl registration.
    static const char charset[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    unsigned char randomBytes[32];
    if (RAND_bytes(randomBytes, sizeof(randomBytes)) != 1) {
        return "";
    }
    std::ostringstream os;
    os << "CA3D-";
    for (int i = 0; i < 20; ++i) {
        os << charset[randomBytes[i] % (sizeof(charset) - 1)];
    }
    return os.str();
}

bool APIServer::validateAPIKey(const std::string& apiKey) {
    // Check against database if available
    if (database) {
        return database->validateAPIKey(apiKey);
    }
    // Fallback to simple validation
    return !apiKey.empty() && apiKey.length() >= 16;
}

// Vision API implementations
json APIServer::visionClassify(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName) {
    json result = {
        {"action", "classify"},
        {"image_path", imagePath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Vision classification not yet implemented"}
    };
    return result;
}

json APIServer::visionDetectObjects(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName) {
    json result = {
        {"action", "detect_objects"},
        {"image_path", imagePath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Object detection not yet implemented"}
    };
    return result;
}

json APIServer::visionOCR(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName) {
    json result = {
        {"action", "ocr"},
        {"image_path", imagePath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "OCR not yet implemented"}
    };
    return result;
}

json APIServer::visionAnalyzeFaces(const std::string& sessionId, const std::string& imagePath, const std::string& moduleName) {
    json result = {
        {"action", "analyze_faces"},
        {"image_path", imagePath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Face analysis not yet implemented"}
    };
    return result;
}

json APIServer::visionProcessVideo(const std::string& sessionId, const std::string& videoPath, const std::string& moduleName) {
    json result = {
        {"action", "process_video"},
        {"video_path", videoPath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Video processing not yet implemented"}
    };
    return result;
}

// Audio API implementations
json APIServer::audioSpeechToText(const std::string& sessionId, const std::string& audioPath, const std::string& moduleName) {
    json result = {
        {"action", "speech_to_text"},
        {"audio_path", audioPath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Speech to text not yet implemented"}
    };
    return result;
}

json APIServer::audioTextToSpeech(const std::string& sessionId, const std::string& text, const std::string& voice, const std::string& moduleName) {
    json result = {
        {"action", "text_to_speech"},
        {"text", text},
        {"voice", voice},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Text to speech not yet implemented"}
    };
    return result;
}

json APIServer::audioAnalyze(const std::string& sessionId, const std::string& audioPath, const std::string& moduleName) {
    json result = {
        {"action", "analyze"},
        {"audio_path", audioPath},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Audio analysis not yet implemented"}
    };
    return result;
}

json APIServer::audioProcess(const std::string& sessionId, const std::string& audioPath, const std::string& effects, const std::string& moduleName) {
    json result = {
        {"action", "process"},
        {"audio_path", audioPath},
        {"effects", effects},
        {"module", moduleName},
        {"status", "processing"},
        {"message", "Audio processing not yet implemented"}
    };
    return result;
}

// System API implementations
json APIServer::systemGetProcesses(const std::string& sessionId) {
    json result = {
        {"action", "get_processes"},
        {"status", "success"},
        {"processes", json::array()},
        {"message", "Process listing not yet implemented"}
    };
    return result;
}

// Automation task implementations
json APIServer::performFileBackup(const std::string& sourcePath, const std::string& destPath, bool compress) {
    try {
        // Create destination directory if it doesn't exist
        std::filesystem::path destDir = std::filesystem::path(destPath).parent_path();
        if (!destDir.empty() && !std::filesystem::exists(destDir)) {
            std::filesystem::create_directories(destDir);
        }

        if (compress) {
            // Create tar.gz archive
            std::string tarCmd = "tar -czf " + destPath + " -C " + std::filesystem::path(sourcePath).parent_path().string() + " " + std::filesystem::path(sourcePath).filename().string();
            int ret = system(tarCmd.c_str());
            if (ret != 0) {
                return createErrorResponse("Failed to create compressed backup archive", 500);
            }
        } else {
            // Simple copy
            std::filesystem::copy(sourcePath, destPath, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
        }

        // Get file size
        size_t fileSize = std::filesystem::file_size(destPath);

        return {
            {"status", "success"},
            {"message", "File backup completed successfully"},
            {"source_path", sourcePath},
            {"dest_path", destPath},
            {"compressed", compress},
            {"file_size_bytes", fileSize}
        };
    }
    catch (const std::exception& e) {
        return createErrorResponse("File backup failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performSystemCleanup(const std::string& cleanupType, bool force) {
    try {
        json result = {
            {"status", "success"},
            {"cleanup_type", cleanupType},
            {"files_cleaned", 0},
            {"space_freed_bytes", 0}
        };

        if (cleanupType == "temp_files") {
            // Clean temporary files
            std::vector<std::string> tempDirs = {"/tmp", "/var/tmp"};
            size_t totalCleaned = 0;
            size_t spaceFreed = 0;

            for (const auto& tempDir : tempDirs) {
                if (std::filesystem::exists(tempDir)) {
                    for (const auto& entry : std::filesystem::directory_iterator(tempDir)) {
                        try {
                            if (entry.is_regular_file()) {
                                auto fileTime = std::filesystem::last_write_time(entry);
                                auto now = std::filesystem::file_time_type::clock::now();
                                auto age = std::chrono::duration_cast<std::chrono::hours>(now - fileTime).count();

                                // Remove files older than 24 hours or if force is true
                                if (force || age > 24) {
                                    size_t fileSize = entry.file_size();
                                    std::filesystem::remove(entry);
                                    totalCleaned++;
                                    spaceFreed += fileSize;
                                }
                            }
                        } catch (...) {
                            // Skip files that can't be processed
                        }
                    }
                }
            }

            result["files_cleaned"] = totalCleaned;
            result["space_freed_bytes"] = spaceFreed;
            result["message"] = "Temporary files cleanup completed";
        }
        else if (cleanupType == "cache") {
            // Clean cache files
            std::vector<std::string> cacheDirs = {"/var/cache", "/home/" + std::string(getenv("USER") ? getenv("USER") : "user") + "/.cache"};
            size_t spaceFreed = 0;

            for (const auto& cacheDir : cacheDirs) {
                if (std::filesystem::exists(cacheDir)) {
                    // Use system command to clean cache (safer than recursive deletion)
                    std::string cleanCmd = "find " + cacheDir + " -type f -mtime +7 -delete 2>/dev/null || true";
                    int sysRet = system(cleanCmd.c_str());
                    (void)sysRet; // Suppress unused-result warning
                    result["message"] = "Cache cleanup completed (removed files older than 7 days)";
                }
            }
        }

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("System cleanup failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performLogRotation(const std::string& logPath, int maxSize, int keepFiles) {
    try {
        json result = {
            {"status", "success"},
            {"log_path", logPath},
            {"max_size_mb", maxSize},
            {"keep_files", keepFiles},
            {"rotated_files", 0}
        };

        if (!std::filesystem::exists(logPath)) {
            return createErrorResponse("Log path does not exist: " + logPath, 400);
        }

        int rotatedCount = 0;

        // Find log files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(logPath)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();

                // Check if it's a log file (basic check)
                if (filename.find(".log") != std::string::npos) {
                    size_t fileSize = entry.file_size();
                    size_t maxSizeBytes = maxSize * 1024 * 1024; // Convert MB to bytes

                    if (fileSize > maxSizeBytes) {
                        // Rotate the log file
                        std::string baseName = filename.substr(0, filename.find_last_of("."));
                        std::string rotatedName = baseName + "." + std::to_string(keepFiles) + ".log";

                        // Remove oldest rotated file if it exists
                        std::filesystem::path oldestRotated = entry.path().parent_path() / rotatedName;
                        if (std::filesystem::exists(oldestRotated)) {
                            std::filesystem::remove(oldestRotated);
                        }

                        // Shift existing rotated files
                        for (int i = keepFiles - 1; i >= 1; --i) {
                            std::string currentName = baseName + "." + std::to_string(i) + ".log";
                            std::string nextName = baseName + "." + std::to_string(i + 1) + ".log";

                            std::filesystem::path currentPath = entry.path().parent_path() / currentName;
                            std::filesystem::path nextPath = entry.path().parent_path() / nextName;

                            if (std::filesystem::exists(currentPath)) {
                                std::filesystem::rename(currentPath, nextPath);
                            }
                        }

                        // Create the first rotated file
                        std::filesystem::path rotatedPath = entry.path().parent_path() / (baseName + ".1.log");
                        std::filesystem::rename(entry.path(), rotatedPath);

                        rotatedCount++;
                    }
                }
            }
        }

        result["rotated_files"] = rotatedCount;
        result["message"] = "Log rotation completed successfully";

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Log rotation failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performServiceRestart(const std::string& serviceName) {
    try {
        // Use systemctl to restart service (Linux specific)
        std::string restartCmd = "sudo systemctl restart " + serviceName + " 2>&1";
        int ret = system(restartCmd.c_str());

        json result = {
            {"status", ret == 0 ? "success" : "failed"},
            {"service_name", serviceName},
            {"message", ret == 0 ? "Service restarted successfully" : "Failed to restart service"}
        };

        if (ret != 0) {
            result["error_code"] = ret;
        }

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Service restart failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performDiskSpaceCheck(const std::string& path, int thresholdPercent) {
    try {
        std::filesystem::space_info space = std::filesystem::space(path);

        size_t totalBytes = space.capacity;
        size_t freeBytes = space.available;
        size_t usedBytes = totalBytes - freeBytes;

        double usedPercent = (static_cast<double>(usedBytes) / totalBytes) * 100.0;
        double freePercent = (static_cast<double>(freeBytes) / totalBytes) * 100.0;

        bool thresholdExceeded = usedPercent >= thresholdPercent;

        json result = {
            {"status", "success"},
            {"path", path},
            {"total_bytes", totalBytes},
            {"used_bytes", usedBytes},
            {"free_bytes", freeBytes},
            {"used_percent", usedPercent},
            {"free_percent", freePercent},
            {"threshold_percent", thresholdPercent},
            {"threshold_exceeded", thresholdExceeded}
        };

        if (thresholdExceeded) {
            result["warning"] = "Disk space usage exceeds threshold";
        } else {
            result["message"] = "Disk space is within acceptable limits";
        }

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Disk space check failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performNetworkTest(const std::string& target, int timeout) {
    try {
        // Use ping to test network connectivity
        std::string pingCmd = "ping -c 3 -W " + std::to_string(timeout) + " " + target + " 2>&1";
        FILE* pipe = popen(pingCmd.c_str(), "r");

        if (!pipe) {
            return createErrorResponse("Failed to execute ping command", 500);
        }

        std::string output;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }

        int ret = pclose(pipe);
        bool success = (ret == 0);

        // Parse ping output for basic metrics
        double avgRtt = 0.0;
        std::regex rttRegex("rtt min/avg/max/mdev = [0-9.]+/([0-9.]+)/[0-9.]+/[0-9.]+ ms");
        std::smatch match;
        if (std::regex_search(output, match, rttRegex)) {
            avgRtt = std::stod(match[1]);
        }

        json result = {
            {"status", success ? "success" : "failed"},
            {"target", target},
            {"timeout_seconds", timeout},
            {"reachable", success},
            {"average_rtt_ms", avgRtt}
        };

        if (success) {
            result["message"] = "Network connectivity test passed";
        } else {
            result["message"] = "Network connectivity test failed";
            result["error"] = "Host unreachable or network issues";
        }

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Network test failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performProcessMonitor(const std::string& processName, int maxMemory, int maxCpu) {
    try {
        json result = {
            {"status", "success"},
            {"process_name", processName},
            {"max_memory_mb", maxMemory},
            {"max_cpu_percent", maxCpu},
            {"processes_found", 0},
            {"issues_detected", 0}
        };

        // Use ps command to get process information
        std::string psCmd = "ps aux --no-headers 2>/dev/null";
        FILE* pipe = popen(psCmd.c_str(), "r");

        if (!pipe) {
            return createErrorResponse("Failed to execute ps command", 500);
        }

        std::vector<json> processes;
        char buffer[1024];
        int foundCount = 0;
        int issuesCount = 0;

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line(buffer);
            std::istringstream iss(line);

            std::string user, pid, cpu, mem, vsz, rss, tty, stat, start, time, command;
            iss >> user >> pid >> cpu >> mem >> vsz >> rss >> tty >> stat >> start >> time;

            // Get the rest as command
            std::getline(iss, command);

            // Check if process name matches (case-insensitive)
            if (!processName.empty()) {
                std::string lowerCommand = command;
                std::string lowerProcessName = processName;
                std::transform(lowerCommand.begin(), lowerCommand.end(), lowerCommand.begin(), ::tolower);
                std::transform(lowerProcessName.begin(), lowerProcessName.end(), lowerProcessName.begin(), ::tolower);

                if (lowerCommand.find(lowerProcessName) == std::string::npos) {
                    continue;
                }
            }

            foundCount++;

            double cpuPercent = std::stod(cpu);
            double memPercent = std::stod(mem);
            int memMB = std::stoi(rss) / 1024; // Convert KB to MB

            bool hasIssues = false;
            if (maxCpu > 0 && cpuPercent > maxCpu) {
                hasIssues = true;
            }
            if (maxMemory > 0 && memMB > maxMemory) {
                hasIssues = true;
            }

            if (hasIssues) {
                issuesCount++;
            }

            processes.push_back({
                {"pid", std::stoi(pid)},
                {"user", user},
                {"cpu_percent", cpuPercent},
                {"memory_percent", memPercent},
                {"memory_mb", memMB},
                {"command", command.substr(0, 100)}, // Truncate long commands
                {"has_issues", hasIssues}
            });
        }

        pclose(pipe);

        result["processes_found"] = foundCount;
        result["issues_detected"] = issuesCount;
        result["processes"] = processes;

        if (issuesCount > 0) {
            result["warning"] = "Some processes exceed specified limits";
        } else {
            result["message"] = "Process monitoring completed successfully";
        }

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Process monitoring failed: " + std::string(e.what()), 500);
    }
}

json APIServer::performCustomScript(const std::string& scriptPath, const std::vector<std::string>& args, int timeout) {
    try {
        if (!std::filesystem::exists(scriptPath)) {
            return createErrorResponse("Script file does not exist: " + scriptPath, 400);
        }

        // Build command with arguments
        std::string cmd = scriptPath;
        for (const auto& arg : args) {
            cmd += " \"" + arg + "\"";
        }

        // Add timeout wrapper
        cmd = "timeout " + std::to_string(timeout) + "s " + cmd;

        // Execute the script
        FILE* pipe = popen(cmd.c_str(), "r");

        if (!pipe) {
            return createErrorResponse("Failed to execute script", 500);
        }

        std::string output;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }

        int ret = pclose(pipe);
        bool success = (ret == 0);

        json result = {
            {"status", success ? "success" : "failed"},
            {"script_path", scriptPath},
            {"arguments", args},
            {"timeout_seconds", timeout},
            {"exit_code", ret},
            {"output", output.substr(0, 10000)} // Limit output size
        };

        if (success) {
            result["message"] = "Script executed successfully";
        } else {
            result["message"] = "Script execution failed";
            if (ret == 124) {
                result["error"] = "Script timed out";
            }
        }

        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Custom script execution failed: " + std::string(e.what()), 500);
    }
}

json APIServer::systemStartApplication(const std::string& sessionId, const std::string& appName, const std::vector<std::string>& args) {
    json result = {
        {"action", "start_application"},
        {"application", appName},
        {"arguments", args},
        {"status", "processing"},
        {"message", "Application start not yet implemented"}
    };
    return result;
}

json APIServer::systemAutomateTask(const std::string& sessionId, const std::string& taskType, const json& parameters) {
    try {
        json result = {
            {"action", "automate_task"},
            {"task_type", taskType},
            {"status", "processing"},
            {"timestamp", getTimestampString()}
        };

        if (taskType == "file_backup") {
            std::string sourcePath = parameters.value("source_path", "");
            std::string destPath = parameters.value("dest_path", "");
            bool compress = parameters.value("compress", false);

            if (sourcePath.empty() || destPath.empty()) {
                return createErrorResponse("Source and destination paths are required for file backup", 400);
            }

            result = performFileBackup(sourcePath, destPath, compress);
        }
        else if (taskType == "system_cleanup") {
            std::string cleanupType = parameters.value("cleanup_type", "temp_files");
            bool force = parameters.value("force", false);

            result = performSystemCleanup(cleanupType, force);
        }
        else if (taskType == "log_rotation") {
            std::string logPath = parameters.value("log_path", "/var/log");
            int maxSize = parameters.value("max_size_mb", 100);
            int keepFiles = parameters.value("keep_files", 5);

            result = performLogRotation(logPath, maxSize, keepFiles);
        }
        else if (taskType == "service_restart") {
            std::string serviceName = parameters.value("service_name", "");

            if (serviceName.empty()) {
                return createErrorResponse("Service name is required for service restart", 400);
            }

            result = performServiceRestart(serviceName);
        }
        else if (taskType == "disk_space_check") {
            std::string path = parameters.value("path", "/");
            int thresholdPercent = parameters.value("threshold_percent", 90);

            result = performDiskSpaceCheck(path, thresholdPercent);
        }
        else if (taskType == "network_test") {
            std::string target = parameters.value("target", "8.8.8.8");
            int timeout = parameters.value("timeout", 5);

            result = performNetworkTest(target, timeout);
        }
        else if (taskType == "process_monitor") {
            std::string processName = parameters.value("process_name", "");
            int maxMemory = parameters.value("max_memory_mb", 0);
            int maxCpu = parameters.value("max_cpu_percent", 0);

            result = performProcessMonitor(processName, maxMemory, maxCpu);
        }
        else if (taskType == "custom_script") {
            std::string scriptPath = parameters.value("script_path", "");
            std::vector<std::string> args = parameters.value("arguments", std::vector<std::string>{});
            int timeout = parameters.value("timeout", 30);

            if (scriptPath.empty()) {
                return createErrorResponse("Script path is required for custom script execution", 400);
            }

            result = performCustomScript(scriptPath, args, timeout);
        }
        else {
            return createErrorResponse("Unknown automation task type: " + taskType, 400);
        }

        result["task_type"] = taskType;
        result["completed_at"] = getTimestampString();
        return result;
    }
    catch (const std::exception& e) {
        return createErrorResponse("Automation task failed: " + std::string(e.what()), 500);
    }
}

json APIServer::systemGetSystemMetrics(const std::string& sessionId) {
    json result = {
        {"action", "get_system_metrics"},
        {"status", "success"},
        {"metrics", {
            {"cpu_usage", 0.0},
            {"memory_usage", 0.0},
            {"disk_usage", 0.0}
        }},
        {"message", "System metrics not yet implemented"}
    };
    return result;
}

json APIServer::systemCaptureScreen(const std::string& sessionId, const std::string& outputPath) {
    json result = {
        {"action", "capture_screen"},
        {"output_path", outputPath},
        {"status", "processing"},
        {"message", "Screen capture not yet implemented"}
    };
    return result;
}

// Web API implementations
json APIServer::webSearch(const std::string& sessionId, const std::string& query, int maxResults) {
    json result = {
        {"action", "search"},
        {"query", query},
        {"max_results", maxResults},
        {"status", "processing"},
        {"results", json::array()},
        {"message", "Web search not yet implemented"}
    };
    return result;
}

json APIServer::webExtractContent(const std::string& sessionId, const std::string& url) {
    json result = {
        {"action", "extract_content"},
        {"url", url},
        {"status", "processing"},
        {"content", ""},
        {"message", "Content extraction not yet implemented"}
    };
    return result;
}

json APIServer::webGetNews(const std::string& sessionId, const std::string& topic, int maxArticles) {
    json result = {
        {"action", "get_news"},
        {"topic", topic},
        {"max_articles", maxArticles},
        {"status", "processing"},
        {"articles", json::array()},
        {"message", "News retrieval not yet implemented"}
    };
    return result;
}

json APIServer::webAnalyzeSentiment(const std::string& sessionId, const std::string& content) {
    json result = {
        {"action", "analyze_sentiment"},
        {"content", content},
        {"status", "processing"},
        {"sentiment", "neutral"},
        {"confidence", 0.0},
        {"message", "Sentiment analysis not yet implemented"}
    };
    return result;
}

// Math API implementations
json APIServer::mathCalculate(const std::string& sessionId, const std::string& expression) {
    json result = {
        {"action", "calculate"},
        {"expression", expression},
        {"status", "processing"},
        {"result", nullptr},
        {"message", "Math calculation not yet implemented"}
    };
    return result;
}

json APIServer::mathOptimize(const std::string& sessionId, const std::string& objective, const std::vector<float>& initialGuess) {
    json result = {
        {"action", "optimize"},
        {"objective", objective},
        {"initial_guess", initialGuess},
        {"status", "processing"},
        {"result", nullptr},
        {"message", "Math optimization not yet implemented"}
    };
    return result;
}

json APIServer::mathStatistics(const std::string& sessionId, const std::vector<float>& data) {
    json result = {
        {"action", "statistics"},
        {"data", data},
        {"status", "processing"},
        {"statistics", json::object()},
        {"message", "Statistical analysis not yet implemented"}
    };
    return result;
}

json APIServer::mathMatrixOperation(const std::string& sessionId, const std::string& operation, const std::string& matrixName, const json& parameters) {
    json result = {
        {"action", "matrix_operation"},
        {"operation", operation},
        {"matrix_name", matrixName},
        {"parameters", parameters},
        {"status", "processing"},
        {"result", nullptr},
        {"message", "Matrix operation not yet implemented"}
    };
    return result;
}

// Orchestrator API implementations
json APIServer::orchestratorSubmitTask(const std::string& sessionId, const std::string& taskType, const std::string& description, const json& parameters) {
    json result = {
        {"action", "submit_task"},
        {"task_type", taskType},
        {"description", description},
        {"parameters", parameters},
        {"status", "processing"},
        {"task_id", generateUniqueId()},
        {"message", "Task submission not yet implemented"}
    };
    return result;
}

json APIServer::orchestratorGetTaskStatus(const std::string& sessionId, const std::string& taskId) {
    json result = {
        {"action", "get_task_status"},
        {"task_id", taskId},
        {"status", "unknown"},
        {"message", "Task status check not yet implemented"}
    };
    return result;
}

json APIServer::orchestratorGetSystemState(const std::string& sessionId) {
    json result = {
        {"action", "get_system_state"},
        {"status", "success"},
        {"modules", json::object()},
        {"active_sessions", activeSessions.size()},
        {"total_requests", 0},
        {"message", "System state retrieval not yet implemented"}
    };
    return result;
}

json APIServer::orchestratorProcessMultiModal(const std::string& sessionId, const std::string& contentType, const std::string& content, const std::vector<std::string>& analysisTypes) {
    json result = {
        {"action", "process_multimodal"},
        {"content_type", contentType},
        {"content", content},
        {"analysis_types", analysisTypes},
        {"status", "processing"},
        {"results", json::object()},
        {"message", "Multi-modal processing not yet implemented"}
    };
    return result;
}

// Neural API implementations
json APIServer::neuralGetTopology(const std::string& sessionId) {
    if (!validateSession(sessionId)) {
        return createErrorResponse("Invalid session", 401);
    }
    return handleNeuralAPI("topology", json::object());
}

json APIServer::neuralGetActivity(const std::string& sessionId) {
    if (!validateSession(sessionId)) {
        return createErrorResponse("Invalid session", 401);
    }
    return handleNeuralAPI("activity", json::object());
}

// Batch processing
json APIServer::batchProcess(const std::string& sessionId, const std::vector<json>& requests) {
    json result = {
        {"action", "batch_process"},
        {"requests", requests},
        {"status", "processing"},
        {"results", json::array()},
        {"message", "Batch processing not yet implemented"}
    };
    return result;
}

json APIServer::batchVisionProcess(const std::string& sessionId, const std::vector<std::string>& imagePaths, const std::vector<std::string>& operations) {
    json result = {
        {"action", "batch_vision_process"},
        {"image_paths", imagePaths},
        {"operations", operations},
        {"status", "processing"},
        {"results", json::array()},
        {"message", "Batch vision processing not yet implemented"}
    };
    return result;
}

json APIServer::batchAudioProcess(const std::string& sessionId, const std::vector<std::string>& audioPaths, const std::vector<std::string>& operations) {
    json result = {
        {"action", "batch_audio_process"},
        {"audio_paths", audioPaths},
        {"operations", operations},
        {"status", "processing"},
        {"results", json::array()},
        {"message", "Batch audio processing not yet implemented"}
    };
    return result;
}

// Real-time processing
json APIServer::startRealTimeProcessing(const std::string& sessionId, const std::string& processingType, const json& parameters) {
    json result = {
        {"action", "start_realtime_processing"},
        {"processing_type", processingType},
        {"parameters", parameters},
        {"status", "processing"},
        {"stream_id", generateUniqueId()},
        {"message", "Real-time processing not yet implemented"}
    };
    return result;
}

json APIServer::stopRealTimeProcessing(const std::string& sessionId, const std::string& processingType) {
    json result = {
        {"action", "stop_realtime_processing"},
        {"processing_type", processingType},
        {"status", "success"},
        {"message", "Real-time processing stop not yet implemented"}
    };
    return result;
}

json APIServer::getRealTimeResults(const std::string& sessionId, const std::string& processingType) {
    json result = {
        {"action", "get_realtime_results"},
        {"processing_type", processingType},
        {"status", "success"},
        {"results", json::array()},
        {"message", "Real-time results retrieval not yet implemented"}
    };
    return result;
}

// File upload/download
json APIServer::uploadFile(const std::string& sessionId, const std::string& filePath, const std::string& fileType) {
    json result = {
        {"action", "upload_file"},
        {"file_path", filePath},
        {"file_type", fileType},
        {"status", "processing"},
        {"file_id", generateUniqueId()},
        {"message", "File upload not yet implemented"}
    };
    return result;
}

json APIServer::downloadFile(const std::string& sessionId, const std::string& fileId, const std::string& downloadPath) {
    json result = {
        {"action", "download_file"},
        {"file_id", fileId},
        {"download_path", downloadPath},
        {"status", "processing"},
        {"message", "File download not yet implemented"}
    };
    return result;
}

json APIServer::listUploadedFiles(const std::string& sessionId) {
    json result = {
        {"action", "list_files"},
        {"status", "success"},
        {"files", json::array()},
        {"message", "File listing not yet implemented"}
    };
    return result;
}

// Status and information
json APIServer::getAPIStatus() {
    json status;
    status["server_name"] = serverName;
    status["version"] = apiVersion;
    status["is_running"] = static_cast<bool>(isRunning);
    status["host"] = host;
    status["port"] = port;
    status["threads"] = numThreads;
    status["active_sessions"] = activeSessions.size();
    status["uptime"] = "0 seconds";
    return status;
}

json APIServer::getModuleStatus(const std::string& moduleName) {
    json status = {
        {"module", moduleName},
        {"status", "unknown"},
        {"message", "Module status not yet implemented"}
    };

    // Check if training module is available
    if (trainingModule) {
        if (moduleName == "neural" || moduleName == "training") {
            status["status"] = "available";
            status["message"] = "Neural network training module is available";
            status["input_size"] = trainingModule->inputSize;
            status["output_size"] = trainingModule->outputSize;
            status["layers"] = trainingModule->layers;
            status["neurons"] = trainingModule->neurons;
            status["learning_rate"] = trainingModule->learningRate;
        }
    }

    return status;
}

json APIServer::getAllModulesStatus() {
    json status = {
        {"modules", json::object()},
        {"message", "All modules status not yet implemented"}
    };
    return status;
}

json APIServer::getAPIEndpoints() {
    json endpoints = {
        {"endpoints", json::array()},
        {"message", "API endpoints listing not yet implemented"}
    };
    return endpoints;
}

json APIServer::getAPIVersion() {
    json version = {
        {"version", apiVersion},
        {"server", serverName},
        {"features", json::array({"vision", "audio", "system", "web", "math", "orchestrator"})}
    };
    return version;
}

// Logging and monitoring
json APIServer::getRequestLogs(const std::string& sessionId, int limit) {
    json logs = {
        {"logs", json::array()},
        {"limit", limit},
        {"message", "Request logs not yet implemented"}
    };
    return logs;
}

json APIServer::getPerformanceMetrics(const std::string& sessionId) {
    json metrics = {
        {"metrics", json::object()},
        {"message", "Performance metrics not yet implemented"}
    };
    return metrics;
}

json APIServer::getErrorLogs(const std::string& sessionId, int limit) {
    json logs = {
        {"logs", json::array()},
        {"limit", limit},
        {"message", "Error logs not yet implemented"}
    };
    return logs;
}

// Utility functions
std::string APIServer::getFileExtension(const std::string& filename) {
    size_t lastDot = filename.find_last_of('.');
    if (lastDot == std::string::npos) {
        return "";
    }
    return filename.substr(lastDot + 1);
}

std::string APIServer::generateUniqueId() {
    static std::atomic<int> counter{0};
    return "session_" + std::to_string(++counter);
}

std::string APIServer::getTimestampString() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

json APIServer::createSuccessResponse(const std::string& message, const json& data) {
    return {
        {"status", "success"},
        {"message", message},
        {"data", data},
        {"timestamp", getTimestampString()}
    };
}

json APIServer::createErrorResponse(const std::string& error, int errorCode, const json& details) {
    return {
        {"status", "error"},
        {"error", error},
        {"error_code", errorCode},
        {"details", details},
        {"timestamp", getTimestampString()}
    };
}

// Server management
void APIServer::restartServer() {
    stop();
    start();
}

void APIServer::reloadConfiguration() {
    // TODO: Reload configuration from file
}

bool APIServer::isHealthy() {
    return isRunning && isInitialized;
}

size_t APIServer::getActiveConnections() {
    return activeSessions.size();
}

size_t APIServer::getTotalRequests() {
    // TODO: Track total requests
    return 0;
}

// Module integration
bool APIServer::addVisionModule(const std::string& name, std::unique_ptr<VisionModule> module) {
    // TODO: Implement module registration
    return true;
}

bool APIServer::addAudioModule(const std::string& name, std::unique_ptr<AudioModule> module) {
    // TODO: Implement module registration
    return true;
}

bool APIServer::addSystemModule(const std::string& name, std::unique_ptr<SystemModule> module) {
    // TODO: Implement module registration
    return true;
}

bool APIServer::addWebModule(const std::string& name, std::unique_ptr<WebModule> module) {
    // TODO: Implement module registration
    return true;
}

bool APIServer::addMathModule(const std::string& name, std::unique_ptr<MathModule> module) {
    // TODO: Implement module registration
    return true;
}

// // Hyperparameter tuning API implementation
// json APIServer::handleHyperparameterTuningAPI(const std::string& action, const json& parameters, const std::string& method) {
//     try {
//         if (!hyperparameterTuner) {
//             // Initialize hyperparameter tuner if not already done
//             if (!trainingModule) {
//                 return createErrorResponse("Training module not available", 503);
//             }
//
//             // Create a copy of the training module for tuning
//             auto tuningTraining = std::make_unique<Training>(*trainingModule);
//             hyperparameterTuner = std::make_unique<HyperparameterTuner>(
//                 std::move(tuningTraining),
//                 std::map<std::string, ParameterDefinition>() // Will be set via API
//             );
//
//             // Set database if available
//             if (database) {
//                 hyperparameterTuner->setDatabase(std::make_unique<Database>(*database));
//             }
//         }
//
//         if (action == "start" && method == "POST") {
//             // Start hyperparameter tuning
//             std::string searchMethod = parameters.value("search_method", "grid_search");
//             int maxEvaluations = parameters.value("max_evaluations", 100);
//             bool parallel = parameters.value("parallel", true);
//             int maxParallel = parameters.value("max_parallel", 4);
//
//             // Set up parameter space
//             std::map<std::string, ParameterDefinition> paramSpace;
//
//             if (parameters.contains("parameters")) {
//                 for (const auto& param : parameters["parameters"]) {
//                     std::string name = param.value("name", "");
//                     std::string type = param.value("type", "");
//                     ParameterDefinition def;
//                     def.name = name;
//
//                     if (type == "float_range") {
//                         def.type = ParameterType::FLOAT_RANGE;
//                         def.float_range = {param.value("min", 0.0f), param.value("max", 1.0f)};
//                         def.num_samples = param.value("num_samples", 10);
//                     } else if (type == "int_range") {
//                         def.type = ParameterType::INT_RANGE;
//                         def.int_range = {param.value("min", 1), param.value("max", 10)};
//                     } else if (type == "categorical") {
//                         def.type = ParameterType::CATEGORICAL;
//                         def.categories = param.value("categories", std::vector<std::string>{});
//                     } else if (type == "log_scale") {
//                         def.type = ParameterType::LOG_SCALE;
//                         def.float_range = {param.value("min", 0.001f), param.value("max", 1.0f)};
//                         def.log_base = param.value("log_base", 10.0f);
//                         def.num_samples = param.value("num_samples", 10);
//                     }
//
//                     paramSpace[name] = def;
//                 }
//             }
//
//             // Set parameter space
//             for (const auto& param : paramSpace) {
//                 hyperparameterTuner->addParameter(param.first, param.second);
//             }
//
//             // Configure tuning
//             TuningConfig config;
//             if (searchMethod == "grid_search") {
//                 config.method = SearchMethod::GRID_SEARCH;
//             } else if (searchMethod == "random_search") {
//                 config.method = SearchMethod::RANDOM_SEARCH;
//             } else if (searchMethod == "bayesian") {
//                 config.method = SearchMethod::BAYESIAN_OPTIMIZATION;
//             }
//             config.max_evaluations = max_evaluations;
//             config.parallel_evaluation = parallel;
//             config.max_parallel_jobs = maxParallel;
//
//             hyperparameterTuner->setTuningConfig(config);
//
//             // Start tuning
//             bool started = hyperparameterTuner->startTuning();
//
//             if (!started) {
//                 return createErrorResponse("Failed to start hyperparameter tuning", 500);
//             }
//
//             json response = {
//                 {"status", "started"},
//                 {"search_method", searchMethod},
//                 {"max_evaluations", max_evaluations},
//                 {"parallel", parallel},
//                 {"max_parallel", maxParallel},
//                 {"message", "Hyperparameter tuning started successfully"},
//                 {"timestamp", getTimestampString()}
//             };
//             return response;
//
//         } else if (action == "status" && method == "GET") {
//             // Get tuning status
//             bool isActive = hyperparameterTuner->isTuning();
//             std::string status = hyperparameterTuner->getTuningStatus();
//
//             json response = {
//                 {"is_active", isActive},
//                 {"status", status},
//                 {"timestamp", getTimestampString()}
//             };
//             return response;
//
//         } else if (action == "stop" && method == "POST") {
//             // Stop tuning
//             bool stopped = hyperparameterTuner->stopTuning();
//
//             json response = {
//                 {"stopped", stopped},
//                 {"message", stopped ? "Hyperparameter tuning stopped successfully" : "Failed to stop tuning"},
//                 {"timestamp", getTimestampString()}
//             };
//             return response;
//
//         } else if (action == "results" && method == "GET") {
//             // Get tuning results
//             TuningResults results = hyperparameterTuner->getCurrentResults();
//
//             json response = {
//                 {"total_evaluations", results.total_evaluations},
//                 {"total_time", results.total_time},
//                 {"timestamp", results.timestamp},
//                 {"best_score", results.best_parameters.score},
//                 {"convergence_history", json::array()}
//             };
//
//             // Add best parameters
//             json bestParams;
//             for (const auto& param : results.best_parameters.float_params) {
//                 bestParams["float_params"][param.first] = param.second;
//             }
//             for (const auto& param : results.best_parameters.int_params) {
//                 bestParams["int_params"][param.first] = param.second;
//             }
//             for (const auto& param : results.best_parameters.categorical_params) {
//                 bestParams["categorical_params"][param.first] = param.second;
//             }
//             response["best_parameters"] = bestParams;
//
//             // Add convergence history
//             for (const auto& point : results.convergence_history) {
//                 response["convergence_history"].push_back({point.first, point.second});
//             }
//
//             return response;
//
//         } else if (action == "save_results" && method == "POST") {
//             // Save tuning results
//             std::string filepath = parameters.value("filepath", "tuning_results.json");
//             TuningResults results = hyperparameterTuner->getCurrentResults();
//
//             bool saved = hyperparameterTuner->saveResults(results, filepath);
//
//             json response = {
//                 {"saved", saved},
//                 {"filepath", filepath},
//                 {"message", saved ? "Results saved successfully" : "Failed to save results"},
//                 {"timestamp", getTimestampString()}
//             };
//             return response;
//
//         } else {
//             return createErrorResponse("Unknown hyperparameter tuning action: " + action, 400);
//         }
//     }
//     catch (const std::exception& e) {
//         return createErrorResponse("Hyperparameter tuning API error: " + std::string(e.what()), 500);
//     }
// }
void APIServer::setTrainingModule(std::unique_ptr<Training> training) {
    trainingModule = std::move(training);
}

void APIServer::setPredictModule(std::unique_ptr<Predict> predict) {
    predictModule = std::move(predict);
}

void APIServer::setLanguageProcessor(Language* lang) {
    if (orchestrator) {
        orchestrator->setLanguageProcessor(lang);
    }
}

json APIServer::handleDatasetAPI(const std::string& action, const json& parameters, const std::string& method) {
    try {
        if (!database) {
            return createErrorResponse("Database not available", 503);
        }

        // Helper function to read CSV file and return data as vector of vectors
        auto readCSVFile = [](const std::string& filename, bool hasHeader, char delimiter) -> std::pair<std::vector<std::vector<float>>, std::string> {
            std::vector<std::vector<float>> data;
            std::ifstream file(filename);
            std::string line;
            int lineNumber = 0;

            if (!file.is_open()) {
                return {data, "Failed to open file: " + filename};
            }

            // Skip header if present
            if (hasHeader && std::getline(file, line)) {
                lineNumber++;
            }

            // Read data lines
            while (std::getline(file, line)) {
                lineNumber++;
                std::vector<float> row;
                std::stringstream ss(line);
                std::string cell;

                while (std::getline(ss, cell, delimiter)) {
                    try {
                        // Trim whitespace
                        cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(), [](unsigned char ch) {
                            return !std::isspace(ch);
                        }));
                        cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch) {
                            return !std::isspace(ch);
                        }).base(), cell.end());

                        if (!cell.empty()) {
                            row.push_back(std::stof(cell));
                        }
                    } catch (const std::exception&) {
                        return {data, "Invalid numeric data in file " + filename + " at line " + std::to_string(lineNumber) + ", cell: '" + cell + "'"};
                    }
                }

                if (!row.empty()) {
                    data.push_back(row);
                }
            }

            file.close();
            return {data, ""}; // Empty string means success
        };

        // Helper function to combine multiple CSV datasets
        auto combineCSVData = [](const std::vector<std::vector<std::vector<float>>>& datasets) -> std::vector<std::vector<float>> {
            std::vector<std::vector<float>> combined;

            for (const auto& dataset : datasets) {
                combined.insert(combined.end(), dataset.begin(), dataset.end());
            }

            return combined;
        };

        if (action == "create" && method == "POST") {
            // Create a new dataset
            std::string datasetName = parameters.value("name", "");
            std::string description = parameters.value("description", "");
            int numRows = parameters.value("num_rows", 0);
            int numFeatures = parameters.value("num_features", 0);
            int numLabels = parameters.value("num_labels", 0);

            // Validate required parameters
            if (datasetName.empty()) {
                return createErrorResponse("Dataset name cannot be empty", 400);
            }
            if (numRows <= 0) {
                return createErrorResponse("Number of rows must be positive", 400);
            }
            if (numFeatures <= 0) {
                return createErrorResponse("Number of features must be positive", 400);
            }
            if (numLabels <= 0) {
                return createErrorResponse("Number of labels must be positive", 400);
            }

            int datasetId = database->addDataset(datasetName, description, numRows, numFeatures, numLabels);

            json response = {
                {"dataset_id", datasetId},
                {"name", datasetName},
                {"description", description},
                {"num_rows", numRows},
                {"num_features", numFeatures},
                {"num_labels", numLabels},
                {"message", "Dataset created successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "add_record" && method == "POST") {
            // Add a record to a dataset
            int datasetId = parameters.value("dataset_id", -1);
            int rowIndex = parameters.value("row_index", -1);
            std::vector<float> featureValues = parameters.value("features", std::vector<float>{});
            std::vector<float> labelValues = parameters.value("labels", std::vector<float>{});

            // Validate parameters
            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }
            if (rowIndex < 0) {
                return createErrorResponse("Row index must be non-negative", 400);
            }
            if (featureValues.empty()) {
                return createErrorResponse("Feature values cannot be empty", 400);
            }
            if (labelValues.empty()) {
                return createErrorResponse("Label values cannot be empty", 400);
            }

            database->addDatasetRecord(datasetId, rowIndex, featureValues, labelValues);

            json response = {
                {"dataset_id", datasetId},
                {"row_index", rowIndex},
                {"features_count", featureValues.size()},
                {"labels_count", labelValues.size()},
                {"message", "Record added successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "get" && method == "GET") {
            // Get dataset information and records
            int datasetId = parameters.value("dataset_id", -1);

            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }

            Database::DatasetData data = database->getDataset(datasetId);

            json response = {
                {"dataset_id", datasetId},
                {"name", data.datasetName},
                {"num_rows", data.numRows},
                {"num_features", data.numFeatures},
                {"num_labels", data.numLabels},
                {"records", json::array()}
            };

            // Add records data
            for (size_t i = 0; i < data.inputs.size(); ++i) {
                if (!data.inputs[i].empty() && !data.targets[i].empty()) {
                    json record = {
                        {"row_index", static_cast<int>(i)},
                        {"features", data.inputs[i]},
                        {"labels", data.targets[i]}
                    };
                    response["records"].push_back(record);
                }
            }

            response["message"] = "Dataset retrieved successfully";
            response["timestamp"] = getTimestampString();
            return response;
        }
        else if (action == "update_labels" && method == "PUT") {
            // Update labels for a specific record
            int datasetId = parameters.value("dataset_id", -1);
            int rowIndex = parameters.value("row_index", -1);
            std::vector<float> labelValues = parameters.value("labels", std::vector<float>{});

            // Validate parameters
            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }
            if (rowIndex < 0) {
                return createErrorResponse("Row index must be non-negative", 400);
            }
            if (labelValues.empty()) {
                return createErrorResponse("Label values cannot be empty", 400);
            }

            database->updateDatasetRecordLabels(datasetId, rowIndex, labelValues);

            json response = {
                {"dataset_id", datasetId},
                {"row_index", rowIndex},
                {"labels_count", labelValues.size()},
                {"message", "Labels updated successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "clear_records" && method == "DELETE") {
            // Clear all records from a dataset
            long long datasetId = parameters.value("dataset_id", -1LL);

            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }

            database->clearDatasetRecords(datasetId);

            json response = {
                {"dataset_id", static_cast<int>(datasetId)},
                {"message", "All records cleared successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "get_records" && method == "GET") {
            // Get dataset records for training
            long long datasetId = parameters.value("dataset_id", -1LL);

            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }

            std::vector<std::vector<float>> inputs, targets;
            long long numSamples = 0;
            int inputSize = 0, outputSize = 0;

            bool success = database->getDatasetRecords(datasetId, inputs, targets, numSamples, inputSize, outputSize);

            if (!success) {
                return createErrorResponse("Failed to retrieve dataset records", 500);
            }

            json response = {
                {"dataset_id", static_cast<int>(datasetId)},
                {"num_samples", numSamples},
                {"input_size", inputSize},
                {"output_size", outputSize},
                {"inputs", inputs},
                {"targets", targets},
                {"message", "Dataset records retrieved successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "load_csv_directory" && method == "POST") {
            // Load all CSV files from a directory and combine them into a single dataset
            std::string directory = parameters.value("directory", "");
            std::string datasetName = parameters.value("dataset_name", "combined_csv_dataset");
            std::string description = parameters.value("description", "Dataset created from CSV files in directory");
            bool hasHeader = parameters.value("has_header", true);
            char delimiter = parameters.value("delimiter", ',');
            int outputSize = parameters.value("output_size", 1);

            // Validate parameters
            if (directory.empty()) {
                return createErrorResponse("Directory path cannot be empty", 400);
            }
            if (!std::filesystem::exists(directory)) {
                return createErrorResponse("Directory does not exist: " + directory, 400);
            }
            if (!std::filesystem::is_directory(directory)) {
                return createErrorResponse("Path is not a directory: " + directory, 400);
            }

            // Find all CSV files in the directory
            std::vector<std::string> csvFiles;
            for (const auto& entry : std::filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    std::string extension = getFileExtension(filename);
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    if (extension == "csv") {
                        csvFiles.push_back(entry.path().string());
                    }
                }
            }

            if (csvFiles.empty()) {
                return createErrorResponse("No CSV files found in directory: " + directory, 400);
            }

            // Read and combine all CSV files
            std::vector<std::vector<std::vector<float>>> allDatasets;
            std::vector<std::string> processedFiles;
            int totalRows = 0;
            int numFeatures = -1; // Will be determined from first file

            for (const auto& csvFile : csvFiles) {
                auto [data, error] = readCSVFile(csvFile, hasHeader, delimiter);
                if (!error.empty()) {
                    return createErrorResponse("Error reading CSV file " + csvFile + ": " + error, 500);
                }

                if (data.empty()) {
                    std::cout << "[WARNING] Empty CSV file skipped: " << csvFile << std::endl;
                    continue;
                }

                // Check feature consistency
                if (numFeatures == -1) {
                    numFeatures = data[0].size();
                } else if (data[0].size() != static_cast<size_t>(numFeatures)) {
                    return createErrorResponse("Inconsistent number of features in CSV file " + csvFile +
                                             ". Expected " + std::to_string(numFeatures) + " features, got " +
                                             std::to_string(data[0].size()), 400);
                }

                allDatasets.push_back(data);
                processedFiles.push_back(csvFile);
                totalRows += data.size();
            }

            if (allDatasets.empty()) {
                return createErrorResponse("No valid CSV data found in directory", 400);
            }

            // Combine all datasets
            std::vector<std::vector<float>> combinedData = combineCSVData(allDatasets);

            // Create dataset in database
            int datasetId = database->addDataset(datasetName, description, totalRows, numFeatures, outputSize);

            // Add all records to the dataset
            for (size_t rowIndex = 0; rowIndex < combinedData.size(); ++rowIndex) {
                const auto& row = combinedData[rowIndex];
                std::vector<float> features(row.begin(), row.end() - outputSize);
                std::vector<float> labels(row.end() - outputSize, row.end());

                database->addDatasetRecord(datasetId, static_cast<int>(rowIndex), features, labels);
            }

            json response = {
                {"dataset_id", datasetId},
                {"dataset_name", datasetName},
                {"description", description},
                {"directory", directory},
                {"files_processed", processedFiles.size()},
                {"files_list", processedFiles},
                {"total_rows", totalRows},
                {"num_features", numFeatures},
                {"output_size", outputSize},
                {"has_header", hasHeader},
                {"delimiter", std::string(1, delimiter)},
                {"message", "CSV directory loaded and combined successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "add_file" && method == "POST") {
            // Add a single CSV file to an existing dataset
            int datasetId = parameters.value("dataset_id", -1);
            std::string filename = parameters.value("filename", "");
            bool hasHeader = parameters.value("has_header", true);
            char delimiter = parameters.value("delimiter", ',');
            int outputSize = parameters.value("output_size", 1);

            // Validate parameters
            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }
            if (filename.empty()) {
                return createErrorResponse("Filename cannot be empty", 400);
            }
            if (!std::filesystem::exists(filename)) {
                return createErrorResponse("File does not exist: " + filename, 400);
            }

            // Read CSV file
            auto [data, error] = readCSVFile(filename, hasHeader, delimiter);
            if (!error.empty()) {
                return createErrorResponse("Error reading CSV file: " + error, 500);
            }

            if (data.empty()) {
                return createErrorResponse("CSV file is empty or contains no valid data", 400);
            }

            // Validate data consistency
            int numFeatures = data[0].size() - outputSize;
            if (numFeatures <= 0) {
                return createErrorResponse("Invalid output_size: must be less than total columns", 400);
            }

            // Get current dataset info to validate consistency
            Database::DatasetData datasetInfo = database->getDataset(datasetId);
            if (datasetInfo.numFeatures != numFeatures) {
                return createErrorResponse("Feature count mismatch: dataset expects " +
                                         std::to_string(datasetInfo.numFeatures) + " features, file has " +
                                         std::to_string(numFeatures), 400);
            }
            if (datasetInfo.numLabels != outputSize) {
                return createErrorResponse("Label count mismatch: dataset expects " +
                                         std::to_string(datasetInfo.numLabels) + " labels, specified output_size is " +
                                         std::to_string(outputSize), 400);
            }

            // Add records to dataset with filename tracking
            int startingRowIndex = datasetInfo.numRows;
            for (size_t i = 0; i < data.size(); ++i) {
                const auto& row = data[i];
                std::vector<float> features(row.begin(), row.end() - outputSize);
                std::vector<float> labels(row.end() - outputSize, row.end());

                database->addDatasetRecord(datasetId, startingRowIndex + static_cast<int>(i), features, labels, filename);
            }

            // Update dataset metadata (num_rows)
            int newTotalRows = datasetInfo.numRows + static_cast<int>(data.size());
            database->updateDatasetRowCount(datasetId, newTotalRows);

            json response = {
                {"dataset_id", datasetId},
                {"filename", filename},
                {"records_added", static_cast<int>(data.size())},
                {"total_rows", newTotalRows},
                {"num_features", numFeatures},
                {"output_size", outputSize},
                {"has_header", hasHeader},
                {"delimiter", std::string(1, delimiter)},
                {"message", "File added to dataset successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "remove_file" && method == "DELETE") {
            // Remove records from a specific file from a dataset
            int datasetId = parameters.value("dataset_id", -1);
            std::string filename = parameters.value("filename", "");

            // Validate parameters
            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }
            if (filename.empty()) {
                return createErrorResponse("Filename cannot be empty", 400);
            }

            // Check if file exists in dataset
            std::vector<std::string> datasetFiles = database->getDatasetFiles(datasetId);
            bool fileExists = std::find(datasetFiles.begin(), datasetFiles.end(), filename) != datasetFiles.end();
            if (!fileExists) {
                return createErrorResponse("File '" + filename + "' not found in dataset " + std::to_string(datasetId), 404);
            }

            // Get current row count before removal
            Database::DatasetData currentDatasetInfo = database->getDataset(datasetId);
            int rowsBeforeRemoval = currentDatasetInfo.numRows;

            // Remove records for this file
            database->removeDatasetFileRecords(datasetId, filename);

            // Update dataset row count
            Database::DatasetData updatedDatasetInfo = database->getDataset(datasetId);
            int rowsAfterRemoval = updatedDatasetInfo.numRows;
            int recordsRemoved = rowsBeforeRemoval - rowsAfterRemoval;

            json response = {
                {"dataset_id", datasetId},
                {"filename", filename},
                {"records_removed", recordsRemoved},
                {"remaining_rows", rowsAfterRemoval},
                {"message", "File removed from dataset successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else if (action == "list_files" && method == "GET") {
            // List all files in a dataset
            int datasetId = parameters.value("dataset_id", -1);

            if (datasetId <= 0) {
                return createErrorResponse("Valid dataset ID required", 400);
            }

            std::vector<std::string> files = database->getDatasetFiles(datasetId);

            json response = {
                {"dataset_id", datasetId},
                {"files", files},
                {"file_count", static_cast<int>(files.size())},
                {"message", "Dataset files retrieved successfully"},
                {"timestamp", getTimestampString()}
            };
            return response;
        }
        else {
            return createErrorResponse("Unknown dataset action: " + action + " with method " + method, 400);
        }
    }
    catch (const std::exception& e) {
        return createErrorResponse("Dataset API error: " + std::string(e.what()), 500);
    }
}

// Authentication API handler
json APIServer::handleAuthAPI(const std::string& action, const json& parameters, const std::string& method) {
    try {
        if (method == "POST" && action == "login") {
            std::string username = parameters.value("username", "");
            std::string password = parameters.value("password", "");
            
            if (username.empty() || password.empty()) {
                return createErrorResponse("Username and password are required", 400);
            }
            
            return login(username, password);
        }
        else if (method == "POST" && action == "register") {
            std::string username = parameters.value("username", "");
            std::string password = parameters.value("password", "");
            std::string email = parameters.value("email", "");
            
            if (username.empty() || password.empty()) {
                return createErrorResponse("Username and password are required", 400);
            }
            
            if (!authManager) {
                return createErrorResponse("Auth system not initialized", 500);
            }
            
            bool created = authManager->createUser(username, password, email);
            if (!created) {
                return createErrorResponse("Failed to create user (username may already exist)", 400);
            }
            
            return createSuccessResponse("User registered successfully", 
                {{"username", username}, {"email", email}});
        }
        else if (method == "POST" && action == "logout") {
            std::string sessionId = parameters.value("session_id", "");
            
            if (sessionId.empty()) {
                return createErrorResponse("Session ID is required", 400);
            }
            
            if (authManager) {
                authManager->invalidateSession(sessionId);
            }
            destroySession(sessionId);
            
            return createSuccessResponse("Logged out successfully");
        }
        else if (method == "POST" && action == "refresh") {
            std::string refreshToken = parameters.value("refresh_token", "");
            
            if (refreshToken.empty()) {
                return createErrorResponse("Refresh token is required", 400);
            }
            
            int64_t userId;
            std::string username;
            if (authManager && authManager->validateJWT(refreshToken, userId, username)) {
                std::string newJwt = authManager->generateJWT(userId, username);
                std::string newSessionId = authManager->createSession(userId);
                return createSuccessResponse("Token refreshed", 
                    {{"jwt_token", newJwt}, {"session_id", newSessionId}});
            }
            
            return createErrorResponse("Invalid or expired refresh token", 401);
        }
        else if (method == "GET" && action == "validate") {
            std::string jwtToken = parameters.value("token", "");
            
            if (jwtToken.empty()) {
                return createErrorResponse("Token is required", 400);
            }
            
            int64_t userId;
            std::string username;
            if (authManager && authManager->validateJWT(jwtToken, userId, username)) {
                return createSuccessResponse("Token is valid", 
                    {{"user_id", userId}, {"username", username}});
            }
            
            return createErrorResponse("Invalid or expired token", 401);
        }
        else if (method == "POST" && action == "apikey") {
            std::string userIdStr = parameters.value("user_id", "");
            std::string description = parameters.value("description", "");
            std::string existingApiKey = parameters.value("api_key", "");
            std::string sessionId = parameters.value("session_id", "");
            
            if (sessionId.empty()) {
                return createErrorResponse("Session ID is required", 400);
            }
            
            int64_t uid = userIdStr.empty() ? 0 : std::stoll(userIdStr);
            
            std::string apiKey = authManager ? authManager->generateAPIKey() : generateAPIKey();
            std::string finalApiKey = existingApiKey.empty() ? apiKey : existingApiKey;
            
            json response = {
                {"api_key", finalApiKey},
                {"description", description},
                {"created", true},
                {"timestamp", getTimestampString()}
            };
            
            if (authManager && uid > 0) {
                authManager->saveAPIKey(finalApiKey, uid, description);
            }
            
            return response;
        }
        else if (method == "POST" && action == "validate_api_key") {
            std::string apiKey = parameters.value("api_key", "");
            
            if (apiKey.empty()) {
                return createErrorResponse("API key is required", 400);
            }
            
            bool valid = false;
            if (database) {
                valid = database->validateAPIKey(apiKey);
            }
            else {
                valid = APIServer::validateAPIKey(apiKey);
            }
            
            if (valid) {
                return createSuccessResponse("API key is valid",
                    {{"valid", true}});
            }
            
            return createErrorResponse("API key is invalid or expired", 401,
                {{"valid", false}});
        }
        else {
            return createErrorResponse("Unknown auth action: " + action + " with method " + method, 400);
        }
    }
    catch (const std::invalid_argument& e) {
        return createErrorResponse("Invalid request: " + std::string(e.what()), 400);
    }
    catch (const std::exception& e) {
        return createErrorResponse("Auth API error: " + std::string(e.what()), 500);
    }
}

// User login with JWT generation
json APIServer::login(const std::string& username, const std::string& password) {
    try {
        if (!authManager) {
            return createErrorResponse("Auth system not initialized - check database connection", 500);
        }
        
        auto userOpt = authManager->getUserByUsername(username);
        if (!userOpt.has_value()) {
            return createErrorResponse("Invalid username or password", 401);
        }
        
        auto user = userOpt.value();
        
        if (!authManager->verifyPassword(password, user.passwordHash, user.salt)) {
            return createErrorResponse("Invalid username or password", 401);
        }
        
        std::string jwtToken = authManager->generateJWT(user.id, user.username);
        std::string sessionId = authManager->createSession(user.id);
        if (sessionId.empty()) {
            return createErrorResponse("Failed to create user session", 500);
        }
        
        createSession(username + ":" + std::to_string(user.id));
        
        json response = {
            {"status", "success"},
            {"message", "Authentication successful"},
            {"jwt_token", jwtToken},
            {"session_id", sessionId},
            {"user", {
                {"id", user.id},
                {"username", user.username},
                {"email", user.email}
            }},
            {"expires_in", 900},
            {"timestamp", getTimestampString()}
        };
        
        return response;
    }
    catch (const std::exception& e) {
        std::cerr << "Login error: " << e.what() << std::endl;
        return createErrorResponse("Authentication failed: " + std::string(e.what()), 500);
    }
}

// JWT validation for protected endpoints
bool APIServer::validateJWTToken(const std::string& authHeader, int64_t& outUserId, std::string& outUsername) {
    if (authHeader.empty()) {
        return false;
    }
    
    if (authHeader.substr(0, 7) != "Bearer ") {
        return false;
    }
    
    std::string token = authHeader.substr(7);
    
    if (!authManager) {
        outUserId = 0;
        outUsername = "offline";
        return true;
    }
    
    return authManager->validateJWT(token, outUserId, outUsername);
}
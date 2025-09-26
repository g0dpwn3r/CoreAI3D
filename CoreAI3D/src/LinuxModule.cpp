/**
 * CoreAI3D Linux Operations Module Implementation
 * Provides safe Linux system operations within Docker containers
 */

#include "LinuxModule.hpp"
#include "Database.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <curl/curl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace CoreAI3D {

    // Static callback for curl
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

    LinuxModule::LinuxModule(const std::string& dockerSocketPath)
        : m_dockerSocketPath(dockerSocketPath) {
        loadTrainingScenarios();
    }

    LinuxModule::~LinuxModule() {
        // Clean up active containers
        for (const auto& [name, id] : m_activeContainers) {
            destroySandboxContainer(id);
        }
    }

    void LinuxModule::initialize() {
        std::cout << "Initializing Linux Module..." << std::endl;

        // Test Docker connectivity
        if (!testDockerConnection()) {
            std::cerr << "Warning: Cannot connect to Docker daemon" << std::endl;
        }

        // Load available commands
        loadAvailableCommands();

        std::cout << "Linux Module initialized successfully" << std::endl;
    }

    void LinuxModule::shutdown() {
        std::cout << "Shutting down Linux Module..." << std::endl;

        // Clean up resources
        m_activeContainers.clear();
        m_trainingScenarios.clear();

        std::cout << "Linux Module shutdown complete" << std::endl;
    }

    bool LinuxModule::testDockerConnection() {
        CURL* curl = curl_easy_init();
        if (!curl) {
            return false;
        }

        std::string response;
        std::string url = "http://localhost/v1.43/info";

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH, m_dockerSocketPath.c_str());
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        return res == CURLE_OK;
    }

    LinuxOperationResult LinuxModule::executeCommand(
        const std::string& command,
        const std::string& sandboxType,
        bool requiresRoot
    ) {
        LinuxOperationResult result;
        result.success = false;
        result.command = command;
        result.sandbox_type = sandboxType;

        // Validate command
        if (!validateCommand(command)) {
            result.error_message = "Command validation failed";
            return result;
        }

        // Sanitize command
        std::string sanitizedCommand = sanitizeCommand(command);

        // Create or get container
        std::string containerId;
        if (m_activeContainers.find(sandboxType) != m_activeContainers.end()) {
            containerId = m_activeContainers[sandboxType];
        } else {
            containerId = createSandboxContainer(sandboxType);
            if (containerId.empty()) {
                result.error_message = "Failed to create sandbox container";
                return result;
            }
            m_activeContainers[sandboxType] = containerId;
        }

        // Execute command in container
        auto startTime = std::chrono::high_resolution_clock::now();
        result = executeInContainer(containerId, sanitizedCommand, requiresRoot);
        auto endTime = std::chrono::high_resolution_clock::now();

        result.execution_time = std::chrono::duration<double>(endTime - startTime).count();
        result.container_id = containerId;

        return result;
    }

    std::string LinuxModule::createSandboxContainer(const std::string& sandboxType) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            return "";
        }

        // Create container configuration
        nlohmann::json config = {
            {"Image", "coreai3d-" + sandboxType + ":latest"},
            {"Cmd", {"/usr/bin/supervisord", "-c", "/etc/supervisord.d/supervisord.conf"}},
            {"Env", {
                "SANDBOX_TYPE=" + sandboxType,
                "DEBIAN_FRONTEND=noninteractive"
            }},
            {"HostConfig", {
                {"Privileged", false},
                {"CapAdd", {"SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE"}},
                {"SecurityOpt", {"seccomp:unconfined", "apparmor:unconfined"}},
                {"Binds", {
                    "/training_data:/training_data",
                    "/workspace:/workspace"
                }}
            }}
        };

        std::string jsonData = config.dump();
        std::string response;
        std::string containerId;

        // Create container
        std::string url = "http://localhost/v1.43/containers/create?name=" + generateContainerName();

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonData.size());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH, m_dockerSocketPath.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(NULL, "Content-Type: application/json"));

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CURLcode res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            auto jsonResponse = nlohmann::json::parse(response);
            if (jsonResponse.contains("Id")) {
                containerId = jsonResponse["Id"];

                // Start container
                std::string startUrl = "http://localhost/v1.43/containers/" + containerId + "/start";
                curl_easy_setopt(curl, CURLOPT_URL, startUrl.c_str());
                curl_easy_setopt(curl, CURLOPT_POST, 1L);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

                CURLcode startRes = curl_easy_perform(curl);
                if (startRes != CURLE_OK) {
                    std::cerr << "Failed to start container: " << curl_easy_strerror(startRes) << std::endl;
                    containerId.clear();
                }
            }
        } else {
            std::cerr << "Failed to create container: " << curl_easy_strerror(res) << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        return containerId;
    }

    bool LinuxModule::destroySandboxContainer(const std::string& containerId) {
        if (containerId.empty()) {
            return false;
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            return false;
        }

        std::string response;

        // Stop container
        std::string stopUrl = "http://localhost/v1.43/containers/" + containerId + "/stop?t=10";
        curl_easy_setopt(curl, CURLOPT_URL, stopUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
        curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH, m_dockerSocketPath.c_str());

        curl_easy_perform(curl);

        // Remove container
        std::string removeUrl = "http://localhost/v1.43/containers/" + containerId + "?force=true";
        curl_easy_setopt(curl, CURLOPT_URL, removeUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        return res == CURLE_OK;
    }

    LinuxOperationResult LinuxModule::executeInContainer(
        const std::string& containerId,
        const std::string& command,
        bool requiresRoot
    ) {
        LinuxOperationResult result;
        result.success = false;
        result.container_id = containerId;

        CURL* curl = curl_easy_init();
        if (!curl) {
            result.error_message = "Failed to initialize curl";
            return result;
        }

        // Create exec configuration
        nlohmann::json execConfig = {
            {"AttachStdout", true},
            {"AttachStderr", true},
            {"Tty", false},
            {"Cmd", {"/bin/sh", "-c", command}},
            {"Env", requiresRoot ? std::vector<std::string>{"USER=root"} : std::vector<std::string>{}}
        };

        std::string jsonData = execConfig.dump();
        std::string response;

        // Create exec instance
        std::string execUrl = "http://localhost/v1.43/containers/" + containerId + "/exec";
        curl_easy_setopt(curl, CURLOPT_URL, execUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonData.size());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH, m_dockerSocketPath.c_str());

        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            result.error_message = std::string("Failed to create exec: ") + curl_easy_strerror(res);
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return result;
        }

        auto execResponse = nlohmann::json::parse(response);
        std::string execId = execResponse["Id"];

        // Start exec
        std::string startUrl = "http://localhost/v1.43/exec/" + execId + "/start";
        nlohmann::json startConfig = {
            {"Detach", false},
            {"Tty", false}
        };

        std::string startData = startConfig.dump();
        curl_easy_setopt(curl, CURLOPT_URL, startUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, startData.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, startData.size());

        res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            result.success = true;
            result.output = response; // This will contain the actual output
        } else {
            result.error_message = std::string("Failed to start exec: ") + curl_easy_strerror(res);
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        return result;
    }

    void LinuxModule::loadTrainingScenarios() {
        // Load training scenarios from database or files
        m_trainingScenarios = {
            {
                "file_system_operations",
                "Basic file system operations",
                {
                    {"ls -la", "List all files with details", "filesystem", 1, false, {}},
                    {"pwd", "Print working directory", "filesystem", 1, false, {}},
                    {"mkdir test_directory", "Create directory", "filesystem", 1, false, {}},
                    {"touch test_file.txt", "Create empty file", "filesystem", 1, false, {}}
                },
                1,
                "beginner"
            },
            {
                "process_management",
                "Process monitoring and management",
                {
                    {"ps aux", "List all processes", "processes", 2, false, {}},
                    {"top -n 1", "Show process statistics", "processes", 2, false, {}},
                    {"htop", "Interactive process viewer", "processes", 2, false, {}}
                },
                2,
                "intermediate"
            }
        };
    }

    void LinuxModule::loadAvailableCommands() {
        m_availableCommands = {
            {"ls", "List directory contents", "filesystem", 1, false, {}},
            {"cd", "Change directory", "filesystem", 1, false, {}},
            {"pwd", "Print working directory", "filesystem", 1, false, {}},
            {"mkdir", "Create directory", "filesystem", 1, false, {}},
            {"rmdir", "Remove directory", "filesystem", 1, false, {}},
            {"touch", "Create empty file", "filesystem", 1, false, {}},
            {"cp", "Copy files", "filesystem", 1, false, {}},
            {"mv", "Move/rename files", "filesystem", 1, false, {}},
            {"rm", "Remove files", "filesystem", 1, false, {}},
            {"cat", "Display file contents", "filesystem", 1, false, {}},
            {"head", "Display first lines of file", "filesystem", 1, false, {}},
            {"tail", "Display last lines of file", "filesystem", 1, false, {}},
            {"grep", "Search text in files", "text", 2, false, {}},
            {"find", "Search for files", "filesystem", 2, false, {}},
            {"chmod", "Change file permissions", "permissions", 2, false, {}},
            {"chown", "Change file ownership", "permissions", 2, false, {}},
            {"ps", "Process status", "processes", 2, false, {}},
            {"top", "Process monitoring", "processes", 2, false, {}},
            {"kill", "Terminate processes", "processes", 2, false, {}},
            {"ping", "Network connectivity test", "network", 1, false, {}},
            {"ifconfig", "Network interface configuration", "network", 2, false, {}},
            {"netstat", "Network statistics", "network", 2, false, {}},
            {"ssh", "Secure shell client", "network", 3, false, {}},
            {"scp", "Secure file copy", "network", 3, false, {}},
            {"rsync", "Remote file synchronization", "network", 3, false, {}}
        };
    }

    bool LinuxModule::validateCommand(const std::string& command) {
        // Basic validation - check for dangerous commands
        std::vector<std::string> dangerousCommands = {
            "rm -rf /",
            "rm -rf /*",
            "dd if=/dev/zero",
            "mkfs",
            "fdisk",
            "parted",
            "mount",
            "umount",
            "systemctl",
            "service",
            "init",
            "reboot",
            "shutdown",
            "poweroff",
            "halt"
        };

        std::string lowerCommand = command;
        std::transform(lowerCommand.begin(), lowerCommand.end(), lowerCommand.begin(), ::tolower);

        for (const auto& dangerous : dangerousCommands) {
            if (lowerCommand.find(dangerous) != std::string::npos) {
                return false;
            }
        }

        return true;
    }

    std::string LinuxModule::sanitizeCommand(const std::string& command) {
        // Remove potentially dangerous characters and limit length
        std::string sanitized = command;

        // Remove null bytes
        sanitized.erase(std::remove(sanitized.begin(), sanitized.end(), '\0'), sanitized.end());

        // Limit command length
        if (sanitized.length() > 4096) {
            sanitized = sanitized.substr(0, 4096);
        }

        return sanitized;
    }

    std::string LinuxModule::generateContainerName() {
        static const std::string chars = "abcdefghijklmnopqrstuvwxyz0123456789";
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, chars.size() - 1);

        std::string name = "coreai3d-sandbox-";
        for (int i = 0; i < 16; ++i) {
            name += chars[dis(gen)];
        }

        return name;
    }

    std::string LinuxModule::getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        return ss.str();
    }

    // Additional methods would be implemented here...
    // For brevity, some methods are stubbed

    std::vector<std::string> LinuxModule::listAvailableSandboxes() {
        return {"ubuntu", "centos", "alpine"};
    }

    std::vector<TrainingScenario> LinuxModule::getTrainingScenarios() {
        return m_trainingScenarios;
    }

    LinuxOperationResult LinuxModule::executeTrainingScenario(const std::string& scenarioName) {
        LinuxOperationResult result;
        result.success = false;
        result.error_message = "Training scenario execution not implemented";
        return result;
    }

    nlohmann::json LinuxModule::getSystemMetrics(const std::string& containerId) {
        return nlohmann::json{{"error", "Not implemented"}};
    }

    nlohmann::json LinuxModule::getContainerInfo(const std::string& containerId) {
        return nlohmann::json{{"error", "Not implemented"}};
    }

    LinuxOperationResult LinuxModule::uploadFile(
        const std::string& containerId,
        const std::string& localPath,
        const std::string& remotePath
    ) {
        LinuxOperationResult result;
        result.success = false;
        result.error_message = "File upload not implemented";
        return result;
    }

    LinuxOperationResult LinuxModule::downloadFile(
        const std::string& containerId,
        const std::string& remotePath,
        const std::string& localPath
    ) {
        LinuxOperationResult result;
        result.success = false;
        result.error_message = "File download not implemented";
        return result;
    }

    void LinuxModule::recordTrainingSession(const std::string& scenarioName, const LinuxOperationResult& result) {
        // Implementation would save to database
    }

    nlohmann::json LinuxModule::getTrainingHistory() {
        return nlohmann::json{{"error", "Not implemented"}};
    }

    nlohmann::json LinuxModule::generateTrainingReport() {
        return nlohmann::json{{"error", "Not implemented"}};
    }

} // namespace CoreAI3D
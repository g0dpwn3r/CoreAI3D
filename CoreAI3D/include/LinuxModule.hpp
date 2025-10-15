/**
 * CoreAI3D Linux Operations Module
 * Provides safe Linux system operations within Docker containers
 */

#ifndef LINUX_MODULE_HPP
#define LINUX_MODULE_HPP

#include "Core.hpp"
#include "ModuleOrchestrator.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <future>

namespace CoreAI3D {

    struct LinuxCommand {
        std::string command;
        std::string description;
        std::string category;
        int difficulty;
        bool requires_root;
        std::vector<std::string> dependencies;
    };

    struct LinuxOperationResult {
        bool success;
        std::string command;
        std::string output;
        std::string error_message;
        int exit_code;
        double execution_time;
        std::string container_id;
        std::string sandbox_type;
    };

    struct TrainingScenario {
        std::string name;
        std::string description;
        std::vector<LinuxCommand> commands;
        int difficulty_level;
        std::string category;
    };

    class LinuxModule {
    public:
        LinuxModule(const std::string& dockerSocketPath = "/var/run/docker.sock");
        ~LinuxModule();

        // Module interface
        void initialize();
        void shutdown();
        std::string getName() const { return "LinuxModule"; }
        std::string getVersion() const { return "1.0.0"; }
        std::string getType() const { return "SYSTEM"; }

        // Linux operations
        LinuxOperationResult executeCommand(
            const std::string& command,
            const std::string& sandboxType = "ubuntu",
            bool requiresRoot = false
        );

        std::vector<std::string> listAvailableSandboxes();
        std::vector<TrainingScenario> getTrainingScenarios();
        LinuxOperationResult executeTrainingScenario(const std::string& scenarioName);

        // System monitoring
        nlohmann::json getSystemMetrics(const std::string& containerId = "");
        nlohmann::json getContainerInfo(const std::string& containerId);

        // File operations
        LinuxOperationResult uploadFile(
            const std::string& containerId,
            const std::string& localPath,
            const std::string& remotePath
        );

        LinuxOperationResult downloadFile(
            const std::string& containerId,
            const std::string& remotePath,
            const std::string& localPath
        );

        // Training and learning
        void recordTrainingSession(const std::string& scenarioName, const LinuxOperationResult& result);
        nlohmann::json getTrainingHistory();
        nlohmann::json generateTrainingReport();

    private:
        std::string m_dockerSocketPath;
        std::map<std::string, std::string> m_activeContainers;
        std::vector<TrainingScenario> m_trainingScenarios;
        std::vector<LinuxCommand> m_availableCommands;

        // Docker operations
        std::string createSandboxContainer(const std::string& sandboxType);
        bool destroySandboxContainer(const std::string& containerId);
        LinuxOperationResult executeInContainer(
            const std::string& containerId,
            const std::string& command,
            bool requiresRoot = false
        );

        // Training data management
        void loadTrainingScenarios();
        bool testDockerConnection();
        void loadAvailableCommands();
        void saveTrainingResult(const std::string& scenarioName, const LinuxOperationResult& result);

        // Command validation
        bool validateCommand(const std::string& command);
        std::vector<std::string> getCommandSuggestions(const std::string& partialCommand);

        // Security checks
        bool isCommandAllowed(const std::string& command);
        std::string sanitizeCommand(const std::string& command);

        // Utility functions
        std::string generateContainerName();
        std::string getCurrentTimestamp();
        nlohmann::json parseContainerLogs(const std::string& logs);
    };

} // namespace CoreAI3D

#endif // LINUX_MODULE_HPP
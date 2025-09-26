#ifndef MODULE_ORCHESTRATOR_HPP
#define MODULE_ORCHESTRATOR_HPP

#include "main.hpp"
#include "Core.hpp"
#include "VisionModule.hpp"
#include "AudioModule.hpp"
#include "SystemModule.hpp"
#include "WebModule.hpp"
#include "MathModule.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

// Forward declarations for module types
class VisionModule;
class AudioModule;
class SystemModule;
class WebModule;
class MathModule;

struct ModuleInfo {
    std::string name;
    std::string type;
    std::string description;
    bool isActive;
    bool isAvailable;
    float performanceScore;
    size_t memoryUsage;
    std::map<std::string, std::string> capabilities;
    std::map<std::string, std::string> status;
};

struct OrchestrationTask {
    std::string taskId;
    std::string taskType;
    std::string description;
    std::map<std::string, std::string> parameters;
    std::string priority;
    std::string status;
    std::string assignedModule;
    std::string result;
    std::string errorMessage;
    time_t createdTime;
    time_t startTime;
    time_t endTime;
    std::vector<std::string> dependencies;
    std::vector<std::string> subtasks;
};

struct SystemState {
    std::vector<float> numericalState;
    std::map<std::string, float> metrics;
    std::map<std::string, std::string> status;
    std::vector<std::string> activeTasks;
    std::vector<std::string> availableResources;
    float overallHealth;
    float performanceIndex;
};

class ModuleOrchestrator {
private:
    std::unique_ptr<CoreAI> orchestratorCore;
    std::string orchestratorName;
    bool isInitialized;
    std::atomic<bool> isRunning;

    // Module management
    std::map<std::string, std::unique_ptr<VisionModule>> visionModules;
    std::map<std::string, std::unique_ptr<AudioModule>> audioModules;
    std::map<std::string, std::unique_ptr<SystemModule>> systemModules;
    std::map<std::string, std::unique_ptr<WebModule>> webModules;
    std::map<std::string, std::unique_ptr<MathModule>> mathModules;

    // Task management
    std::queue<OrchestrationTask> taskQueue;
    std::map<std::string, OrchestrationTask> activeTasks;
    std::map<std::string, OrchestrationTask> completedTasks;
    std::mutex taskMutex;
    std::condition_variable taskCondition;
    std::unique_ptr<std::thread> taskProcessorThread;

    // Resource management
    size_t maxMemoryUsage;
    int maxConcurrentTasks;
    std::map<std::string, float> resourceLimits;
    std::map<std::string, float> currentResourceUsage;

    // Communication and coordination
    std::map<std::string, std::string> interModuleChannels;
    std::map<std::string, std::function<void(const std::string&)>> eventHandlers;
    std::map<std::string, std::vector<std::string>> moduleDependencies;

protected:
    // Core orchestration functions
    virtual std::vector<float> processOrchestrationData(const std::vector<float>& inputData);
    virtual std::string determineOptimalModule(const OrchestrationTask& task);
    virtual std::vector<std::string> planTaskExecution(const OrchestrationTask& task);
    virtual bool validateTaskPrerequisites(const OrchestrationTask& task);

    // Module coordination
    virtual bool coordinateVisionAudio(const std::string& visionModule, const std::string& audioModule);
    virtual bool coordinateSystemWeb(const std::string& systemModule, const std::string& webModule);
    virtual bool coordinateMathVision(const std::string& mathModule, const std::string& visionModule);
    virtual bool coordinateMultiModalProcessing(const std::vector<std::string>& moduleNames);

    // Resource allocation
    virtual bool allocateResources(const std::string& moduleName, const std::map<std::string, float>& requirements);
    virtual void releaseResources(const std::string& moduleName);
    virtual bool checkResourceAvailability(const std::map<std::string, float>& requirements);

    // Task processing
    virtual void processTaskQueue();
    virtual void executeTask(OrchestrationTask& task);
    virtual void handleTaskResult(const OrchestrationTask& task);
    virtual void handleTaskError(const OrchestrationTask& task, const std::string& error);

    // Inter-module communication
    virtual void sendMessageToModule(const std::string& moduleName, const std::string& message);
    virtual std::string receiveMessageFromModule(const std::string& moduleName);
    virtual void broadcastMessage(const std::string& message);

public:
    // Constructor
    ModuleOrchestrator(const std::string& name);
    virtual ~ModuleOrchestrator();

    // Initialization
    bool initialize(const std::string& configPath = "");
    void setMaxMemoryUsage(size_t maxMemory);
    void setMaxConcurrentTasks(int maxTasks);
    void setResourceLimits(const std::map<std::string, float>& limits);

    // Module management
    bool addVisionModule(const std::string& name, std::unique_ptr<VisionModule> module);
    bool addAudioModule(const std::string& name, std::unique_ptr<AudioModule> module);
    bool addSystemModule(const std::string& name, std::unique_ptr<SystemModule> module);
    bool addWebModule(const std::string& name, std::unique_ptr<WebModule> module);
    bool addMathModule(const std::string& name, std::unique_ptr<MathModule> module);

    bool removeModule(const std::string& moduleName);
    bool activateModule(const std::string& moduleName);
    bool deactivateModule(const std::string& moduleName);

    // Module access
    VisionModule* getVisionModule(const std::string& name);
    AudioModule* getAudioModule(const std::string& name);
    SystemModule* getSystemModule(const std::string& name);
    WebModule* getWebModule(const std::string& name);
    MathModule* getMathModule(const std::string& name);

    // Task orchestration
    std::string submitTask(const std::string& taskType, const std::string& description,
                          const std::map<std::string, std::string>& parameters,
                          const std::string& priority = "normal",
                          const std::vector<std::string>& dependencies = {});
    bool cancelTask(const std::string& taskId);
    OrchestrationTask getTaskStatus(const std::string& taskId);
    std::vector<OrchestrationTask> getActiveTasks();
    std::vector<OrchestrationTask> getTaskHistory();

    // Multi-modal processing
    std::vector<float> processMultiModalData(const std::string& dataType, const std::vector<float>& data,
                                            const std::vector<std::string>& requiredModules);
    std::string analyzeMultiModalContent(const std::string& content, const std::vector<std::string>& analysisTypes);
    std::vector<float> extractMultiModalFeatures(const std::string& contentType, const std::vector<float>& data);

    // Intelligent routing
    std::string routeToOptimalModule(const std::string& taskType, const std::map<std::string, std::string>& parameters);
    std::vector<std::string> getCapableModules(const std::string& capability);
    std::vector<std::string> getAvailableModules();

    // System state management
    SystemState getCurrentSystemState();
    std::vector<float> getSystemStateAsNumbers();
    bool isSystemHealthy();
    float getOverallPerformanceScore();

    // Resource monitoring
    std::map<std::string, float> getResourceUsage();
    std::map<std::string, ModuleInfo> getModuleInformation();
    std::vector<std::string> getSystemDiagnostics();

    // Event handling
    void registerEventHandler(const std::string& eventType, std::function<void(const std::string&)> handler);
    void unregisterEventHandler(const std::string& eventType);
    void triggerEvent(const std::string& eventType, const std::string& eventData);

    // Communication interface
    bool sendInterModuleMessage(const std::string& fromModule, const std::string& toModule, const std::string& message);
    std::vector<std::string> getModuleMessages(const std::string& moduleName);
    void clearModuleMessages(const std::string& moduleName);

    // Coordination methods
    bool coordinateModules(const std::vector<std::string>& moduleNames, const std::string& coordinationType);
    bool synchronizeModules(const std::vector<std::string>& moduleNames);
    bool transferDataBetweenModules(const std::string& fromModule, const std::string& toModule,
                                   const std::string& dataType, const std::vector<float>& data);

    // Advanced orchestration
    std::string createComplexTask(const std::string& taskDescription,
                                 const std::map<std::string, std::string>& requirements,
                                 const std::vector<std::string>& constraints);
    bool executeComplexTask(const std::string& taskId);
    std::vector<std::string> breakDownComplexTask(const std::string& taskDescription);

    // Learning and adaptation
    bool learnFromTaskResults(const std::vector<OrchestrationTask>& completedTasks);
    bool optimizeModuleAssignment(const std::string& taskType);
    bool adaptToResourceConstraints();

    // Status and information
    bool isReady() const { return isInitialized; }
    std::string getOrchestratorName() const { return orchestratorName; }
    bool isOrchestrationRunning() const { return isRunning; }
    int getActiveTaskCount() const { return activeTasks.size(); }

    // Orchestration control
    void stopOrchestration();

    // Memory management
    void clearTaskHistory();
    void clearModuleCaches();
    size_t getTotalMemoryUsage() const;

    // Training interface for orchestration learning
    virtual bool trainOnOrchestrationData(const std::string& dataPath, int epochs = 10);
    virtual bool learnCoordinationPatterns(const std::vector<OrchestrationTask>& taskHistory);

    // Utility functions
    std::string generateTaskId();
    std::string getTimestampString();
    bool validateModuleCompatibility(const std::string& moduleA, const std::string& moduleB);
    std::vector<std::string> getSupportedTaskTypes();
};

#endif // MODULE_ORCHESTRATOR_HPP
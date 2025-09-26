#ifndef SYSTEM_MODULE_HPP
#define SYSTEM_MODULE_HPP

#include "main.hpp"
#include "Core.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <atomic>
#include <functional>

// Forward declarations for Windows API types
struct ProcessInfo {
    int processId;
    std::string processName;
    std::string executablePath;
    size_t memoryUsage;
    float cpuUsage;
    std::string status;
};

struct WindowInfo {
    void* handle;
    std::string title;
    int x, y, width, height;
    bool isVisible;
    bool isMinimized;
    std::string className;
};

struct FileSystemInfo {
    std::string path;
    std::string name;
    size_t size;
    std::string type;
    std::string permissions;
    std::string lastModified;
    bool isDirectory;
    bool isHidden;
    bool isSystem;
};

struct KeyboardMouseState {
    std::vector<bool> keysPressed; // 256 keys
    int mouseX, mouseY;
    bool leftButton, rightButton, middleButton;
    int wheelDelta;
};

struct AutomationTask {
    std::string taskId;
    std::string taskType;
    std::string description;
    std::map<std::string, std::string> parameters;
    std::string status;
    std::string result;
    std::string errorMessage;
    time_t createdTime;
    time_t startTime;
    time_t endTime;
};

class SystemModule {
private:
    std::unique_ptr<CoreAI> systemCore;
    std::string moduleName;
    bool isInitialized;
    std::atomic<bool> isRunning;

    // System state tracking
    std::map<int, ProcessInfo> monitoredProcesses;
    std::vector<WindowInfo> activeWindows;
    KeyboardMouseState currentInputState;
    std::vector<AutomationTask> taskHistory;

    // Automation parameters
    int automationDelay;
    bool enableSafetyChecks;
    int maxRetries;
    std::string logFilePath;

    // Thread management
    std::unique_ptr<std::thread> monitoringThread;
    std::unique_ptr<std::thread> automationThread;

protected:
    // Core system interaction functions
    virtual bool executeSystemCommand(const std::string& command, const std::vector<std::string>& args);
    virtual std::vector<float> getSystemStateVector();
    virtual bool applySystemAction(const std::vector<float>& actionVector);

    // Process management
    virtual std::vector<ProcessInfo> enumerateProcesses();
    virtual bool startProcess(const std::string& executable, const std::vector<std::string>& args);
    virtual bool stopProcess(int processId);
    virtual bool restartProcess(int processId);

    // Window management
    virtual std::vector<WindowInfo> enumerateWindows();
    virtual bool activateWindow(const std::string& windowTitle);
    virtual bool moveWindow(const std::string& windowTitle, int x, int y, int width, int height);
    virtual bool minimizeWindow(const std::string& windowTitle);
    virtual bool maximizeWindow(const std::string& windowTitle);
    virtual bool closeWindow(const std::string& windowTitle);

    // File system operations
    virtual std::vector<FileSystemInfo> listDirectory(const std::string& path);
    virtual bool createDirectory(const std::string& path);
    virtual bool deleteFile(const std::string& path);
    virtual bool copyFile(const std::string& source, const std::string& destination);
    virtual bool moveFile(const std::string& source, const std::string& destination);
    virtual std::vector<float> readFileAsNumbers(const std::string& path);
    virtual bool writeNumbersToFile(const std::string& path, const std::vector<float>& data);

    // Input simulation
    virtual bool simulateKeyPress(int keyCode, bool pressed);
    virtual bool simulateKeyCombination(const std::vector<int>& keyCodes);
    virtual bool simulateMouseClick(int x, int y, const std::string& button);
    virtual bool simulateMouseMove(int x, int y);
    virtual bool simulateMouseWheel(int delta);

    // Screen operations
    virtual std::vector<float> captureScreen();
    virtual std::vector<float> captureWindow(const std::string& windowTitle);
    virtual std::vector<float> captureRegion(int x, int y, int width, int height);

    // Registry operations (Windows)
    virtual std::string readRegistryValue(const std::string& keyPath, const std::string& valueName);
    virtual bool writeRegistryValue(const std::string& keyPath, const std::string& valueName, const std::string& value);
    virtual bool deleteRegistryValue(const std::string& keyPath, const std::string& valueName);

    // Service management
    virtual bool startService(const std::string& serviceName);
    virtual bool stopService(const std::string& serviceName);
    virtual bool restartService(const std::string& serviceName);
    virtual std::string getServiceStatus(const std::string& serviceName);

public:
    // Constructor
    SystemModule(const std::string& name);
    virtual ~SystemModule();

    // Initialization
    bool initialize(const std::string& configPath = "");
    void setAutomationDelay(int milliseconds);
    void setSafetyChecks(bool enable);
    void setMaxRetries(int retries);
    void setLogFile(const std::string& logPath);

    // Core system automation interface
    virtual std::vector<float> getSystemState();
    virtual bool executeAutomationAction(const std::string& action, const std::map<std::string, std::string>& parameters);
    virtual std::string getAutomationResult(const std::string& taskId);

    // Process automation
    std::vector<ProcessInfo> getRunningProcesses();
    bool startApplication(const std::string& appName, const std::vector<std::string>& args = {});
    bool closeApplication(const std::string& appName);
    bool waitForProcess(const std::string& processName, int timeoutSeconds = 30);
    bool monitorProcess(int processId);

    // Window automation
    std::vector<WindowInfo> getActiveWindows();
    bool switchToWindow(const std::string& windowTitle);
    bool arrangeWindows(const std::string& arrangement);
    bool simulateUserInteraction(const std::string& interactionType, const std::map<std::string, std::string>& parameters);

    // File system automation
    std::vector<FileSystemInfo> scanDirectory(const std::string& path);
    bool organizeFiles(const std::string& sourcePath, const std::string& destinationPath, const std::string& criteria);
    bool backupFiles(const std::string& sourcePath, const std::string& backupPath);
    bool synchronizeDirectories(const std::string& sourcePath, const std::string& destPath);
    bool cleanTemporaryFiles(const std::string& path);

    // Input automation
    bool typeText(const std::string& text, int delayBetweenKeys = 50);
    bool pressKey(int keyCode);
    bool pressKeyCombination(const std::vector<int>& keyCodes);
    bool clickAt(int x, int y, const std::string& button = "left");
    bool doubleClickAt(int x, int y);
    bool rightClickAt(int x, int y);
    bool dragFromTo(int startX, int startY, int endX, int endY);
    bool scrollWheel(int x, int y, int delta);

    // Screen automation
    bool captureScreenToFile(const std::string& filePath);
    bool captureWindowToFile(const std::string& windowTitle, const std::string& filePath);
    bool waitForScreenChange(int x, int y, int width, int height, int timeoutSeconds = 10);
    bool waitForImageOnScreen(const std::string& imagePath, int timeoutSeconds = 10);

    // System monitoring
    struct SystemMetrics {
        float cpuUsage;
        float memoryUsage;
        float diskUsage;
        int activeProcesses;
        int activeWindows;
        float networkUsage;
        std::vector<float> temperatureReadings;
        std::vector<float> fanSpeeds;
    };

    virtual SystemMetrics getSystemMetrics();
    virtual void startSystemMonitoring(int intervalSeconds = 5);
    virtual void stopSystemMonitoring();
    virtual std::vector<SystemMetrics> getMonitoringHistory();

    // Logging functions
    void logSystemMetrics(const SystemMetrics& metrics);

    // Task scheduling
    std::string scheduleTask(const std::string& taskName, const std::string& schedule, const std::string& command);
    bool cancelScheduledTask(const std::string& taskId);
    std::vector<AutomationTask> getScheduledTasks();
    std::vector<AutomationTask> getTaskHistory();

    // Power management
    bool shutdownSystem(int delaySeconds = 0);
    bool restartSystem(int delaySeconds = 0);
    bool hibernateSystem();
    bool sleepSystem();
    bool lockWorkstation();

    // Network automation
    bool downloadFile(const std::string& url, const std::string& localPath);
    bool uploadFile(const std::string& localPath, const std::string& url);
    bool checkInternetConnection();
    std::string getNetworkInfo();

    // Clipboard automation
    std::string getClipboardText();
    bool setClipboardText(const std::string& text);
    bool clearClipboard();

    // Registry automation (Windows)
    std::string readRegistry(const std::string& keyPath, const std::string& valueName);
    bool writeRegistry(const std::string& keyPath, const std::string& valueName, const std::string& value);
    bool backupRegistry(const std::string& backupPath);
    bool restoreRegistry(const std::string& backupPath);

    // Service automation
    bool manageService(const std::string& serviceName, const std::string& action);
    std::vector<std::string> getSystemServices();
    std::string getServiceInfo(const std::string& serviceName);

    // Advanced automation
    bool createAutomationScript(const std::string& scriptName, const std::vector<std::string>& commands);
    bool executeAutomationScript(const std::string& scriptName);
    bool createMacro(const std::string& macroName, const std::vector<std::string>& actions);
    bool executeMacro(const std::string& macroName);

    // Safety and validation
    bool validateAction(const std::string& action, const std::map<std::string, std::string>& parameters);
    bool checkPermissions(const std::string& action);
    std::vector<std::string> getSafetyWarnings();

    // Status and information
    bool isReady() const { return isInitialized; }
    std::string getModuleName() const { return moduleName; }
    bool isAutomationRunning() const { return isRunning; }

    // Memory management
    void clearTaskHistory();
    size_t getMemoryUsage() const;

    // Training interface for system-specific learning
    virtual bool trainOnSystemData(const std::string& dataPath, int epochs = 10);
    virtual bool learnAutomationPatterns(const std::vector<std::string>& logFiles);

    // Real-time monitoring
    virtual void startRealTimeMonitoring();
    virtual void stopRealTimeMonitoring();
    virtual bool isMonitoringActive() const;
};

#endif // SYSTEM_MODULE_HPP
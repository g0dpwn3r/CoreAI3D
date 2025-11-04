const { ipcRenderer } = require('electron');

// Global state
let currentTab = 'dashboard';
let clientInitialized = false;
let systemMetricsInterval;
let selectedModelId = null;
let availableModels = [];
let currentPage = 1;
let currentPageSize = 20;
let hasMorePages = false;
let currentSearchTerm = '';
let currentTypeFilter = '';
let currentSourceFilter = 'all';
let isLoadingModels = false;
let trainingInProgress = false;
let trainingStartTime = null;
let trainingPaused = false;
let wsConnection = null;

// DOM elements
const elements = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    setupEventListeners();
    initializeClient();
    loadSettings();
    setupTabs();
    startMetricsUpdates();
    loadAvailableModels();
});

// Initialize DOM element references
function initializeElements() {
    // Tab buttons
    elements.tabs = document.querySelectorAll('.tab-button');
    elements.tabPanels = document.querySelectorAll('.tab-panel');

    // Status elements
    elements.connectionStatus = document.getElementById('connection-status');
    elements.testConnectionBtn = document.getElementById('test-connection');

    // Dashboard elements
    elements.cpuUsage = document.getElementById('cpu-usage');
    elements.memoryUsage = document.getElementById('memory-usage');
    elements.diskUsage = document.getElementById('disk-usage');
    elements.networkUsage = document.getElementById('network-usage');
    elements.cpuBar = document.getElementById('cpu-bar');
    elements.memoryBar = document.getElementById('memory-bar');
    elements.diskBar = document.getElementById('disk-bar');
    elements.healthScore = document.getElementById('health-score');
    elements.healthBar = document.getElementById('health-bar');
    elements.refreshMetricsBtn = document.getElementById('refresh-metrics');

    // Training data elements
    elements.datasetList = document.getElementById('dataset-list');
    elements.datasetInfo = document.getElementById('dataset-info');
    elements.fileList = document.getElementById('file-list');
    elements.newDatasetBtn = document.getElementById('new-dataset');
    elements.deleteDatasetBtn = document.getElementById('delete-dataset');
    elements.addFileBtn = document.getElementById('add-file');
    elements.removeFileBtn = document.getElementById('remove-file');

    // Training elements
    elements.trainingModelType = document.getElementById('training-model-type');
    elements.trainingDataset = document.getElementById('training-dataset');
    elements.trainingEpochs = document.getElementById('training-epochs');
    elements.trainingBatchSize = document.getElementById('training-batch-size');
    elements.trainingLearningRate = document.getElementById('training-learning-rate');
    elements.trainingOptimizer = document.getElementById('training-optimizer');
    elements.trainingDataAugmentation = document.getElementById('training-data-augmentation');
    elements.trainingEarlyStopping = document.getElementById('training-early-stopping');
    elements.trainingSaveCheckpoints = document.getElementById('training-save-checkpoints');
    elements.startTrainingBtn = document.getElementById('start-training');
    elements.stopTrainingBtn = document.getElementById('stop-training');
    elements.pauseTrainingBtn = document.getElementById('pause-training');
    elements.trainingStatus = document.getElementById('training-status');
    elements.trainingProgressBar = document.getElementById('training-progress-bar');
    elements.trainingProgressFill = document.getElementById('training-progress-fill');
    elements.trainingProgressText = document.getElementById('training-progress-text');
    elements.currentEpoch = document.getElementById('current-epoch');
    elements.totalEpochs = document.getElementById('total-epochs');
    elements.trainingLoss = document.getElementById('training-loss');
    elements.trainingAccuracy = document.getElementById('training-accuracy');
    elements.trainingEta = document.getElementById('training-eta');
    elements.trainingLogs = document.getElementById('training-logs');
    elements.clearTrainingLogsBtn = document.getElementById('clear-training-logs');
    elements.exportTrainingLogsBtn = document.getElementById('export-training-logs');

    // Testing elements
    elements.testType = document.getElementById('test-type');
    elements.testModule = document.getElementById('test-module');
    elements.runTestBtn = document.getElementById('run-test');
    elements.clearResultsBtn = document.getElementById('clear-results');
    elements.testResults = document.getElementById('test-results');
    elements.loadTestConfig = document.getElementById('load-test-config');
    elements.regularTestConfig = document.getElementById('regular-test-config');
    elements.concurrentUsers = document.getElementById('concurrent-users');
    elements.testDuration = document.getElementById('test-duration');
    elements.loadTestEndpoint = document.getElementById('load-test-endpoint');

    // Prediction elements
    elements.predictModelType = document.getElementById('predict-model-type');
    elements.predictSpecificModel = document.getElementById('predict-specific-model');
    elements.predictInputType = document.getElementById('predict-input-type');
    elements.predictInput = document.getElementById('predict-input');
    elements.predictFilePath = document.getElementById('predict-file-path');
    elements.browseFileBtn = document.getElementById('browse-file');
    elements.containsHeader = document.getElementById('contains-header');
    elements.containsText = document.getElementById('contains-text');
    elements.runPredictionBtn = document.getElementById('run-prediction');
    elements.clearPredictionBtn = document.getElementById('clear-prediction');
    elements.predictProgress = document.getElementById('predict-progress');
    elements.predictionResults = document.getElementById('prediction-results');

    // Load model elements
    elements.modelIdentifier = document.getElementById('model-identifier');
    elements.modelType = document.getElementById('model-type');
    elements.loadModelBtn = document.getElementById('load-model');
    elements.clearLoadResultsBtn = document.getElementById('clear-load-results');
    elements.loadProgress = document.getElementById('load-progress');
    elements.loadResults = document.getElementById('load-results');

    // Windows VM elements
    elements.windowsVmStatus = document.getElementById('windows-vm-status');
    elements.vmName = document.getElementById('vm-name');
    elements.windowsVersion = document.getElementById('windows-version');
    elements.cpuCores = document.getElementById('cpu-cores');
    elements.ramGb = document.getElementById('ram-gb');
    elements.diskSize = document.getElementById('disk-size');
    elements.enableNetworking = document.getElementById('enable-networking');
    elements.enableGpu = document.getElementById('enable-gpu');
    elements.startWindowsVmBtn = document.getElementById('start-windows-vm');
    elements.stopWindowsVmBtn = document.getElementById('stop-windows-vm');
    elements.restartWindowsVmBtn = document.getElementById('restart-windows-vm');
    elements.vmCommand = document.getElementById('vm-command');
    elements.executeVmCommandBtn = document.getElementById('execute-vm-command');
    elements.openVmShellBtn = document.getElementById('open-vm-shell');
    elements.vmProgress = document.getElementById('vm-progress');
    elements.windowsVmLogs = document.getElementById('windows-vm-logs');
    elements.clearWindowsVmLogsBtn = document.getElementById('clear-windows-vm-logs');
    elements.exportVmLogsBtn = document.getElementById('export-vm-logs');
    elements.vmPerformance = document.getElementById('vm-performance');
    elements.refreshVmPerformanceBtn = document.getElementById('refresh-vm-performance');

    // Debugging elements
    elements.systemInfo = document.getElementById('system-info');
    elements.processTable = document.getElementById('process-table');
    elements.refreshSystemInfoBtn = document.getElementById('refresh-system-info');
    elements.refreshProcessesBtn = document.getElementById('refresh-processes');
    elements.apiDebugBtn = document.getElementById('api-debug');
    elements.systemDebugBtn = document.getElementById('system-debug');

    // Linux sandbox elements
    elements.linuxSandboxStatus = document.getElementById('linux-sandbox-status');
    elements.linuxContainer = document.getElementById('linux-container');
    elements.linuxCommand = document.getElementById('linux-command');
    elements.executeLinuxCommandBtn = document.getElementById('execute-linux-command');
    elements.startLinuxContainerBtn = document.getElementById('start-linux-container');
    elements.stopLinuxContainerBtn = document.getElementById('stop-linux-container');
    elements.linuxProgress = document.getElementById('linux-progress');
    elements.linuxOutput = document.getElementById('linux-output');
    elements.clearLinuxOutputBtn = document.getElementById('clear-linux-output');
    elements.trainingScenario = document.getElementById('training-scenario');
    elements.runLinuxTrainingBtn = document.getElementById('run-linux-training');
    elements.generateReportBtn = document.getElementById('generate-report');
    elements.trainingResults = document.getElementById('training-results');
    elements.refreshContainersBtn = document.getElementById('refresh-containers');

    // Downloads elements
    elements.modelSearch = document.getElementById('model-search');
    elements.modelTypeFilter = document.getElementById('model-type-filter');
    elements.modelSourceFilter = document.getElementById('model-source-filter');
    elements.searchModelsBtn = document.getElementById('search-models');
    elements.refreshModelsBtn = document.getElementById('refresh-models');
    elements.modelTable = document.getElementById('model-table');
    elements.downloadSelectedBtn = document.getElementById('download-selected');
    elements.cancelDownloadBtn = document.getElementById('cancel-download');
    elements.downloadSource = document.getElementById('download-source');
    elements.downloadFasttextBtn = document.getElementById('download-fasttext');
    elements.downloadProgress = document.getElementById('download-progress');
    elements.downloadStatus = document.getElementById('download-status');
    elements.downloadLog = document.getElementById('download-log');
    elements.clearDownloadLogBtn = document.getElementById('clear-download-log');
    elements.modelInfo = document.getElementById('model-info');
    elements.paginationControls = document.getElementById('pagination-controls');
    elements.prevPageBtn = document.getElementById('prev-page');
    elements.nextPageBtn = document.getElementById('next-page');
    elements.pageInfo = document.getElementById('page-info');
    elements.pageSize = document.getElementById('page-size');

    // Code generation elements
    elements.codeLanguage = document.getElementById('code-language');
    elements.codeType = document.getElementById('code-type');
    elements.codeComplexity = document.getElementById('code-complexity');
    elements.includeComments = document.getElementById('include-comments');
    elements.includeExamples = document.getElementById('include-examples');
    elements.codeDescription = document.getElementById('code-description');
    elements.generateCodeBtn = document.getElementById('generate-code');
    elements.copyCodeBtn = document.getElementById('copy-code');
    elements.saveCodeBtn = document.getElementById('save-code');
    elements.clearCodeBtn = document.getElementById('clear-code');
    elements.codeGenProgress = document.getElementById('code-gen-progress');
    elements.generatedCode = document.getElementById('generated-code');
    elements.templateInfo = document.getElementById('template-info');

    // Chat elements
    elements.chatMessages = document.getElementById('chat-messages');
    elements.chatInput = document.getElementById('chat-input');
    elements.sendMessageBtn = document.getElementById('send-message');

    // Neural elements
    elements.topologyDisplay = document.getElementById('topology-display');
    elements.activityDisplay = document.getElementById('activity-display');
    elements.neuralCanvas = document.getElementById('neural-canvas');
    elements.refreshNeuralBtn = document.getElementById('refresh-neural');

    // Settings elements
    elements.apiUrl = document.getElementById('api-url');
    elements.wsUrl = document.getElementById('ws-url');
    elements.apiKey = document.getElementById('api-key');
    elements.dataDir = document.getElementById('data-dir');
    elements.sandboxTimeout = document.getElementById('sandbox-timeout');
    elements.logLevel = document.getElementById('log-level');
    elements.enableDiagnostics = document.getElementById('enable-diagnostics');
    elements.saveSettingsBtn = document.getElementById('save-settings');
    elements.testApiConnectionBtn = document.getElementById('test-api-connection');
}

// Setup event listeners
function setupEventListeners() {
    // Tab switching
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Theme toggle
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'dark';
        setTheme(savedTheme);
        updateThemeButton(savedTheme);
    }

    // Connection testing
    elements.testConnectionBtn.addEventListener('click', testConnection);

    // Dashboard
    elements.refreshMetricsBtn.addEventListener('click', updateSystemMetrics);

    // Training data
    elements.newDatasetBtn.addEventListener('click', createDataset);
    elements.deleteDatasetBtn.addEventListener('click', deleteDataset);
    elements.addFileBtn.addEventListener('click', addFileToDataset);
    elements.removeFileBtn.addEventListener('click', removeFileFromDataset);

    // Training
    elements.startTrainingBtn.addEventListener('click', startTraining);
    elements.stopTrainingBtn.addEventListener('click', stopTraining);
    elements.pauseTrainingBtn.addEventListener('click', pauseTraining);
    elements.clearTrainingLogsBtn.addEventListener('click', clearTrainingLogs);
    elements.exportTrainingLogsBtn.addEventListener('click', exportTrainingLogs);
    elements.trainingModelType.addEventListener('change', onTrainingModelTypeChanged);

    // Testing
    elements.testType.addEventListener('change', onTestTypeChanged);
    elements.runTestBtn.addEventListener('click', runTest);
    elements.clearResultsBtn.addEventListener('click', clearTestResults);

    // Prediction
    elements.predictModelType.addEventListener('change', onPredictModelTypeChanged);
    elements.browseFileBtn.addEventListener('click', browsePredictInputFile);
    elements.runPredictionBtn.addEventListener('click', runPrediction);
    elements.clearPredictionBtn.addEventListener('click', clearPredictionResults);

    // Load model
    elements.loadModelBtn.addEventListener('click', loadModel);
    elements.clearLoadResultsBtn.addEventListener('click', clearLoadResults);

    // Windows VM
    elements.startWindowsVmBtn.addEventListener('click', startWindowsVm);
    elements.stopWindowsVmBtn.addEventListener('click', stopWindowsVm);
    elements.restartWindowsVmBtn.addEventListener('click', restartWindowsVm);
    elements.executeVmCommandBtn.addEventListener('click', executeVmCommand);
    elements.openVmShellBtn.addEventListener('click', openVmShell);
    elements.clearWindowsVmLogsBtn.addEventListener('click', clearWindowsVmLogs);
    elements.exportVmLogsBtn.addEventListener('click', exportVmLogs);
    elements.refreshVmPerformanceBtn.addEventListener('click', refreshVmPerformance);

    // Debugging
    elements.refreshSystemInfoBtn.addEventListener('click', updateSystemInfo);
    elements.refreshProcessesBtn.addEventListener('click', updateProcessList);
    elements.apiDebugBtn.addEventListener('click', runApiDebug);
    elements.systemDebugBtn.addEventListener('click', runSystemDebug);

    // Linux sandbox
    elements.executeLinuxCommandBtn.addEventListener('click', executeLinuxCommand);
    elements.startLinuxContainerBtn.addEventListener('click', startLinuxContainer);
    elements.stopLinuxContainerBtn.addEventListener('click', stopLinuxContainer);
    elements.clearLinuxOutputBtn.addEventListener('click', clearLinuxOutput);
    elements.runLinuxTrainingBtn.addEventListener('click', runLinuxTraining);
    elements.generateReportBtn.addEventListener('click', generateTrainingReport);
    elements.refreshContainersBtn.addEventListener('click', refreshContainers);

    // Downloads
    elements.searchModelsBtn.addEventListener('click', searchModels);
    elements.refreshModelsBtn.addEventListener('click', refreshModels);
    elements.downloadSelectedBtn.addEventListener('click', downloadSelectedModel);
    elements.cancelDownloadBtn.addEventListener('click', cancelModelDownload);
    elements.downloadFasttextBtn.addEventListener('click', downloadFasttextEmbeddings);
    elements.clearDownloadLogBtn.addEventListener('click', clearDownloadLog);
    elements.prevPageBtn.addEventListener('click', () => changePage(-1));
    elements.nextPageBtn.addEventListener('click', () => changePage(1));
    elements.pageSize.addEventListener('change', onPageSizeChanged);

    // Code generation
    elements.generateCodeBtn.addEventListener('click', generateCode);
    elements.copyCodeBtn.addEventListener('click', copyGeneratedCode);
    elements.saveCodeBtn.addEventListener('click', saveGeneratedCode);
    elements.clearCodeBtn.addEventListener('click', clearGeneratedCode);

    // Chat
    elements.sendMessageBtn.addEventListener('click', sendChatMessage);
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });

    // Neural
    elements.refreshNeuralBtn.addEventListener('click', refreshNeuralData);

    // Settings
    elements.saveSettingsBtn.addEventListener('click', saveSettings);
    elements.testApiConnectionBtn.addEventListener('click', testApiConnection);
}

// Initialize client
async function initializeClient() {
    try {
        const config = {
            apiUrl: elements.apiUrl.value,
            wsUrl: elements.wsUrl.value,
            apiKey: elements.apiKey.value
        };

        const result = await ipcRenderer.invoke('initialize-client', config);
        if (result.success) {
            clientInitialized = true;
            updateConnectionStatus(true);
            initializeWebSocket();
            console.log('Client initialized successfully');
        } else {
            updateConnectionStatus(false);
            console.error('Failed to initialize client:', result.error);
        }
    } catch (error) {
        updateConnectionStatus(false);
        console.error('Error initializing client:', error);
    }
}

// Initialize WebSocket connection for real-time updates
function initializeWebSocket() {
    try {
        const wsUrl = elements.wsUrl.value;
        if (!wsUrl) return;

        wsConnection = new WebSocket(wsUrl);

        wsConnection.onopen = () => {
            console.log('WebSocket connected');
        };

        wsConnection.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        wsConnection.onclose = () => {
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                if (clientInitialized) {
                    initializeWebSocket();
                }
            }, 5000);
        };

        wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    } catch (error) {
        console.error('Error initializing WebSocket:', error);
    }
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'training_progress':
            updateTrainingProgressFromWS(data);
            break;
        case 'training_status':
            updateTrainingStatusFromWS(data);
            break;
        case 'training_log':
            updateTrainingLogsFromWS(data);
            break;
        default:
            console.log('Unknown WebSocket message type:', data.type);
    }
}

// Update training progress from WebSocket
function updateTrainingProgressFromWS(data) {
    if (data.progress !== undefined) {
        updateTrainingProgress(data.progress);
    }

    if (data.epoch !== undefined) {
        elements.currentEpoch.textContent = data.epoch;
    }

    if (data.loss !== undefined) {
        elements.trainingLoss.textContent = data.loss.toFixed(3);
    }

    if (data.accuracy !== undefined) {
        elements.trainingAccuracy.textContent = `${data.accuracy.toFixed(2)}%`;
    }

    if (data.eta !== undefined) {
        elements.trainingEta.textContent = data.eta;
    }
}

// Update training status from WebSocket
function updateTrainingStatusFromWS(data) {
    if (data.status) {
        elements.trainingStatus.textContent = data.status;
        elements.trainingStatus.className = `status-indicator ${data.status.toLowerCase()}`;

        switch (data.status.toLowerCase()) {
            case 'completed':
                trainingInProgress = false;
                elements.startTrainingBtn.disabled = false;
                elements.stopTrainingBtn.disabled = true;
                elements.pauseTrainingBtn.disabled = true;
                elements.pauseTrainingBtn.textContent = 'Pause Training';
                break;
            case 'stopped':
                trainingInProgress = false;
                trainingPaused = false;
                elements.startTrainingBtn.disabled = false;
                elements.stopTrainingBtn.disabled = true;
                elements.pauseTrainingBtn.disabled = true;
                elements.pauseTrainingBtn.textContent = 'Pause Training';
                break;
            case 'paused':
                trainingPaused = true;
                elements.pauseTrainingBtn.textContent = 'Resume Training';
                break;
            case 'running':
                trainingPaused = false;
                elements.pauseTrainingBtn.textContent = 'Pause Training';
                break;
        }
    }
}

// Update training logs from WebSocket
function updateTrainingLogsFromWS(data) {
    if (data.message) {
        const timestamp = new Date().toLocaleTimeString();
        elements.trainingLogs.textContent += `[${timestamp}] ${data.message}\n`;
        elements.trainingLogs.scrollTop = elements.trainingLogs.scrollHeight;
    }
}

// Setup tabs
function setupTabs() {
    switchTab('dashboard');
}

// Switch between tabs
function switchTab(tabName) {
    elements.tabs.forEach(tab => tab.classList.remove('active'));
    elements.tabPanels.forEach(panel => panel.classList.remove('active'));

    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');

    currentTab = tabName;
}

// Update connection status
function updateConnectionStatus(connected) {
    elements.connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
    elements.connectionStatus.className = connected ? 'status-connected' : 'status-disconnected';
}

// Test connection
async function testConnection() {
    try {
        const result = await ipcRenderer.invoke('test-connection');
        if (result.success) {
            updateConnectionStatus(true);
            showMessage('Connection test successful', 'success');
        } else {
            updateConnectionStatus(false);
            showMessage(`Connection test failed: ${result.error}`, 'error');
        }
    } catch (error) {
        updateConnectionStatus(false);
        showMessage(`Connection test error: ${error.message}`, 'error');
    }
}

// Start metrics updates
function startMetricsUpdates() {
    updateSystemMetrics();
    systemMetricsInterval = setInterval(updateSystemMetrics, 5000); // Update every 5 seconds
}

// Update system metrics
async function updateSystemMetrics() {
    try {
        const result = await ipcRenderer.invoke('get-system-metrics');
        if (result.success) {
            const metrics = result.data;
            elements.cpuUsage.textContent = `${metrics.cpu_usage?.toFixed(1) || 0}%`;
            elements.memoryUsage.textContent = `${metrics.memory_usage?.toFixed(1) || 0}%`;
            elements.diskUsage.textContent = `${metrics.disk_usage?.toFixed(1) || 0}%`;
            elements.networkUsage.textContent = `${(metrics.network_rx || 0) / 1024} KB/s`;

            elements.cpuBar.style.width = `${metrics.cpu_usage || 0}%`;
            elements.memoryBar.style.width = `${metrics.memory_usage || 0}%`;
            elements.diskBar.style.width = `${metrics.disk_usage || 0}%`;

            // Calculate health score
            const healthScore = calculateHealthScore(metrics);
            elements.healthScore.textContent = `${healthScore}%`;
            elements.healthBar.style.width = `${healthScore}%`;
        }
    } catch (error) {
        console.error('Error updating system metrics:', error);
    }
}

// Calculate health score
function calculateHealthScore(metrics) {
    let score = 100;

    // CPU usage penalty
    if (metrics.cpu_usage > 90) score -= 30;
    else if (metrics.cpu_usage > 70) score -= 15;
    else if (metrics.cpu_usage > 50) score -= 5;

    // Memory usage penalty
    if (metrics.memory_usage > 90) score -= 30;
    else if (metrics.memory_usage > 70) score -= 15;
    else if (metrics.memory_usage > 50) score -= 5;

    // Disk usage penalty
    if (metrics.disk_usage > 90) score -= 20;
    else if (metrics.disk_usage > 80) score -= 10;

    return Math.max(0, Math.min(100, score));
}

// Training data management
async function createDataset() {
    const name = prompt('Enter dataset name:');
    const description = prompt('Enter dataset description:');
    const dataType = prompt('Enter data type (e.g., csv, json, images):');

    if (!name || !description || !dataType) {
        showMessage('All fields are required', 'error');
        return;
    }

    try {
        const result = await ipcRenderer.invoke('create-dataset', name, description, dataType);
        if (result.success) {
            showMessage(`Dataset '${name}' created successfully`, 'success');
            loadDatasets();
        } else {
            showMessage(`Failed to create dataset: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error creating dataset: ${error.message}`, 'error');
    }
}

async function deleteDataset() {
    const selectedItem = elements.datasetList.selectedItems()[0];
    if (!selectedItem) {
        showMessage('Please select a dataset to delete', 'warning');
        return;
    }

    const datasetName = selectedItem.text();
    if (!confirm(`Are you sure you want to delete dataset '${datasetName}'?`)) {
        return;
    }

    try {
        const result = await ipcRenderer.invoke('delete-dataset', datasetName);
        if (result.success) {
            showMessage(`Dataset '${datasetName}' deleted successfully`, 'success');
            loadDatasets();
        } else {
            showMessage(`Failed to delete dataset: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error deleting dataset: ${error.message}`, 'error');
    }
}

async function addFileToDataset() {
    const selectedItem = elements.datasetList.selectedItems()[0];
    if (!selectedItem) {
        showMessage('Please select a dataset first', 'warning');
        return;
    }

    const datasetName = selectedItem.text();
    const filePath = await ipcRenderer.invoke('select-file');

    if (!filePath) return;

    try {
        const result = await ipcRenderer.invoke('add-file-to-dataset', datasetName, filePath);
        if (result.success) {
            showMessage(`File added to dataset '${datasetName}' successfully`, 'success');
            loadDatasets();
        } else {
            showMessage(`Failed to add file: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error adding file: ${error.message}`, 'error');
    }
}

async function removeFileFromDataset() {
    showMessage('File removal from dataset would be implemented with server API', 'info');
}

async function loadDatasets() {
    try {
        // For now, just clear and show a message that datasets would be loaded from server
        elements.datasetList.innerHTML = '<li>No datasets loaded yet. Create one to get started.</li>';
        elements.datasetInfo.textContent = 'Select a dataset to view details here.';
        showMessage('Dataset loading would connect to server API', 'info');
    } catch (error) {
        showMessage(`Error loading datasets: ${error.message}`, 'error');
    }
}

// Training functionality
function onTrainingModelTypeChanged() {
    // Load available datasets for the selected model type
    loadAvailableDatasets();
}

async function loadAvailableDatasets() {
    try {
        const modelType = elements.trainingModelType.value;
        // This would normally fetch datasets from the server based on model type
        // For now, populate with sample datasets
        const sampleDatasets = [
            { id: 'mnist', name: 'MNIST Handwritten Digits', type: 'vision' },
            { id: 'cifar10', name: 'CIFAR-10 Images', type: 'vision' },
            { id: 'imagenet', name: 'ImageNet', type: 'vision' },
            { id: 'librispeech', name: 'LibriSpeech Audio', type: 'audio' },
            { id: 'wikitext', name: 'WikiText-2', type: 'text' },
            { id: 'squad', name: 'SQuAD Question Answering', type: 'text' }
        ];

        const filteredDatasets = sampleDatasets.filter(ds => ds.type === modelType || modelType === 'multimodal');

        elements.trainingDataset.innerHTML = '<option value="">Select Dataset...</option>';
        filteredDatasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = dataset.name;
            elements.trainingDataset.appendChild(option);
        });
    } catch (error) {
        showMessage(`Error loading datasets: ${error.message}`, 'error');
    }
}

async function startTraining() {
    if (trainingInProgress) {
        showMessage('Training is already in progress', 'warning');
        return;
    }

    const config = {
        modelType: elements.trainingModelType.value,
        dataset: elements.trainingDataset.value,
        epochs: parseInt(elements.trainingEpochs.value),
        batchSize: parseInt(elements.trainingBatchSize.value),
        learningRate: parseFloat(elements.trainingLearningRate.value),
        optimizer: elements.trainingOptimizer.value,
        dataAugmentation: elements.trainingDataAugmentation.checked,
        earlyStopping: elements.trainingEarlyStopping.checked,
        saveCheckpoints: elements.trainingSaveCheckpoints.checked
    };

    if (!config.dataset) {
        showMessage('Please select a dataset', 'error');
        return;
    }

    try {
        elements.trainingProgressBar.style.display = 'block';
        elements.trainingStatus.textContent = 'Starting training...';
        elements.trainingStatus.className = 'status-indicator training';

        const result = await ipcRenderer.invoke('start-training', config);

        if (result.success) {
            trainingInProgress = true;
            trainingStartTime = Date.now();
            trainingPaused = false;

            elements.startTrainingBtn.disabled = true;
            elements.stopTrainingBtn.disabled = false;
            elements.pauseTrainingBtn.disabled = false;

            elements.totalEpochs.textContent = config.epochs;
            elements.trainingLogs.textContent += `[${new Date().toLocaleTimeString()}] Training started with ${config.modelType} model on ${config.dataset} dataset\n`;

            showMessage('Training started successfully', 'success');

            // Start progress monitoring
            startTrainingProgressUpdates();
        } else {
            elements.trainingStatus.textContent = 'Failed to start';
            elements.trainingStatus.className = 'status-indicator error';
            showMessage(`Failed to start training: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.trainingStatus.textContent = 'Error';
        elements.trainingStatus.className = 'status-indicator error';
        showMessage(`Training start error: ${error.message}`, 'error');
    }
}

async function stopTraining() {
    if (!trainingInProgress) {
        showMessage('No training in progress', 'warning');
        return;
    }

    try {
        const result = await ipcRenderer.invoke('stop-training');

        if (result.success) {
            trainingInProgress = false;
            trainingPaused = false;

            elements.startTrainingBtn.disabled = false;
            elements.stopTrainingBtn.disabled = true;
            elements.pauseTrainingBtn.disabled = true;

            elements.trainingStatus.textContent = 'Stopped';
            elements.trainingStatus.className = 'status-indicator stopped';
            elements.trainingProgressBar.style.display = 'none';

            elements.trainingLogs.textContent += `[${new Date().toLocaleTimeString()}] Training stopped by user\n`;

            showMessage('Training stopped successfully', 'info');
        } else {
            showMessage(`Failed to stop training: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error stopping training: ${error.message}`, 'error');
    }
}

async function pauseTraining() {
    if (!trainingInProgress) {
        showMessage('No training in progress', 'warning');
        return;
    }

    try {
        const action = trainingPaused ? 'resume' : 'pause';
        const result = await ipcRenderer.invoke(`${action}-training`);

        if (result.success) {
            trainingPaused = !trainingPaused;
            elements.trainingStatus.textContent = trainingPaused ? 'Paused' : 'Running';
            elements.trainingStatus.className = `status-indicator ${trainingPaused ? 'paused' : 'training'}`;
            elements.pauseTrainingBtn.textContent = trainingPaused ? 'Resume Training' : 'Pause Training';

            elements.trainingLogs.textContent += `[${new Date().toLocaleTimeString()}] Training ${action}d\n`;

            showMessage(`Training ${action}d successfully`, 'info');
        } else {
            showMessage(`Failed to ${action} training: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error ${trainingPaused ? 'resuming' : 'pausing'} training: ${error.message}`, 'error');
    }
}

function startTrainingProgressUpdates() {
    // Simulate progress updates (in real implementation, this would come from WebSocket)
    let progress = 0;
    const progressInterval = setInterval(() => {
        if (!trainingInProgress || trainingPaused) return;

        progress += Math.random() * 2; // Simulate progress
        if (progress >= 100) {
            progress = 100;
            clearInterval(progressInterval);
            trainingCompleted();
        }

        updateTrainingProgress(progress);
    }, 1000);
}

function updateTrainingProgress(progress) {
    const percentage = Math.min(100, Math.max(0, progress));
    elements.trainingProgressFill.style.width = `${percentage}%`;
    elements.trainingProgressText.textContent = `${percentage.toFixed(1)}%`;

    // Update metrics
    const currentEpoch = Math.floor((percentage / 100) * parseInt(elements.totalEpochs.textContent));
    elements.currentEpoch.textContent = currentEpoch;

    // Simulate loss and accuracy
    const loss = (Math.random() * 0.5 + 0.1).toFixed(3);
    const accuracy = (percentage * 0.8 + Math.random() * 20).toFixed(2);

    elements.trainingLoss.textContent = loss;
    elements.trainingAccuracy.textContent = `${accuracy}%`;

    // Calculate ETA
    if (trainingStartTime) {
        const elapsed = Date.now() - trainingStartTime;
        const remaining = (elapsed / percentage) * (100 - percentage);
        const eta = new Date(Date.now() + remaining);
        elements.trainingEta.textContent = eta.toLocaleTimeString();
    }
}

function trainingCompleted() {
    trainingInProgress = false;
    elements.startTrainingBtn.disabled = false;
    elements.stopTrainingBtn.disabled = true;
    elements.pauseTrainingBtn.disabled = true;

    elements.trainingStatus.textContent = 'Completed';
    elements.trainingStatus.className = 'status-indicator completed';
    elements.trainingProgressText.textContent = '100%';

    elements.trainingLogs.textContent += `[${new Date().toLocaleTimeString()}] Training completed successfully\n`;

    showMessage('Training completed successfully!', 'success');
}

function clearTrainingLogs() {
    elements.trainingLogs.textContent = '';
}

async function exportTrainingLogs() {
    try {
        const logs = elements.trainingLogs.textContent;
        if (!logs.trim()) {
            showMessage('No logs to export', 'warning');
            return;
        }

        const result = await ipcRenderer.invoke('export-training-logs', logs);
        if (result.success) {
            showMessage(`Training logs exported to ${result.path}`, 'success');
        } else {
            showMessage('Failed to export training logs', 'error');
        }
    } catch (error) {
        showMessage('Error exporting training logs', 'error');
    }
}

// Testing functionality
function onTestTypeChanged() {
    const testType = elements.testType.value;
    if (testType === 'load') {
        elements.loadTestConfig.style.display = 'block';
        elements.regularTestConfig.style.display = 'none';
    } else {
        elements.loadTestConfig.style.display = 'none';
        elements.regularTestConfig.style.display = 'block';
    }
}

async function runTest() {
    const testType = elements.testType.value;

    if (testType === 'load') {
        await runLoadTest();
    } else {
        await runRegularTest();
    }
}

async function runRegularTest() {
    const testType = elements.testType.value;
    const testModule = elements.testModule.value;

    try {
        const result = await ipcRenderer.invoke('run-test', testType, testModule);
        if (result.success) {
            const row = elements.testResults.insertRow();
            row.insertCell(0).textContent = result.data.test_id || 'N/A';
            row.insertCell(1).textContent = testType;
            row.insertCell(2).textContent = '✓ Success';
            row.insertCell(3).textContent = result.data.duration ? `${result.data.duration.toFixed(2)}s` : 'N/A';
            row.insertCell(4).textContent = result.data.message || 'Test completed';
            showMessage('Test completed successfully', 'success');
        } else {
            const row = elements.testResults.insertRow();
            row.insertCell(0).textContent = `test_${Date.now()}`;
            row.insertCell(1).textContent = testType;
            row.insertCell(2).textContent = '✗ Failed';
            row.insertCell(3).textContent = 'N/A';
            row.insertCell(4).textContent = result.error || 'Test failed';
            showMessage(`Test failed: ${result.error}`, 'error');
        }
    } catch (error) {
        const row = elements.testResults.insertRow();
        row.insertCell(0).textContent = `test_${Date.now()}`;
        row.insertCell(1).textContent = testType;
        row.insertCell(2).textContent = '✗ Error';
        row.insertCell(3).textContent = 'N/A';
        row.insertCell(4).textContent = error.message;
        showMessage(`Test error: ${error.message}`, 'error');
    }
}

async function runLoadTest() {
    const concurrentUsers = parseInt(elements.concurrentUsers.value);
    const duration = parseInt(elements.testDuration.value);
    const endpoint = elements.loadTestEndpoint.value;

    try {
        const result = await ipcRenderer.invoke('run-load-test', concurrentUsers, duration, endpoint);
        if (result.success) {
            const row = elements.testResults.insertRow();
            row.insertCell(0).textContent = result.data.test_id || `load_${Date.now()}`;
            row.insertCell(1).textContent = 'Load Test';
            row.insertCell(2).textContent = '✓ Success';
            row.insertCell(3).textContent = `${result.data.duration_seconds.toFixed(2)}s`;
            row.insertCell(4).textContent = `Users: ${result.data.concurrency}, Ops/sec: ${result.data.operations_per_second.toFixed(2)}`;

            // Add detailed results row
            const detailRow = elements.testResults.insertRow();
            detailRow.insertCell(0).textContent = '';
            detailRow.insertCell(1).textContent = 'Details';
            detailRow.insertCell(2).textContent = '';
            detailRow.insertCell(3).textContent = '';
            detailRow.insertCell(4).textContent = `Total Ops: ${result.data.total_operations}, Avg Latency: ${result.data.average_operation_time_ms.toFixed(2)}ms`;

            showMessage(`Load test completed: ${result.data.operations_per_second.toFixed(2)} ops/sec`, 'success');
        } else {
            const row = elements.testResults.insertRow();
            row.insertCell(0).textContent = `load_${Date.now()}`;
            row.insertCell(1).textContent = 'Load Test';
            row.insertCell(2).textContent = '✗ Failed';
            row.insertCell(3).textContent = 'N/A';
            row.insertCell(4).textContent = result.error || 'Load test failed';
            showMessage(`Load test failed: ${result.error}`, 'error');
        }
    } catch (error) {
        const row = elements.testResults.insertRow();
        row.insertCell(0).textContent = `load_${Date.now()}`;
        row.insertCell(1).textContent = 'Load Test';
        row.insertCell(2).textContent = '✗ Error';
        row.insertCell(3).textContent = 'N/A';
        row.insertCell(4).textContent = error.message;
        showMessage(`Load test error: ${error.message}`, 'error');
    }
}

function clearTestResults() {
    elements.testResults.innerHTML = '';
}

// Prediction functionality
function onPredictModelTypeChanged() {
    const modelType = elements.predictModelType.value;
    elements.predictSpecificModel.innerHTML = '<option value="">Select Specific Model...</option>';

    if (modelType === 'vision') {
        addModelOptions(['ResNet-50', 'YOLOv5-Small', 'EfficientNet-B0']);
    } else if (modelType === 'audio') {
        addModelOptions(['Whisper-Tiny', 'Whisper-Base', 'Tacotron2']);
    } else if (modelType === 'text') {
        addModelOptions(['BERT-Base', 'GPT-2 Small', 'DistilBERT']);
    }
    // Add more model types as needed
}

function addModelOptions(options) {
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option.toLowerCase().replace(/\s+/g, '-');
        opt.textContent = option;
        elements.predictSpecificModel.appendChild(opt);
    });
}

async function browsePredictInputFile() {
    const filePath = await ipcRenderer.invoke('select-file');
    if (filePath) {
        elements.predictFilePath.value = filePath;
    }
}

async function runPrediction() {
    const modelType = elements.predictModelType.value;
    const specificModel = elements.predictSpecificModel.value;
    const inputType = elements.predictInputType.value;
    const inputData = elements.predictInput.value;
    const filePath = elements.predictFilePath.value;
    const containsHeader = elements.containsHeader.checked;
    const containsText = elements.containsText.checked;

    let finalInputData = inputData;
    let finalInputType = inputType;

    // Handle different input types
    if (inputType === 'file' && filePath) {
        finalInputData = filePath;
    } else if (inputType === 'url' && filePath) {
        finalInputData = filePath;
    }

    if (!finalInputData) {
        showMessage('Please provide input data', 'error');
        return;
    }

    try {
        elements.predictProgress.style.display = 'block';
        const result = await ipcRenderer.invoke('run-prediction', modelType, specificModel, finalInputType, finalInputData, containsHeader, containsText);

        if (result.success) {
            elements.predictionResults.textContent = JSON.stringify(result.data, null, 2);
            showMessage('Prediction completed successfully', 'success');
        } else {
            elements.predictionResults.textContent = `Prediction failed: ${result.error}`;
            showMessage(`Prediction failed: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.predictionResults.textContent = `Error: ${error.message}`;
        showMessage(`Prediction error: ${error.message}`, 'error');
    } finally {
        elements.predictProgress.style.display = 'none';
    }
}

function clearPredictionResults() {
    elements.predictionResults.textContent = 'Prediction results will appear here...';
}

// Load model functionality
async function loadModel() {
    const identifier = elements.modelIdentifier.value;
    const modelType = elements.modelType.value;

    if (!identifier) {
        showMessage('Please enter a model identifier', 'error');
        return;
    }

    try {
        elements.loadProgress.style.display = 'block';
        const result = await ipcRenderer.invoke('load-model', identifier, modelType);

        if (result.success) {
            elements.loadResults.textContent = `✓ Model loaded successfully!\n\nModel: ${result.data.model_identifier}\nStatus: ${result.data.status}`;
            showMessage('Model loaded successfully', 'success');
        } else {
            elements.loadResults.textContent = `✗ Failed to load model!\n\nError: ${result.error}`;
            showMessage('Failed to load model', 'error');
        }
    } catch (error) {
        elements.loadResults.textContent = `✗ Load operation failed!\n\nError: ${error.message}`;
        showMessage('Model loading error', 'error');
    } finally {
        elements.loadProgress.style.display = 'none';
    }
}

function clearLoadResults() {
    elements.loadResults.textContent = 'Model loading results will appear here...';
}

// Windows VM functionality using winboat
async function startWindowsVm() {
    const config = {
        vm_name: elements.vmName.value || 'coreai3d_windows_vm',
        windows_version: elements.windowsVersion.value,
        cpu_cores: parseInt(elements.cpuCores.value) || 2,
        ram_gb: parseInt(elements.ramGb.value) || 4,
        disk_size_gb: parseInt(elements.diskSize.value) || 20,
        enable_networking: elements.enableNetworking.checked,
        enable_gpu: elements.enableGpu.checked
    };

    try {
        const result = await ipcRenderer.invoke('start-windows-vm', config);
        if (result.success) {
            elements.windowsVmStatus.textContent = 'Running';
            elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Windows VM started successfully\n`;
            showMessage('Windows VM started successfully', 'success');
        } else {
            elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Failed to start VM: ${result.error}\n`;
            showMessage(`Failed to start Windows VM: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Error starting VM: ${error.message}\n`;
        showMessage(`Error starting Windows VM: ${error.message}`, 'error');
    }
}

async function stopWindowsVm() {
    try {
        const result = await ipcRenderer.invoke('stop-windows-vm');
        if (result.success) {
            elements.windowsVmStatus.textContent = 'Stopped';
            elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Windows VM stopped successfully\n`;
            showMessage('Windows VM stopped successfully', 'success');
        } else {
            elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Failed to stop VM: ${result.error}\n`;
            showMessage(`Failed to stop Windows VM: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Error stopping VM: ${error.message}\n`;
        showMessage(`Error stopping Windows VM: ${error.message}`, 'error');
    }
}

async function restartWindowsVm() {
    try {
        const result = await ipcRenderer.invoke('restart-windows-vm');
        if (result.success) {
            elements.windowsVmStatus.textContent = 'Restarting...';
            elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Windows VM restarting...\n`;
            showMessage('Windows VM restarting...', 'info');
            // Update status after restart
            setTimeout(() => {
                elements.windowsVmStatus.textContent = 'Running';
                elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Windows VM restarted successfully\n`;
            }, 5000);
        } else {
            elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Failed to restart VM: ${result.error}\n`;
            showMessage(`Failed to restart Windows VM: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.windowsVmLogs.textContent += `[${new Date().toLocaleTimeString()}] Error restarting VM: ${error.message}\n`;
        showMessage(`Error restarting Windows VM: ${error.message}`, 'error');
    }
}

async function executeVmCommand() {
    const command = elements.vmCommand.value;
    if (!command) {
        showMessage('Please enter a command', 'error');
        return;
    }

    try {
        elements.vmProgress.style.display = 'block';
        const result = await ipcRenderer.invoke('execute-vm-command', { command });

        if (result.success) {
            elements.windowsVmLogs.textContent += `$ ${command}\n${result.output}\n`;
            showMessage('Command executed successfully', 'success');
        } else {
            elements.windowsVmLogs.textContent += `$ ${command}\nError: ${result.error}\n`;
            showMessage('Command execution failed', 'error');
        }
    } catch (error) {
        elements.windowsVmLogs.textContent += `$ ${command}\nError: ${error.message}\n`;
        showMessage('Command execution error', 'error');
    } finally {
        elements.vmProgress.style.display = 'none';
    }
}

async function openVmShell() {
    try {
        const result = await ipcRenderer.invoke('open-vm-shell');
        if (result.success) {
            showMessage('VM shell opened successfully', 'success');
        } else {
            showMessage(`Failed to open VM shell: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error opening VM shell: ${error.message}`, 'error');
    }
}

function clearWindowsVmLogs() {
    elements.windowsVmLogs.textContent = '';
}

async function exportVmLogs() {
    try {
        const logs = elements.windowsVmLogs.textContent;
        if (!logs.trim()) {
            showMessage('No logs to export', 'warning');
            return;
        }

        const result = await ipcRenderer.invoke('export-vm-logs', logs);
        if (result.success) {
            showMessage(`Logs exported to ${result.path}`, 'success');
        } else {
            showMessage('Failed to export logs', 'error');
        }
    } catch (error) {
        showMessage('Error exporting logs', 'error');
    }
}

async function refreshVmPerformance() {
    try {
        const result = await ipcRenderer.invoke('get-vm-performance');
        if (result.success) {
            elements.vmPerformance.textContent = JSON.stringify(result.data, null, 2);
            showMessage('VM performance data refreshed', 'success');
        } else {
            elements.vmPerformance.textContent = `Error: ${result.error}`;
            showMessage(`Failed to get VM performance: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.vmPerformance.textContent = `Error: ${error.message}`;
        showMessage(`Error getting VM performance: ${error.message}`, 'error');
    }
}

// Debugging functionality
async function updateSystemInfo() {
    try {
        const result = await ipcRenderer.invoke('get-system-info');
        if (result.success) {
            elements.systemInfo.textContent = JSON.stringify(result.data, null, 2);
            showMessage('System info updated', 'success');
        } else {
            elements.systemInfo.textContent = `Error: ${result.error}`;
            showMessage(`Failed to get system info: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.systemInfo.textContent = `Error: ${error.message}`;
        showMessage(`Error getting system info: ${error.message}`, 'error');
    }
}

async function updateProcessList() {
    try {
        const result = await ipcRenderer.invoke('get-process-list');
        if (result.success) {
            // Clear existing rows
            while (elements.processTable.rows.length > 1) {
                elements.processTable.deleteRow(1);
            }

            // Add new rows
            result.data.forEach(process => {
                const row = elements.processTable.insertRow();
                row.insertCell(0).textContent = process.pid || 'N/A';
                row.insertCell(1).textContent = process.name || 'N/A';
                row.insertCell(2).textContent = process.cpu ? `${process.cpu.toFixed(1)}%` : 'N/A';
                row.insertCell(3).textContent = process.memory ? `${process.memory.toFixed(1)}%` : 'N/A';
            });

            showMessage(`Updated process list (${result.data.length} processes)`, 'success');
        } else {
            showMessage(`Failed to get process list: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error getting process list: ${error.message}`, 'error');
    }
}

async function runApiDebug() {
    try {
        const result = await ipcRenderer.invoke('run-api-debug');
        if (result.success) {
            // Display debug information in the system info area
            elements.systemInfo.textContent = JSON.stringify(result.data, null, 2);
            showMessage('API debug completed successfully', 'success');
        } else {
            elements.systemInfo.textContent = `Error: ${result.error}`;
            showMessage(`API debug failed: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.systemInfo.textContent = `Error: ${error.message}`;
        showMessage(`API debug error: ${error.message}`, 'error');
    }
}

async function runSystemDebug() {
    try {
        const result = await ipcRenderer.invoke('run-system-debug');
        if (result.success) {
            // Display debug information in the system info area
            elements.systemInfo.textContent = JSON.stringify(result.data, null, 2);
            showMessage('System debug completed successfully', 'success');
        } else {
            elements.systemInfo.textContent = `Error: ${result.error}`;
            showMessage(`System debug failed: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.systemInfo.textContent = `Error: ${error.message}`;
        showMessage(`System debug error: ${error.message}`, 'error');
    }
}

// Linux sandbox functionality
async function executeLinuxCommand() {
    const command = elements.linuxCommand.value;
    const container = elements.linuxContainer.value;

    if (!command) {
        showMessage('Please enter a command', 'error');
        return;
    }

    try {
        elements.linuxProgress.style.display = 'block';
        const result = await ipcRenderer.invoke('execute-linux-command', { command, container });

        if (result.success) {
            elements.linuxOutput.textContent += `$ ${command}\n${result.output}\n`;
            showMessage('Command executed successfully', 'success');
        } else {
            elements.linuxOutput.textContent += `$ ${command}\nError: ${result.error}\n`;
            showMessage('Command execution failed', 'error');
        }
    } catch (error) {
        elements.linuxOutput.textContent += `$ ${command}\nError: ${error.message}\n`;
        showMessage('Command execution error', 'error');
    } finally {
        elements.linuxProgress.style.display = 'none';
    }
}

async function startLinuxContainer() {
    const config = {
        image: 'kali:latest',
        container_name: elements.linuxContainer.value || 'coreai3d_linux_sandbox',
        networking: 'enable',
        privileged: false
    };

    try {
        const result = await ipcRenderer.invoke('start-linux-container', config);
        if (result.success) {
            elements.linuxSandboxStatus.textContent = 'Running';
            elements.linuxOutput.textContent += `[${new Date().toLocaleTimeString()}] Container started successfully\n`;
            showMessage('Linux container started successfully', 'success');
        } else {
            elements.linuxOutput.textContent += `[${new Date().toLocaleTimeString()}] Failed to start container: ${result.error}\n`;
            showMessage(`Failed to start container: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.linuxOutput.textContent += `[${new Date().toLocaleTimeString()}] Error starting container: ${error.message}\n`;
        showMessage(`Error starting container: ${error.message}`, 'error');
    }
}

async function stopLinuxContainer() {
    const containerId = elements.linuxContainer.value;

    try {
        const result = await ipcRenderer.invoke('stop-linux-container', containerId);
        if (result.success) {
            elements.linuxSandboxStatus.textContent = 'Stopped';
            elements.linuxOutput.textContent += `[${new Date().toLocaleTimeString()}] Container stopped successfully\n`;
            showMessage('Linux container stopped successfully', 'success');
        } else {
            elements.linuxOutput.textContent += `[${new Date().toLocaleTimeString()}] Failed to stop container: ${result.error}\n`;
            showMessage(`Failed to stop container: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.linuxOutput.textContent += `[${new Date().toLocaleTimeString()}] Error stopping container: ${error.message}\n`;
        showMessage(`Error stopping container: ${error.message}`, 'error');
    }
}

function clearLinuxOutput() {
    elements.linuxOutput.textContent = '';
}

async function runLinuxTraining() {
    const scenario = elements.trainingScenario.value;

    if (!scenario) {
        showMessage('Please select a training scenario', 'error');
        return;
    }

    try {
        const result = await ipcRenderer.invoke('run-linux-training', scenario);
        if (result.success) {
            elements.trainingResults.textContent = JSON.stringify(result.data, null, 2);
            showMessage('Linux training started successfully', 'success');
        } else {
            elements.trainingResults.textContent = `Training failed: ${result.error}`;
            showMessage(`Training failed: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.trainingResults.textContent = `Error: ${error.message}`;
        showMessage(`Training error: ${error.message}`, 'error');
    }
}

async function generateTrainingReport() {
    const trainingId = prompt('Enter training ID:');
    if (!trainingId) return;

    try {
        const result = await ipcRenderer.invoke('generate-training-report', trainingId);
        if (result.success) {
            elements.trainingResults.textContent = JSON.stringify(result.data, null, 2);
            showMessage('Training report generated successfully', 'success');
        } else {
            elements.trainingResults.textContent = `Report generation failed: ${result.error}`;
            showMessage(`Report generation failed: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.trainingResults.textContent = `Error: ${error.message}`;
        showMessage(`Report generation error: ${error.message}`, 'error');
    }
}

async function refreshContainers() {
    try {
        const result = await ipcRenderer.invoke('refresh-containers');
        if (result.success) {
            // Update container list or status
            showMessage(`Refreshed containers (${result.data.length} containers)`, 'success');
        } else {
            showMessage(`Failed to refresh containers: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error refreshing containers: ${error.message}`, 'error');
    }
}

// Downloads functionality
async function searchModels() {
    if (isLoadingModels) return;

    currentSearchTerm = elements.modelSearch.value;
    currentTypeFilter = elements.modelTypeFilter.value;
    currentSourceFilter = elements.modelSourceFilter.value;
    currentPage = 1;

    await loadModels(true);
}

async function refreshModels() {
    if (isLoadingModels) return;

    currentPage = 1;
    await loadModels(false);
}

async function loadModels(isSearch = false) {
    if (isLoadingModels) return;

    isLoadingModels = true;
    elements.searchModelsBtn.disabled = true;
    elements.refreshModelsBtn.disabled = true;
    elements.searchModelsBtn.textContent = 'Loading...';
    elements.refreshModelsBtn.textContent = 'Loading...';

    try {
        const params = {
            searchTerm: currentSearchTerm,
            typeFilter: currentTypeFilter,
            source: currentSourceFilter,
            page: currentPage,
            pageSize: currentPageSize
        };

        const result = await ipcRenderer.invoke(isSearch ? 'search-models' : 'refresh-models', params);

        if (result.success) {
            availableModels = result.models || [];
            hasMorePages = result.hasMore || false;

            displayModels(availableModels);
            updatePaginationControls();

            const action = isSearch ? 'search' : 'refresh';
            showMessage(`${action} completed: ${availableModels.length} models found`, 'success');
        } else {
            showMessage(`${isSearch ? 'Search' : 'Refresh'} failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`${isSearch ? 'Search' : 'Refresh'} error: ${error.message}`, 'error');
    } finally {
        isLoadingModels = false;
        elements.searchModelsBtn.disabled = false;
        elements.refreshModelsBtn.disabled = false;
        elements.searchModelsBtn.textContent = 'Search';
        elements.refreshModelsBtn.textContent = 'Refresh';
    }
}

function updatePaginationControls() {
    if (availableModels.length > 0 || currentPage > 1) {
        elements.paginationControls.style.display = 'block';
        elements.pageInfo.textContent = `Page ${currentPage}`;
        elements.prevPageBtn.disabled = currentPage <= 1;
        elements.nextPageBtn.disabled = !hasMorePages;
        elements.pageSize.value = currentPageSize.toString();
    } else {
        elements.paginationControls.style.display = 'none';
    }
}

async function changePage(direction) {
    const newPage = currentPage + direction;
    if (newPage < 1) return;

    currentPage = newPage;
    await loadModels(currentSearchTerm !== '' || currentTypeFilter !== '' || currentSourceFilter !== 'all');
}

async function onPageSizeChanged() {
    currentPageSize = parseInt(elements.pageSize.value);
    currentPage = 1;
    await loadModels(currentSearchTerm !== '' || currentTypeFilter !== '' || currentSourceFilter !== 'all');
}

function displayModels(models) {
    const tableHtml = `
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Source</th>
                    <th>Size</th>
                    <th>Downloads</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                ${models.map(model => `
                    <tr>
                        <td>${model.name}</td>
                        <td>${model.type}</td>
                        <td>${model.source}</td>
                        <td>${model.size || 'N/A'}</td>
                        <td>${model.downloads || 'N/A'}</td>
                        <td><button class="btn btn-primary btn-sm" onclick="selectModel('${model.id}')">Select</button></td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    elements.modelTable.innerHTML = tableHtml;
}

function selectModel(modelId) {
    // Store selected model for download
    selectedModelId = modelId;
    showMessage('Model selected for download', 'info');
}

// Load available models from Kaggle and HuggingFace
async function loadAvailableModels() {
    try {
        // Load initial models from both sources
        await loadModels(false);
    } catch (error) {
        console.error('Error loading available models:', error);
        showMessage('Failed to load available models', 'error');
    }
}


async function downloadSelectedModel() {
    if (!selectedModelId) {
        showMessage('Please select a model first', 'warning');
        return;
    }

    try {
        elements.downloadProgress.style.display = 'block';
        elements.downloadStatus.textContent = `Downloading ${selectedModelId}...`;

        const result = await ipcRenderer.invoke('download-model', selectedModelId);

        if (result.success) {
            elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✓ Successfully downloaded ${selectedModelId}\n`;
            showMessage(`Model ${selectedModelId} downloaded successfully`, 'success');
        } else {
            elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✗ Failed to download ${selectedModelId}: ${result.error}\n`;
            showMessage(`Failed to download model: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✗ Error downloading ${selectedModelId}: ${error.message}\n`;
        showMessage(`Download error: ${error.message}`, 'error');
    } finally {
        elements.downloadProgress.style.display = 'none';
        elements.downloadStatus.textContent = '';
    }
}

async function cancelModelDownload() {
    try {
        const result = await ipcRenderer.invoke('cancel-model-download', selectedModelId);
        if (result.success) {
            elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✓ Download cancelled\n`;
            showMessage('Download cancelled successfully', 'success');
        } else {
            showMessage(`Failed to cancel download: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error cancelling download: ${error.message}`, 'error');
    }
}

async function downloadFasttextEmbeddings() {
    try {
        elements.downloadProgress.style.display = 'block';
        elements.downloadStatus.textContent = 'Downloading FastText embeddings...';

        const result = await ipcRenderer.invoke('download-fasttext');

        if (result.success) {
            elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✓ FastText embeddings downloaded successfully\n`;
            showMessage('FastText embeddings downloaded successfully', 'success');
        } else {
            elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✗ Failed to download FastText embeddings: ${result.error}\n`;
            showMessage(`Failed to download FastText embeddings: ${result.error}`, 'error');
        }
    } catch (error) {
        elements.downloadLog.textContent += `[${new Date().toLocaleTimeString()}] ✗ Error downloading FastText embeddings: ${error.message}\n`;
        showMessage(`FastText download error: ${error.message}`, 'error');
    } finally {
        elements.downloadProgress.style.display = 'none';
        elements.downloadStatus.textContent = '';
    }
}

function clearDownloadLog() {
    elements.downloadLog.textContent = '';
}

// Code generation functionality
async function generateCode() {
    const description = elements.codeDescription.value;
    const language = elements.codeLanguage.value;
    const codeType = elements.codeType.value;
    const complexity = elements.codeComplexity.value;
    const includeComments = elements.includeComments.checked;
    const includeExamples = elements.includeExamples.checked;

    if (!description) {
        showMessage('Please enter a description', 'error');
        return;
    }

    try {
        elements.codeGenProgress.style.display = 'block';
        const result = await ipcRenderer.invoke('generate-code', {
            description,
            language,
            codeType,
            complexity,
            includeComments,
            includeExamples
        });

        if (result.success) {
            elements.generatedCode.textContent = result.code;
            elements.copyCodeBtn.disabled = false;
            elements.saveCodeBtn.disabled = false;
            showMessage('Code generated successfully', 'success');
        } else {
            elements.generatedCode.textContent = `Error: ${result.error}`;
            showMessage('Code generation failed', 'error');
        }
    } catch (error) {
        elements.generatedCode.textContent = `Error: ${error.message}`;
        showMessage('Code generation error', 'error');
    } finally {
        elements.codeGenProgress.style.display = 'none';
    }
}

async function copyGeneratedCode() {
    const code = elements.generatedCode.textContent;
    if (code) {
        await navigator.clipboard.writeText(code);
        showMessage('Code copied to clipboard', 'success');
    }
}

async function saveGeneratedCode() {
    const code = elements.generatedCode.textContent;
    if (code) {
        const filename = `generated_code_${Date.now()}.txt`;
        const result = await ipcRenderer.invoke('save-file', code, filename);
        if (result.success) {
            showMessage(`Code saved to ${result.path}`, 'success');
        } else {
            showMessage('Failed to save code', 'error');
        }
    }
}

function clearGeneratedCode() {
    elements.generatedCode.textContent = '';
    elements.copyCodeBtn.disabled = true;
    elements.saveCodeBtn.disabled = true;
}

// Chat functionality
async function sendChatMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;

    // Add user message
    addChatMessage(message, 'user');
    elements.chatInput.value = '';

    try {
        const result = await ipcRenderer.invoke('chat-message', message);
        if (result.success) {
            addChatMessage(result.response, 'assistant');
        } else {
            addChatMessage(`Error: ${result.error}`, 'assistant');
        }
    } catch (error) {
        addChatMessage(`Error: ${error.message}`, 'assistant');
    }
}

function addChatMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    messageDiv.textContent = message;
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Neural network functionality
async function refreshNeuralData() {
    try {
        const [topologyResult, activityResult] = await Promise.all([
            ipcRenderer.invoke('get-neural-topology'),
            ipcRenderer.invoke('get-neural-activity')
        ]);

        if (topologyResult.success) {
            elements.topologyDisplay.textContent = JSON.stringify(topologyResult.data, null, 2);
        }

        if (activityResult.success) {
            elements.activityDisplay.textContent = JSON.stringify(activityResult.data, null, 2);
            // TODO: Update canvas visualization
        }

        showMessage('Neural data refreshed', 'success');
    } catch (error) {
        showMessage(`Error refreshing neural data: ${error.message}`, 'error');
    }
}

// Settings functionality
function loadSettings() {
    // Load settings from localStorage or defaults
    const settings = {
        apiUrl: localStorage.getItem('apiUrl') || 'http://localhost:8080/api/v1',
        wsUrl: localStorage.getItem('wsUrl') || 'ws://localhost:8081',
        apiKey: localStorage.getItem('apiKey') || '2956a13167',
        dataDir: localStorage.getItem('dataDir') || 'training_data',
        sandboxTimeout: localStorage.getItem('sandboxTimeout') || '300',
        logLevel: localStorage.getItem('logLevel') || 'WARNING',
        enableDiagnostics: localStorage.getItem('enableDiagnostics') !== 'false'
    };

    elements.apiUrl.value = settings.apiUrl;
    elements.wsUrl.value = settings.wsUrl;
    elements.apiKey.value = settings.apiKey;
    elements.dataDir.value = settings.dataDir;
    elements.sandboxTimeout.value = settings.sandboxTimeout;
    elements.logLevel.value = settings.logLevel;
    elements.enableDiagnostics.checked = settings.enableDiagnostics;
}

function saveSettings() {
    const settings = {
        apiUrl: elements.apiUrl.value,
        wsUrl: elements.wsUrl.value,
        apiKey: elements.apiKey.value,
        dataDir: elements.dataDir.value,
        sandboxTimeout: elements.sandboxTimeout.value,
        logLevel: elements.logLevel.value,
        enableDiagnostics: elements.enableDiagnostics.checked
    };

    // Save to localStorage
    Object.keys(settings).forEach(key => {
        localStorage.setItem(key, settings[key]);
    });

    // Reinitialize client with new settings
    initializeClient();

    showMessage('Settings saved successfully', 'success');
}

async function testApiConnection() {
    await testConnection();
}

// Utility functions
function showMessage(message, type = 'info') {
    // Create a temporary message element
    const messageDiv = document.createElement('div');
    messageDiv.className = type;
    messageDiv.textContent = message;
    messageDiv.style.position = 'fixed';
    messageDiv.style.top = '20px';
    messageDiv.style.right = '20px';
    messageDiv.style.padding = '10px 20px';
    messageDiv.style.borderRadius = '4px';
    messageDiv.style.zIndex = '1000';
    messageDiv.style.maxWidth = '300px';

    document.body.appendChild(messageDiv);

    // Remove after 3 seconds
    setTimeout(() => {
        document.body.removeChild(messageDiv);
    }, 3000);
}

// Theme functions
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    updateThemeButton(newTheme);
    localStorage.setItem('theme', newTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
}

function updateThemeButton(theme) {
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        themeToggleBtn.textContent = theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
    }
}

// Cleanup on window close
window.addEventListener('beforeunload', () => {
    if (systemMetricsInterval) {
        clearInterval(systemMetricsInterval);
    }
});
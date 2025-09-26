/**
 * CoreAI3D API Client
 * Easy-to-use client for making automated API calls to the CoreAI3D system
 */

class CoreAI3DClient {
    constructor(config = {}) {
        this.config = {
            baseURL: config.baseURL || 'http://localhost:8080/api/v1',
            wsURL: config.wsURL || 'ws://localhost:8081/ws',
            apiKey: config.apiKey || '',
            sessionId: config.sessionId || '',
            timeout: config.timeout || 30000,
            retries: config.retries || 3,
            retryDelay: config.retryDelay || 1000,
            ...config
        };

        this.ws = null;
        this.isConnected = false;
        this.messageHandlers = new Map();
        this.requestQueue = [];
        this.isProcessingQueue = false;
    }

    // HTTP API Methods
    async request(endpoint, method = 'GET', data = null, options = {}) {
        const url = `${this.config.baseURL}${endpoint}`;
        const requestOptions = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.config.apiKey}`,
                'X-Session-ID': this.config.sessionId,
                ...options.headers
            },
            timeout: this.config.timeout
        };

        if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
            requestOptions.body = JSON.stringify(data);
        }

        let lastError;
        for (let attempt = 0; attempt <= this.config.retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

                const response = await fetch(url, {
                    ...requestOptions,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();
                return result;
            } catch (error) {
                lastError = error;
                if (attempt < this.config.retries) {
                    await this.delay(this.config.retryDelay * Math.pow(2, attempt));
                }
            }
        }

        throw lastError;
    }

    // Convenience methods for different HTTP verbs
    async get(endpoint, options = {}) {
        return this.request(endpoint, 'GET', null, options);
    }

    async post(endpoint, data, options = {}) {
        return this.request(endpoint, 'POST', data, options);
    }

    async put(endpoint, data, options = {}) {
        return this.request(endpoint, 'PUT', data, options);
    }

    async delete(endpoint, options = {}) {
        return this.request(endpoint, 'DELETE', null, options);
    }

    async patch(endpoint, data, options = {}) {
        return this.request(endpoint, 'PATCH', data, options);
    }

    // File upload
    async uploadFile(file, metadata = {}) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', JSON.stringify(metadata));

        return this.request('/files/upload', 'POST', formData, {
            headers: {
                'Authorization': `Bearer ${this.config.apiKey}`,
                'X-Session-ID': this.config.sessionId
            }
        });
    }

    // Batch requests
    async batch(requests) {
        return this.post('/batch', { requests });
    }

    // WebSocket connection
    connectWebSocket() {
        return new Promise((resolve, reject) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                resolve();
                return;
            }

            this.ws = new WebSocket(this.config.wsURL);

            this.ws.onopen = () => {
                this.isConnected = true;
                console.log('WebSocket connected');
                resolve();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                console.log('WebSocket disconnected');
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
        });
    }

    handleWebSocketMessage(data) {
        // Handle different message types
        if (this.messageHandlers.has(data.type)) {
            this.messageHandlers.get(data.type).forEach(handler => handler(data));
        }

        // Default handlers
        switch (data.type) {
            case 'chat_response':
                this.emit('chatResponse', data);
                break;
            case 'stream_data':
                this.emit('streamData', data);
                break;
            case 'system_status':
                this.emit('systemStatus', data);
                break;
            case 'error':
                this.emit('error', data);
                break;
            default:
                this.emit('message', data);
        }
    }

    sendWebSocketMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected');
        }
    }

    // Event handling
    on(event, handler) {
        if (!this.messageHandlers.has(event)) {
            this.messageHandlers.set(event, []);
        }
        this.messageHandlers.get(event).push(handler);
    }

    off(event, handler) {
        if (this.messageHandlers.has(event)) {
            const handlers = this.messageHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.messageHandlers.has(event)) {
            this.messageHandlers.get(event).forEach(handler => handler(data));
        }
    }

    // Chat functionality
    async sendChatMessage(message, options = {}) {
        const messageData = {
            type: 'chat_message',
            content: message,
            timestamp: new Date().toISOString(),
            ...options
        };

        if (this.isConnected) {
            this.sendWebSocketMessage(messageData);
        }

        return this.post('/chat/message', messageData);
    }

    async getChatHistory(limit = 50) {
        return this.get(`/chat/history?limit=${limit}`);
    }

    async clearChatHistory() {
        return this.post('/chat/clear');
    }

    // Vision API methods
    async analyzeImage(imagePath, analysisTypes = ['classification', 'objects']) {
        return this.post('/vision/analyze', {
            imagePath,
            analysisTypes
        });
    }

    async detectObjects(imagePath, confidence = 0.5) {
        return this.post('/vision/detect', {
            imagePath,
            confidence
        });
    }

    async performOCR(imagePath) {
        return this.post('/vision/ocr', { imagePath });
    }

    async analyzeFaces(imagePath) {
        return this.post('/vision/faces', { imagePath });
    }

    async processVideo(videoPath, operations = []) {
        return this.post('/vision/video', {
            videoPath,
            operations
        });
    }

    // Audio API methods
    async speechToText(audioPath) {
        return this.post('/audio/speech-to-text', { audioPath });
    }

    async textToSpeech(text, voice = 'default') {
        return this.post('/audio/text-to-speech', { text, voice });
    }

    async analyzeAudio(audioPath) {
        return this.post('/audio/analyze', { audioPath });
    }

    async processAudio(audioPath, effects = []) {
        return this.post('/audio/process', { audioPath, effects });
    }

    // System API methods
    async getSystemMetrics() {
        return this.get('/system/metrics');
    }

    async getRunningProcesses() {
        return this.get('/system/processes');
    }

    async startApplication(appName, args = []) {
        return this.post('/system/start', { appName, args });
    }

    async stopApplication(appName) {
        return this.post('/system/stop', { appName });
    }

    async captureScreen(outputPath = null) {
        return this.post('/system/capture', { outputPath });
    }

    async automateTask(taskType, parameters = {}) {
        return this.post('/system/automate', { taskType, parameters });
    }

    // Web API methods
    async searchWeb(query, maxResults = 10) {
        return this.post('/web/search', { query, maxResults });
    }

    async getWebPage(url) {
        return this.post('/web/fetch', { url });
    }

    async extractContent(url) {
        return this.post('/web/extract', { url });
    }

    async getNews(topic, maxArticles = 10) {
        return this.post('/web/news', { topic, maxArticles });
    }

    // Math API methods
    async calculate(expression) {
        return this.post('/math/calculate', { expression });
    }

    async optimize(objective, initialGuess, method = 'gradient_descent') {
        return this.post('/math/optimize', { objective, initialGuess, method });
    }

    async getStatistics(data) {
        return this.post('/math/statistics', { data });
    }

    async matrixOperation(operation, matrixName, parameters = {}) {
        return this.post('/math/matrix', { operation, matrixName, parameters });
    }

    // Multi-modal processing
    async analyzeMultiModal(contentType, content, analysisTypes = []) {
        return this.post('/multimodal/analyze', { contentType, content, analysisTypes });
    }

    async processWithAI(content, processingType = 'general') {
        return this.post('/ai/process', { content, processingType });
    }

    // Real-time processing
    async startStream(streamType, parameters = {}) {
        return this.post('/streams/start', { streamType, parameters });
    }

    async stopStream(streamType) {
        return this.post('/streams/stop', { streamType });
    }

    async getStreamStatus(streamType) {
        return this.get(`/streams/status/${streamType}`);
    }

    // Automation and scripting
    async executeCommand(command, parameters = {}) {
        return this.post('/commands/execute', { command, parameters });
    }

    async createAutomationScript(name, commands) {
        return this.post('/automation/scripts', { name, commands });
    }

    async runAutomationScript(name, parameters = {}) {
        return this.post('/automation/run', { name, parameters });
    }

    async scheduleTask(name, schedule, command) {
        return this.post('/scheduler/tasks', { name, schedule, command });
    }

    // Monitoring and status
    async getSystemStatus() {
        return this.get('/status');
    }

    async getModuleStatus(moduleName = null) {
        const endpoint = moduleName ? `/status/modules/${moduleName}` : '/status/modules';
        return this.get(endpoint);
    }

    async getPerformanceMetrics() {
        return this.get('/metrics');
    }

    // Utility methods
    async healthCheck() {
        try {
            const response = await this.get('/health');
            return response.status === 'healthy';
        } catch (error) {
            return false;
        }
    }

    async waitForReady(timeout = 30000) {
        const startTime = Date.now();
        while (Date.now() - startTime < timeout) {
            if (await this.healthCheck()) {
                return true;
            }
            await this.delay(1000);
        }
        return false;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Session management
    async createSession(clientId = '') {
        const response = await this.post('/sessions', { clientId });
        if (response.success) {
            this.config.sessionId = response.sessionId;
        }
        return response;
    }

    async destroySession() {
        if (this.config.sessionId) {
            await this.delete(`/sessions/${this.config.sessionId}`);
            this.config.sessionId = '';
        }
    }

    // Configuration
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }

    getConfig() {
        return { ...this.config };
    }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CoreAI3DClient;
} else if (typeof window !== 'undefined') {
    window.CoreAI3DClient = CoreAI3DClient;
}

// Example usage and automation scripts
class AutomationHelper {
    constructor(client) {
        this.client = client;
    }

    // Automated image analysis pipeline
    async analyzeImageBatch(imagePaths, analysisTypes = ['classification', 'objects', 'faces']) {
        const results = [];

        for (const imagePath of imagePaths) {
            try {
                const result = await this.client.analyzeImage(imagePath, analysisTypes);
                results.push({ imagePath, result, success: true });
            } catch (error) {
                results.push({ imagePath, error: error.message, success: false });
            }
        }

        return results;
    }

    // Automated system monitoring
    async monitorSystem(duration = 60000, interval = 5000) {
        const metrics = [];
        const startTime = Date.now();

        while (Date.now() - startTime < duration) {
            try {
                const metric = await this.client.getSystemMetrics();
                metrics.push({ timestamp: new Date().toISOString(), ...metric });
                await this.client.delay(interval);
            } catch (error) {
                console.error('Monitoring error:', error);
            }
        }

        return metrics;
    }

    // Automated web scraping
    async scrapeWebContent(urls, extractors = ['content', 'links', 'images']) {
        const results = [];

        for (const url of urls) {
            try {
                const result = await this.client.getWebPage(url);
                const extracted = {};

                if (extractors.includes('content')) {
                    extracted.content = await this.client.extractContent(url);
                }

                if (extractors.includes('links')) {
                    extracted.links = result.links || [];
                }

                if (extractors.includes('images')) {
                    extracted.images = result.images || [];
                }

                results.push({ url, extracted, success: true });
            } catch (error) {
                results.push({ url, error: error.message, success: false });
            }
        }

        return results;
    }

    // Automated file processing
    async processFiles(filePaths, operations = []) {
        const results = [];

        for (const filePath of filePaths) {
            try {
                const fileInfo = await this.client.uploadFile(filePath);
                const processed = {};

                for (const operation of operations) {
                    switch (operation.type) {
                        case 'analyze':
                            processed.analysis = await this.client.analyzeMultiModal(
                                fileInfo.fileType,
                                filePath,
                                operation.analysisTypes || ['general']
                            );
                            break;
                        case 'convert':
                            processed.conversion = await this.client.processWithAI(
                                filePath,
                                'convert'
                            );
                            break;
                        default:
                            console.warn(`Unknown operation: ${operation.type}`);
                    }
                }

                results.push({ filePath, processed, success: true });
            } catch (error) {
                results.push({ filePath, error: error.message, success: false });
            }
        }

        return results;
    }

    // Automated chat interaction
    async chatWithAI(messages, options = {}) {
        const responses = [];

        for (const message of messages) {
            try {
                const response = await this.client.sendChatMessage(message, options);
                responses.push({ message, response, success: true });
            } catch (error) {
                responses.push({ message, error: error.message, success: false });
            }
        }

        return responses;
    }
}

// Export automation helper
if (typeof module !== 'undefined' && module.exports) {
    module.exports.AutomationHelper = AutomationHelper;
} else if (typeof window !== 'undefined') {
    window.AutomationHelper = AutomationHelper;
}
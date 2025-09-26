# CoreAI3D Web Interface & API

This directory contains the enhanced web interface and API client for easy automated API calls to the CoreAI3D system.

## Overview

The web interface provides:
- **Real-time chat interface** with WebSocket communication
- **RESTful API** for all AI modules (Vision, Audio, System, Web, Math)
- **File upload capabilities** for images, audio, video, and documents
- **Real-time streaming** for vision, audio, and system monitoring
- **Easy automation** with JavaScript client library
- **Multi-modal processing** combining different AI capabilities

## Files

- `enhanced-interface.jsx` - Main React web interface
- `api-client.js` - JavaScript API client for automation
- `automation-examples.js` - Example automation scripts
- `README.md` - This documentation

## Quick Start

### 1. Basic API Client Usage

```javascript
const { CoreAI3DClient } = require('./api-client');

const client = new CoreAI3DClient({
    baseURL: 'http://localhost:8080/api/v1',
    wsURL: 'ws://localhost:8081/ws',
    apiKey: 'your-api-key-here'
});

// Connect to WebSocket
await client.connectWebSocket();

// Send a chat message
const response = await client.sendChatMessage('Hello, analyze the system status');
console.log(response);
```

### 2. Automated Image Analysis

```javascript
const { AutomationHelper } = require('./api-client');

const client = new CoreAI3DClient({...});
const helper = new AutomationHelper(client);

const imagePaths = ['image1.jpg', 'image2.png', 'image3.jpeg'];
const results = await helper.analyzeImageBatch(imagePaths, [
    'classification',
    'objects',
    'faces'
]);

results.forEach(result => {
    console.log(result.imagePath, result.success ? 'Success' : 'Failed');
});
```

### 3. System Automation

```javascript
// Monitor system for 30 seconds
const metrics = await client.monitorSystem(30000, 5000);
console.log(`Average CPU: ${metrics.avgCpu}%`);

// Start an application
await client.startApplication('notepad.exe');

// Capture screen
await client.captureScreen('screenshot.png');
```

## API Endpoints

### Vision API
```javascript
// Analyze image
await client.analyzeImage('image.jpg', ['classification', 'objects']);

// Detect objects
await client.detectObjects('image.jpg', 0.5);

// OCR
await client.performOCR('document.jpg');

// Face analysis
await client.analyzeFaces('photo.jpg');
```

### Audio API
```javascript
// Speech to text
await client.speechToText('audio.wav');

// Text to speech
await client.textToSpeech('Hello world', 'en-US');

// Analyze audio
await client.analyzeAudio('recording.wav');
```

### System API
```javascript
// Get system metrics
await client.getSystemMetrics();

// List processes
await client.getRunningProcesses();

// Capture screen
await client.captureScreen('screenshot.png');

// Automate task
await client.automateTask('screenshot', { outputPath: 'auto.png' });
```

### Web API
```javascript
// Search web
await client.searchWeb('artificial intelligence', 10);

// Extract content
await client.extractContent('https://example.com');

// Get news
await client.getNews('technology', 5);
```

### Math API
```javascript
// Calculate expression
await client.calculate('2 + 2 * 3');

// Optimize function
await client.optimize('x^2 + y^2', [1, 1]);

// Statistical analysis
await client.getStatistics([1, 2, 3, 4, 5]);
```

## WebSocket Real-time Features

### Chat Interface
```javascript
client.on('chatResponse', (data) => {
    console.log('AI Response:', data.content);
});

await client.sendChatMessage('What is the current CPU usage?');
```

### Real-time Streams
```javascript
// Start vision stream
await client.startStream('vision', { source: 'camera' });

// Handle stream data
client.on('streamData', (data) => {
    if (data.streamType === 'vision') {
        console.log('Frame processed:', data.frameCount);
    }
});
```

### System Monitoring
```javascript
// Start monitoring
await client.startStream('system', { interval: 2000 });

// Handle monitoring data
client.on('streamData', (data) => {
    if (data.streamType === 'system') {
        console.log('CPU:', data.cpuUsage, 'Memory:', data.memoryUsage);
    }
});
```

## Automation Examples

### 1. Image Processing Pipeline
```javascript
async function processImages() {
    const client = new CoreAI3DClient({...});
    const helper = new AutomationHelper(client);

    const results = await helper.analyzeImageBatch(
        ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        ['classification', 'objects', 'ocr']
    );

    return results;
}
```

### 2. System Health Check
```javascript
async function systemHealthCheck() {
    const client = new CoreAI3DClient({...});

    const metrics = await client.getSystemMetrics();
    const processes = await client.getRunningProcesses();

    return {
        cpuUsage: metrics.cpuUsage,
        memoryUsage: metrics.memoryUsage,
        processCount: processes.length,
        healthy: metrics.cpuUsage < 80 && metrics.memoryUsage < 80
    };
}
```

### 3. Web Content Analysis
```javascript
async function analyzeWebContent() {
    const client = new CoreAI3DClient({...});

    const searchResults = await client.searchWeb('AI news', 10);
    const urls = searchResults.map(r => r.url);

    const helper = new AutomationHelper(client);
    const scraped = await helper.scrapeWebContent(urls, ['content', 'links']);

    return scraped;
}
```

### 4. Multi-modal Analysis
```javascript
async function multiModalAnalysis() {
    const client = new CoreAI3DClient({...});

    // Upload and analyze file
    const fileResult = await client.uploadFile('document.pdf');
    const analysis = await client.analyzeMultiModal(
        'pdf',
        'document.pdf',
        ['content', 'structure', 'entities']
    );

    return { file: fileResult, analysis };
}
```

## Configuration Options

### Client Configuration
```javascript
const client = new CoreAI3DClient({
    baseURL: 'http://localhost:8080/api/v1',    // REST API endpoint
    wsURL: 'ws://localhost:8081/ws',            // WebSocket endpoint
    apiKey: 'your-api-key',                      // Authentication key
    sessionId: 'optional-session-id',            // Session identifier
    timeout: 30000,                              // Request timeout (ms)
    retries: 3,                                  // Retry attempts
    retryDelay: 1000                             // Delay between retries (ms)
});
```

### Server Configuration
```javascript
// In your C++ server configuration
const server = new APIServer("CoreAI3D-API", "0.0.0.0", 8080);
server.setNumThreads(4);
server.setRequestTimeout(30000);
server.setMaxRequestSize(50 * 1024 * 1024); // 50MB
server.enableRequestLogging(true);
```

## Error Handling

### Basic Error Handling
```javascript
try {
    const result = await client.analyzeImage('image.jpg');
    console.log('Success:', result);
} catch (error) {
    console.error('Error:', error.message);
    if (error.response) {
        console.error('Response:', error.response.data);
    }
}
```

### Advanced Error Handling
```javascript
client.on('error', (error) => {
    console.error('API Error:', error.message, error.code);
});

client.on('connectionError', () => {
    console.error('Connection lost, attempting reconnection...');
    client.connectWebSocket();
});
```

## Performance Optimization

### Batch Processing
```javascript
// Instead of individual requests
const requests = [
    { endpoint: '/vision/analyze', data: { imagePath: 'img1.jpg' } },
    { endpoint: '/vision/analyze', data: { imagePath: 'img2.jpg' } },
    { endpoint: '/system/metrics', data: {} }
];

const results = await client.batch(requests);
```

### Connection Pooling
```javascript
// Reuse connections for multiple requests
const client = new CoreAI3DClient({ /* config */ });

// All requests share the same connection
await client.analyzeImage('img1.jpg');
await client.getSystemMetrics();
await client.searchWeb('query');
```

### Caching
```javascript
// Results are automatically cached
const result1 = await client.getSystemMetrics();
const result2 = await client.getSystemMetrics(); // Uses cache
```

## Security Considerations

### API Key Management
```javascript
// Store API keys securely
const client = new CoreAI3DClient({
    apiKey: process.env.COREAI3D_API_KEY, // From environment
    // Never hardcode API keys in source code
});
```

### Request Validation
```javascript
// Validate requests before sending
const isValid = await client.healthCheck();
if (!isValid) {
    throw new Error('Server not available');
}
```

### Rate Limiting
```javascript
// Implement rate limiting in your automation
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

for (const task of tasks) {
    await client.executeCommand(task.command, task.params);
    await delay(1000); // 1 second delay between requests
}
```

## Integration Examples

### Node.js Automation Script
```javascript
const { CoreAI3DClient, AutomationHelper } = require('./api-client');

async function main() {
    const client = new CoreAI3DClient({
        baseURL: process.env.API_URL,
        apiKey: process.env.API_KEY
    });

    const helper = new AutomationHelper(client);

    // Your automation logic here
    const systemHealth = await helper.monitorSystem(60000, 10000);
    console.log('System health report:', systemHealth);
}

main().catch(console.error);
```

### Browser Integration
```html
<!DOCTYPE html>
<html>
<head>
    <title>CoreAI3D Integration</title>
</head>
<body>
    <div id="app">
        <h1>AI Assistant</h1>
        <div id="chat"></div>
        <input id="message" placeholder="Type your message..." />
        <button onclick="sendMessage()">Send</button>
    </div>

    <script src="api-client.js"></script>
    <script>
        const client = new CoreAI3DClient({
            baseURL: 'http://localhost:8080/api/v1',
            wsURL: 'ws://localhost:8081/ws'
        });

        async function sendMessage() {
            const input = document.getElementById('message');
            const message = input.value;
            input.value = '';

            const response = await client.sendChatMessage(message);
            document.getElementById('chat').innerHTML += `
                <div>User: ${message}</div>
                <div>AI: ${response.content}</div>
            `;
        }

        // Initialize
        client.connectWebSocket();
    </script>
</body>
</html>
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if the CoreAI3D server is running
   - Verify the baseURL and wsURL are correct
   - Check firewall settings

2. **Authentication Failed**
   - Verify the API key is correct
   - Check if the session is valid
   - Ensure the server accepts the API key

3. **Timeout Errors**
   - Increase the timeout value
   - Check network connectivity
   - Verify server is not overloaded

4. **WebSocket Disconnection**
   - Check network stability
   - Implement reconnection logic
   - Verify WebSocket server is running

### Debug Mode
```javascript
// Enable debug logging
const client = new CoreAI3DClient({
    debug: true,
    logLevel: 'debug'
});

// All requests and responses will be logged
```

## API Reference

For complete API documentation, see the generated OpenAPI/Swagger documentation at:
`http://localhost:8080/api/docs`

Or check the C++ API server implementation in:
- `CoreAI3D/include/APIServer.hpp`
- `CoreAI3D/include/WebSocketServer.hpp`

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the automation examples
3. Examine the API client source code
4. Check server logs for detailed error messages
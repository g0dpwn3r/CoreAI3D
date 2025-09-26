/**
 * CoreAI3D Automation Examples
 * Demonstrates how to use the API client for automated tasks
 */

const CoreAI3DClient = require('./api-client');
const AutomationHelper = require('./api-client').AutomationHelper;

// Example 1: Basic setup and connection
async function basicSetupExample() {
    console.log('=== Basic Setup Example ===');

    // Initialize client
    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        wsURL: 'ws://localhost:8081/ws',
        apiKey: 'your-api-key-here',
        timeout: 30000
    });

    try {
        // Check if server is healthy
        const isHealthy = await client.healthCheck();
        console.log('Server health:', isHealthy ? 'OK' : 'NOT OK');

        if (isHealthy) {
            // Create a session
            const session = await client.createSession('automation-client-1');
            console.log('Session created:', session.sessionId);

            // Get system status
            const status = await client.getSystemStatus();
            console.log('System status:', status);

            // Get module status
            const modules = await client.getModuleStatus();
            console.log('Available modules:', Object.keys(modules));
        }
    } catch (error) {
        console.error('Basic setup error:', error.message);
    }
}

// Example 2: Automated image analysis pipeline
async function imageAnalysisPipeline() {
    console.log('=== Image Analysis Pipeline ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        apiKey: 'your-api-key-here'
    });

    const imagePaths = [
        '/path/to/image1.jpg',
        '/path/to/image2.png',
        '/path/to/image3.jpeg'
    ];

    try {
        const helper = new AutomationHelper(client);
        const results = await helper.analyzeImageBatch(imagePaths, [
            'classification',
            'objects',
            'faces'
        ]);

        results.forEach((result, index) => {
            console.log(`Image ${index + 1}:`, result.success ? 'Success' : 'Failed');
            if (result.success) {
                console.log('  Classifications:', result.result.classifications);
                console.log('  Objects detected:', result.result.objects?.length || 0);
                console.log('  Faces detected:', result.result.faces?.length || 0);
            } else {
                console.log('  Error:', result.error);
            }
        });
    } catch (error) {
        console.error('Image analysis error:', error.message);
    }
}

// Example 3: System monitoring and automation
async function systemMonitoringExample() {
    console.log('=== System Monitoring Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        apiKey: 'your-api-key-here'
    });

    try {
        // Start system monitoring
        console.log('Starting system monitoring...');
        const metrics = await client.monitorSystem(30000, 5000); // 30 seconds, 5 second intervals

        console.log(`Collected ${metrics.length} data points`);

        // Analyze the metrics
        const cpuUsage = metrics.map(m => m.cpuUsage);
        const memoryUsage = metrics.map(m => m.memoryUsage);

        const avgCpu = cpuUsage.reduce((a, b) => a + b, 0) / cpuUsage.length;
        const avgMemory = memoryUsage.reduce((a, b) => a + b, 0) / memoryUsage.length;

        console.log(`Average CPU usage: ${avgCpu.toFixed(2)}%`);
        console.log(`Average memory usage: ${avgMemory.toFixed(2)}%`);

        // Get current processes
        const processes = await client.getRunningProcesses();
        console.log(`Currently running processes: ${processes.length}`);

        // Start an application
        console.log('Starting calculator...');
        const startResult = await client.startApplication('calc.exe');
        console.log('Start result:', startResult.success ? 'Success' : 'Failed');

        // Wait a bit
        await client.delay(3000);

        // Stop the application
        console.log('Stopping calculator...');
        const stopResult = await client.stopApplication('Calculator');
        console.log('Stop result:', stopResult.success ? 'Success' : 'Failed');

    } catch (error) {
        console.error('System monitoring error:', error.message);
    }
}

// Example 4: Web scraping and content analysis
async function webScrapingExample() {
    console.log('=== Web Scraping Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        apiKey: 'your-api-key-here'
    });

    const urls = [
        'https://example.com',
        'https://news.ycombinator.com',
        'https://github.com'
    ];

    try {
        const helper = new AutomationHelper(client);
        const results = await helper.scrapeWebContent(urls, [
            'content',
            'links',
            'images'
        ]);

        results.forEach((result, index) => {
            console.log(`URL ${index + 1}: ${result.url}`);
            console.log('  Success:', result.success);

            if (result.success) {
                console.log('  Content length:', result.extracted.content?.length || 0);
                console.log('  Links found:', result.extracted.links?.length || 0);
                console.log('  Images found:', result.extracted.images?.length || 0);
            } else {
                console.log('  Error:', result.error);
            }
        });

        // Search for specific content
        console.log('\nSearching for "artificial intelligence"...');
        const searchResults = await client.searchWeb('artificial intelligence', 5);
        console.log(`Found ${searchResults.length} results`);

        searchResults.forEach((result, index) => {
            console.log(`${index + 1}. ${result.title}`);
            console.log(`   URL: ${result.url}`);
            console.log(`   Relevance: ${result.relevanceScore}`);
        });

    } catch (error) {
        console.error('Web scraping error:', error.message);
    }
}

// Example 5: Chat automation
async function chatAutomationExample() {
    console.log('=== Chat Automation Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        wsURL: 'ws://localhost:8081/ws',
        apiKey: 'your-api-key-here'
    });

    try {
        // Connect to WebSocket for real-time chat
        await client.connectWebSocket();

        // Set up chat response handler
        client.on('chatResponse', (data) => {
            console.log('AI Response:', data.content);
        });

        // Automated conversation
        const conversation = [
            'Hello, can you analyze the current system status?',
            'What applications are currently running?',
            'Can you capture a screenshot of the desktop?',
            'Search for recent news about AI',
            'Thank you for your help!'
        ];

        const helper = new AutomationHelper(client);
        const responses = await helper.chatWithAI(conversation);

        console.log(`\nCompleted ${responses.length} chat interactions`);

        responses.forEach((response, index) => {
            console.log(`Message ${index + 1}: ${response.success ? 'Success' : 'Failed'}`);
            if (!response.success) {
                console.log(`  Error: ${response.error}`);
            }
        });

    } catch (error) {
        console.error('Chat automation error:', error.message);
    }
}

// Example 6: File processing pipeline
async function fileProcessingExample() {
    console.log('=== File Processing Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        apiKey: 'your-api-key-here'
    });

    const filePaths = [
        '/path/to/document.pdf',
        '/path/to/image.jpg',
        '/path/to/spreadsheet.xlsx'
    ];

    try {
        const helper = new AutomationHelper(client);
        const results = await helper.processFiles(filePaths, [
            { type: 'analyze', analysisTypes: ['content', 'structure', 'metadata'] },
            { type: 'convert', format: 'text' }
        ]);

        results.forEach((result, index) => {
            console.log(`File ${index + 1}: ${result.filePath}`);
            console.log('  Success:', result.success);

            if (result.success) {
                console.log('  Analysis:', result.processed.analysis ? 'Completed' : 'Skipped');
                console.log('  Conversion:', result.processed.conversion ? 'Completed' : 'Skipped');
            } else {
                console.log('  Error:', result.error);
            }
        });

    } catch (error) {
        console.error('File processing error:', error.message);
    }
}

// Example 7: Real-time processing
async function realTimeProcessingExample() {
    console.log('=== Real-time Processing Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        wsURL: 'ws://localhost:8081/ws',
        apiKey: 'your-api-key-here'
    });

    try {
        // Connect to WebSocket
        await client.connectWebSocket();

        // Set up stream data handler
        client.on('streamData', (data) => {
            console.log('Stream data received:', data.streamType, data.data);
        });

        // Start multiple streams
        console.log('Starting vision stream...');
        await client.startStream('vision', { source: 'camera', frameRate: 10 });

        console.log('Starting system monitoring...');
        await client.startStream('system', { interval: 2000 });

        // Let it run for 30 seconds
        console.log('Streams running for 30 seconds...');
        await client.delay(30000);

        // Stop streams
        console.log('Stopping streams...');
        await client.stopStream('vision');
        await client.stopStream('system');

        console.log('Real-time processing completed');

    } catch (error) {
        console.error('Real-time processing error:', error.message);
    }
}

// Example 8: Complex automation workflow
async function complexWorkflowExample() {
    console.log('=== Complex Automation Workflow ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        wsURL: 'ws://localhost:8081/ws',
        apiKey: 'your-api-key-here'
    });

    try {
        // Connect to WebSocket
        await client.connectWebSocket();

        // Step 1: System analysis
        console.log('Step 1: Analyzing system...');
        const systemMetrics = await client.getSystemMetrics();
        console.log('System CPU:', systemMetrics.cpuUsage);
        console.log('System Memory:', systemMetrics.memoryUsage);

        // Step 2: Screen capture and analysis
        console.log('Step 2: Capturing and analyzing screen...');
        const screenshot = await client.captureScreen('/tmp/screenshot.png');
        const screenAnalysis = await client.analyzeImage('/tmp/screenshot.png', ['ocr', 'objects']);
        console.log('Screen text detected:', screenAnalysis.ocr?.text?.length || 0, 'characters');
        console.log('Objects on screen:', screenAnalysis.objects?.length || 0);

        // Step 3: Web search based on screen content
        console.log('Step 3: Searching web for relevant information...');
        const searchQuery = screenAnalysis.ocr?.text?.substring(0, 100) || 'general search';
        const searchResults = await client.searchWeb(searchQuery, 5);
        console.log('Search results:', searchResults.length);

        // Step 4: Generate report
        console.log('Step 4: Generating automation report...');
        const report = {
            timestamp: new Date().toISOString(),
            systemMetrics,
            screenAnalysis: {
                textLength: screenAnalysis.ocr?.text?.length || 0,
                objectCount: screenAnalysis.objects?.length || 0
            },
            searchResults: searchResults.length,
            recommendations: []
        };

        if (systemMetrics.cpuUsage > 80) {
            report.recommendations.push('High CPU usage detected - consider closing unused applications');
        }

        if (systemMetrics.memoryUsage > 80) {
            report.recommendations.push('High memory usage detected - consider freeing up memory');
        }

        console.log('Automation report generated');
        console.log('Recommendations:', report.recommendations);

    } catch (error) {
        console.error('Complex workflow error:', error.message);
    }
}

// Example 9: Error handling and recovery
async function errorHandlingExample() {
    console.log('=== Error Handling Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        apiKey: 'your-api-key-here',
        retries: 5,
        retryDelay: 2000
    });

    try {
        // Test with invalid endpoint
        console.log('Testing error handling with invalid request...');
        await client.get('/invalid-endpoint');
    } catch (error) {
        console.log('Caught expected error:', error.message);
    }

    try {
        // Test with timeout
        console.log('Testing timeout handling...');
        await client.request('/test-timeout', 'GET', null, { timeout: 1 });
    } catch (error) {
        console.log('Caught timeout error:', error.name);
    }

    try {
        // Test recovery
        console.log('Testing recovery mechanisms...');
        const isReady = await client.waitForReady(10000);
        console.log('System ready:', isReady);

        if (isReady) {
            const status = await client.getSystemStatus();
            console.log('System recovered successfully');
        }
    } catch (error) {
        console.error('Recovery test error:', error.message);
    }
}

// Example 10: Performance benchmarking
async function performanceBenchmarkExample() {
    console.log('=== Performance Benchmark Example ===');

    const client = new CoreAI3DClient({
        baseURL: 'http://localhost:8080/api/v1',
        apiKey: 'your-api-key-here'
    });

    const operations = [
        { name: 'Get System Status', operation: () => client.getSystemStatus() },
        { name: 'Get Module Status', operation: () => client.getModuleStatus() },
        { name: 'Simple Calculation', operation: () => client.calculate('2 + 2') },
        { name: 'Search Web', operation: () => client.searchWeb('test', 1) }
    ];

    const results = {};

    for (const op of operations) {
        const times = [];

        console.log(`Benchmarking ${op.name}...`);

        for (let i = 0; i < 10; i++) {
            const start = Date.now();
            try {
                await op.operation();
                const duration = Date.now() - start;
                times.push(duration);
            } catch (error) {
                console.log(`  Attempt ${i + 1} failed:`, error.message);
            }
        }

        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);

        results[op.name] = {
            average: avgTime,
            minimum: minTime,
            maximum: maxTime,
            samples: times.length
        };

        console.log(`  Average: ${avgTime.toFixed(2)}ms`);
        console.log(`  Min: ${minTime}ms, Max: ${maxTime}ms`);
    }

    console.log('\nBenchmark Summary:');
    Object.entries(results).forEach(([name, stats]) => {
        console.log(`${name}: ${stats.average.toFixed(2)}ms avg (${stats.samples} samples)`);
    });
}

// Run examples based on command line arguments
if (typeof require !== 'undefined' && require.main === module) {
    const args = process.argv.slice(2);
    const exampleName = args[0] || 'basic';

    console.log(`Running example: ${exampleName}\n`);

    switch (exampleName) {
        case 'basic':
            basicSetupExample();
            break;
        case 'images':
            imageAnalysisPipeline();
            break;
        case 'system':
            systemMonitoringExample();
            break;
        case 'web':
            webScrapingExample();
            break;
        case 'chat':
            chatAutomationExample();
            break;
        case 'files':
            fileProcessingExample();
            break;
        case 'realtime':
            realTimeProcessingExample();
            break;
        case 'complex':
            complexWorkflowExample();
            break;
        case 'errors':
            errorHandlingExample();
            break;
        case 'benchmark':
            performanceBenchmarkExample();
            break;
        default:
            console.log('Available examples:');
            console.log('  basic     - Basic setup and connection');
            console.log('  images    - Image analysis pipeline');
            console.log('  system    - System monitoring and automation');
            console.log('  web       - Web scraping and content analysis');
            console.log('  chat      - Chat automation');
            console.log('  files     - File processing pipeline');
            console.log('  realtime  - Real-time processing');
            console.log('  complex   - Complex automation workflow');
            console.log('  errors    - Error handling and recovery');
            console.log('  benchmark - Performance benchmarking');
    }
}
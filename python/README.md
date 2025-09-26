# CoreAI3D Python Client & Automation

This directory contains the Python client library and automation tools for easy integration with the CoreAI3D system.

## Overview

The Python client provides:
- **Async/await support** for high-performance automation
- **WebSocket real-time communication** for live data streams
- **Comprehensive API coverage** for all AI modules
- **High-level automation helpers** for common tasks
- **Error handling and retry logic** built-in
- **Type hints and documentation** for better development experience

## Files

- `coreai3d_client.py` - Main Python client library
- `automation_helper.py` - High-level automation utilities
- `python_automation_examples.py` - Comprehensive automation examples
- `README.md` - This documentation

## Quick Start

### 1. Basic Usage

```python
import asyncio
from coreai3d_client import CoreAI3DClient

async def main():
    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key-here'
    }) as client:

        # Check system health
        is_healthy = await client.health_check()
        print(f"System healthy: {is_healthy}")

        # Get system metrics
        metrics = await client.get_system_metrics()
        print(f"CPU Usage: {metrics.data.get('cpuUsage', 0)}%")

        # Send a chat message
        response = await client.send_chat_message("Hello, analyze the current system")
        print(f"AI Response: {response.data.get('content', '')}")

asyncio.run(main())
```

### 2. Using Automation Helper

```python
import asyncio
from coreai3d_client import CoreAI3DClient
from automation_helper import AutomationHelper

async def main():
    async with CoreAI3DClient({...}) as client:
        async with AutomationHelper(client) as helper:

            # Analyze multiple images
            results = await helper.analyze_image_batch([
                'image1.jpg',
                'image2.png',
                'screenshot.bmp'
            ], ['classification', 'objects', 'faces'])

            for result in results:
                print(f"✓ {result.message}")

asyncio.run(main())
```

### 3. Real-time Processing

```python
import asyncio
from coreai3d_client import CoreAI3DClient

async def main():
    async with CoreAI3DClient({...}) as client:

        # Set up event handlers
        async def on_stream_data(data):
            print(f"Stream: {data.stream_type} - {data.data}")

        client.on('stream_data', on_stream_data)

        # Start system monitoring
        await client.start_stream('system', {'interval': 2000})

        # Start vision processing
        await client.start_stream('vision', {'source': 'camera'})

        # Keep running for 30 seconds
        await asyncio.sleep(30)

        # Stop streams
        await client.stop_stream('system')
        await client.stop_stream('vision')

asyncio.run(main())
```

## Client Configuration

### Basic Configuration
```python
client = CoreAI3DClient({
    'base_url': 'http://localhost:8080/api/v1',    # REST API endpoint
    'ws_url': 'ws://localhost:8081/ws',            # WebSocket endpoint
    'api_key': 'your-api-key',                      # Authentication key
    'session_id': 'optional-session-id',            # Session identifier
    'timeout': 30.0,                                # Request timeout (seconds)
    'max_retries': 3,                               # Retry attempts
    'retry_delay': 1.0,                             # Delay between retries
    'max_concurrent': 10,                           # Max concurrent requests
    'debug': False                                  # Enable debug logging
})
```

### Advanced Configuration
```python
client = CoreAI3DClient({
    'base_url': 'https://api.yourserver.com/v1',
    'ws_url': 'wss://api.yourserver.com/ws',
    'api_key': os.getenv('COREAI3D_API_KEY'),
    'timeout': 60.0,
    'max_retries': 5,
    'retry_delay': 2.0,
    'debug': True,
    'session_id': 'persistent-session-123'
})
```

## API Methods

### Vision API
```python
# Analyze image with multiple analysis types
response = await client.analyze_image('image.jpg', ['classification', 'objects', 'faces'])

# Detect objects with confidence threshold
response = await client.detect_objects('image.jpg', confidence=0.7)

# Perform OCR
response = await client.perform_ocr('document.png')

# Analyze faces
response = await client.analyze_faces('photo.jpg')

# Process video
response = await client.process_video('video.mp4', ['frames', 'motion'])
```

### Audio API
```python
# Speech to text
response = await client.speech_to_text('audio.wav')

# Text to speech
response = await client.text_to_speech('Hello world', voice='en-US')

# Analyze audio
response = await client.analyze_audio('recording.wav')

# Process audio with effects
response = await client.process_audio('audio.wav', ['noise_reduction', 'normalization'])
```

### System API
```python
# Get system metrics
response = await client.get_system_metrics()

# Get running processes
response = await client.get_running_processes()

# Start application
response = await client.start_application('notepad.exe', ['file.txt'])

# Capture screen
response = await client.capture_screen('screenshot.png')

# Automate task
response = await client.automate_task('screenshot', {'output_path': 'auto.png'})
```

### Web API
```python
# Search web
response = await client.search_web('artificial intelligence', max_results=10)

# Get web page
response = await client.get_web_page('https://example.com')

# Extract content
response = await client.extract_content('https://example.com')

# Get news
response = await client.get_news('technology', max_articles=5)
```

### Math API
```python
# Calculate expression
response = await client.calculate('2 + 2 * 3')

# Optimize function
response = await client.optimize('x^2 + y^2', [1.0, 1.0], method='gradient_descent')

# Statistical analysis
response = await client.get_statistics([1, 2, 3, 4, 5])

# Matrix operations
response = await client.matrix_operation('determinant', 'matrix_A')
```

### Multi-modal Processing
```python
# Analyze multi-modal content
response = await client.analyze_multimodal('pdf', 'document.pdf', ['content', 'structure'])

# Process with AI
response = await client.process_with_ai('content', 'general')
```

## Automation Helper

### Image Analysis Pipeline
```python
async with AutomationHelper(client) as helper:
    results = await helper.analyze_image_batch([
        'images/photo1.jpg',
        'images/photo2.png',
        'images/diagram.png'
    ], ['classification', 'objects', 'ocr'])

    for result in results:
        print(f"✓ {result.message}")
```

### System Monitoring
```python
async with AutomationHelper(client) as helper:
    # Monitor system for 60 seconds
    result = await helper.monitor_system(duration=60.0, interval=5.0)

    if result.success:
        print(f"Average CPU: {result.data['averages']['cpuUsage']:.2f}%")
        print(f"Total samples: {result.data['samples']}")
```

### Web Scraping
```python
async with AutomationHelper(client) as helper:
    urls = ['https://site1.com', 'https://site2.com', 'https://site3.com']
    results = await helper.scrape_web_content(urls, ['content', 'links', 'metadata'])

    for result in results:
        if result.success:
            print(f"✓ Scraped {result.metadata['url']}")
```

### File Processing
```python
async with AutomationHelper(client) as helper:
    file_paths = ['doc1.pdf', 'image1.jpg', 'data.csv']
    operations = [
        {'type': 'analyze', 'analysis_types': ['content', 'structure']},
        {'type': 'process', 'processing_type': 'general'}
    ]

    results = await helper.process_files(file_paths, operations)
```

### Chat Automation
```python
async with AutomationHelper(client) as helper:
    messages = [
        "What is the current system status?",
        "Can you analyze the screen?",
        "Search for recent AI news",
        "Thank you for your help!"
    ]

    result = await helper.chat_conversation(messages)

    if result.success:
        for exchange in result.data:
            print(f"User: {exchange['user_message']}")
            print(f"AI: {exchange['ai_response']}")
```

## Real-time Processing

### Event Handling
```python
async with CoreAI3DClient({...}) as client:
    # Set up event handlers
    async def on_chat_response(data):
        print(f"Chat: {data.get('content', '')}")

    async def on_stream_data(data):
        print(f"Stream {data.stream_type}: {data.data}")

    async def on_system_status(data):
        print(f"System: CPU {data.get('cpuUsage', 0)}%")

    client.on('chat_response', on_chat_response)
    client.on('stream_data', on_stream_data)
    client.on('system_status', on_system_status)

    # Start streams
    await client.start_stream('system', {'interval': 2000})
    await client.start_stream('vision', {'source': 'camera'})

    # Keep running
    await asyncio.sleep(30)

    # Stop streams
    await client.stop_stream('system')
    await client.stop_stream('vision')
```

### Custom Event Handlers
```python
def create_vision_handler():
    frame_count = 0

    async def handle_vision_data(data):
        nonlocal frame_count
        frame_count += 1
        print(f"Frame {frame_count}: {len(data.get('objects', []))} objects detected")

    return handle_vision_data

async with CoreAI3DClient({...}) as client:
    client.on('stream_data', create_vision_handler())
    await client.start_stream('vision', {'frame_rate': 10})
```

## Error Handling

### Basic Error Handling
```python
async with CoreAI3DClient({...}) as client:
    try:
        response = await client.analyze_image('image.jpg')
        if response.success:
            print(f"Success: {response.data}")
        else:
            print(f"Error: {response.message}")
    except Exception as e:
        print(f"Exception: {e}")
```

### Advanced Error Handling
```python
async with CoreAI3DClient({...}) as client:
    # Set up error handlers
    async def on_connection_error(data):
        print("Connection lost, attempting reconnection...")
        await client.connect()

    async def on_api_error(data):
        print(f"API Error: {data.get('message', '')}")
        if data.get('status_code') == 401:
            print("Authentication failed, refreshing token...")

    client.on('connection_error', on_connection_error)
    client.on('error', on_api_error)

    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.analyze_image('image.jpg')
            if response.success:
                break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = 2 ** attempt
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            await asyncio.sleep(delay)
```

## Performance Optimization

### Concurrent Requests
```python
async def process_concurrent_requests():
    async with CoreAI3DClient({...}) as client:
        # Create multiple tasks
        tasks = [
            client.analyze_image('image1.jpg'),
            client.get_system_metrics(),
            client.search_web('AI news'),
            client.calculate('2 + 2 * 3')
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed: {result}")
            else:
                print(f"Task {i} succeeded: {result.success}")
```

### Batch Processing
```python
async def batch_processing():
    async with CoreAI3DClient({...}) as client:
        # Prepare batch requests
        batch_requests = [
            {'endpoint': '/vision/analyze', 'data': {'imagePath': 'img1.jpg'}},
            {'endpoint': '/vision/analyze', 'data': {'imagePath': 'img2.jpg'}},
            {'endpoint': '/system/metrics', 'data': {}},
            {'endpoint': '/web/search', 'data': {'query': 'test', 'maxResults': 5}}
        ]

        # Execute batch
        response = await client.post('/batch', {'requests': batch_requests})

        if response.success:
            for i, result in enumerate(response.data['results']):
                print(f"Batch {i}: {'✓' if result['success'] else '✗'}")
```

### Connection Pooling
```python
async def connection_pooling():
    # Reuse the same client instance
    client = CoreAI3DClient({...})

    async with client:
        # All requests share the same connection pool
        tasks = []
        for i in range(100):
            tasks.append(client.analyze_image(f'image_{i}.jpg'))

        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

        async def limited_request(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_request(task) for task in tasks])
```

## Security Considerations

### API Key Management
```python
import os

# Load from environment variables
client = CoreAI3DClient({
    'api_key': os.getenv('COREAI3D_API_KEY'),
    'base_url': os.getenv('COREAI3D_BASE_URL'),
    'ws_url': os.getenv('COREAI3D_WS_URL')
})

# Never hardcode API keys in source code
```

### Request Validation
```python
async def safe_request():
    async with CoreAI3DClient({...}) as client:
        # Validate connection before making requests
        is_ready = await client.wait_for_ready(timeout=10.0)
        if not is_ready:
            raise RuntimeError("System not ready")

        # Validate file exists before upload
        file_path = 'data/image.jpg'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Make request with validation
        response = await client.analyze_image(file_path)
        if not response.success:
            raise RuntimeError(f"Analysis failed: {response.message}")

        return response.data
```

### Rate Limiting
```python
async def rate_limited_requests():
    async with CoreAI3DClient({...}) as client:
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        delay = 0.1  # 100ms between requests

        async def make_request(image_path):
            async with semaphore:
                response = await client.analyze_image(image_path)
                await asyncio.sleep(delay)  # Rate limiting
                return response

        # Process images with rate limiting
        image_paths = [f'image_{i}.jpg' for i in range(100)]
        tasks = [make_request(path) for path in image_paths]

        results = await asyncio.gather(*tasks, return_exceptions=True)
```

## Integration Examples

### Command Line Tool
```python
#!/usr/bin/env python3
import asyncio
import argparse
from coreai3d_client import CoreAI3DClient

async def main():
    parser = argparse.ArgumentParser(description='CoreAI3D Command Line Tool')
    parser.add_argument('command', choices=['analyze', 'monitor', 'search', 'chat'])
    parser.add_argument('target', help='Target for the command')
    parser.add_argument('--api-key', required=True, help='API key')
    parser.add_argument('--base-url', default='http://localhost:8080/api/v1')

    args = parser.parse_args()

    async with CoreAI3DClient({
        'api_key': args.api_key,
        'base_url': args.base_url
    }) as client:

        if args.command == 'analyze':
            response = await client.analyze_image(args.target)
            print(json.dumps(response.data, indent=2))

        elif args.command == 'monitor':
            response = await client.get_system_metrics()
            print(f"CPU: {response.data.get('cpuUsage', 0)}%")
            print(f"Memory: {response.data.get('memoryUsage', 0)}%")

        elif args.command == 'search':
            response = await client.search_web(args.target, max_results=5)
            for result in response.data:
                print(f"- {result.get('title', '')}")

        elif args.command == 'chat':
            response = await client.send_chat_message(args.target)
            print(response.data.get('content', ''))

if __name__ == '__main__':
    asyncio.run(main())
```

### Web Application Integration
```python
from flask import Flask, request, jsonify
from coreai3d_client import CoreAI3DClient
import asyncio

app = Flask(__name__)

@app.route('/api/analyze', methods=['POST'])
async def analyze_image():
    data = request.json
    image_path = data.get('image_path')

    async with CoreAI3DClient({
        'api_key': 'your-api-key',
        'base_url': 'http://localhost:8080/api/v1'
    }) as client:

        response = await client.analyze_image(image_path)
        return jsonify(response.data)

@app.route('/api/chat', methods=['POST'])
async def chat():
    data = request.json
    message = data.get('message')

    async with CoreAI3DClient({
        'api_key': 'your-api-key',
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws'
    }) as client:

        response = await client.send_chat_message(message)
        return jsonify({'response': response.data.get('content', '')})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Jupyter Notebook Integration
```python
# In a Jupyter notebook cell:
import asyncio
from coreai3d_client import CoreAI3DClient, AutomationHelper

async def analyze_system():
    async with CoreAI3DClient({
        'api_key': 'your-api-key',
        'base_url': 'http://localhost:8080/api/v1'
    }) as client:

        async with AutomationHelper(client) as helper:
            # Get system metrics
            metrics = await client.get_system_metrics()
            print(f"CPU Usage: {metrics.data.get('cpuUsage', 0)}%")

            # Monitor for 10 seconds
            result = await helper.monitor_system(duration=10.0, interval=2.0)
            print(f"Monitoring completed: {result.success}")

            return metrics.data, result.data

# Run the async function
cpu_usage, monitoring_data = await analyze_system()
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```python
   # Check if server is running
   is_healthy = await client.health_check()
   if not is_healthy:
       print("Server is not available")
   ```

2. **Authentication Failed**
   ```python
   # Verify API key
   try:
       response = await client.get_system_status()
       print(f"Auth successful: {response.success}")
   except Exception as e:
       print(f"Auth failed: {e}")
   ```

3. **Timeout Errors**
   ```python
   # Increase timeout
   client = CoreAI3DClient({
       'timeout': 60.0,  # 60 seconds
       'max_retries': 5
   })
   ```

4. **WebSocket Disconnection**
   ```python
   # Implement reconnection logic
   async def maintain_connection():
       while True:
           try:
               if not client.is_connected:
                   await client.connect()
               await asyncio.sleep(10)  # Check every 10 seconds
           except Exception as e:
               print(f"Connection error: {e}")
               await asyncio.sleep(5)
   ```

### Debug Mode
```python
# Enable debug logging
client = CoreAI3DClient({
    'debug': True,
    'timeout': 30.0
})

# All requests and responses will be logged
```

### Performance Monitoring
```python
import time

async def performance_test():
    client = CoreAI3DClient({'debug': True})

    async with client:
        # Test various operations
        operations = [
            ('health_check', lambda: client.health_check()),
            ('system_metrics', lambda: client.get_system_metrics()),
            ('chat_message', lambda: client.send_chat_message('Hello')),
        ]

        for name, operation in operations:
            start_time = time.time()
            try:
                result = await operation()
                duration = time.time() - start_time
                print(f"{name}: {duration:.3f}s - {'✓' if result.success else '✗'}")
            except Exception as e:
                duration = time.time() - start_time
                print(f"{name}: {duration:.3f}s - Error: {e}")
```

## API Reference

For complete API documentation, see:
- `coreai3d_client.py` - Main client implementation
- `automation_helper.py` - High-level automation utilities
- `python_automation_examples.py` - Comprehensive examples

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the example scripts
3. Examine the client source code
4. Check server logs for detailed error messages
5. Test with the debug mode enabled
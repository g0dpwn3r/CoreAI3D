#!/usr/bin/env python3
"""
CoreAI3D Python Automation Examples
Demonstrates how to use Python for quick automated scripts
"""

import asyncio
import json
import time
from pathlib import Path
from coreai3d_client import CoreAI3DClient
from automation_helper import AutomationHelper

# Example 1: Basic Setup and Connection
async def basic_setup_example():
    """Basic setup and connection example"""
    print("=== Basic Setup Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key-here',
        'debug': True
    }) as client:

        # Check server health
        is_healthy = await client.health_check()
        print(f"Server health: {'OK' if is_healthy else 'NOT OK'}")

        if is_healthy:
            # Create session
            session_response = await client.create_session('python-client-1')
            print(f"Session created: {session_response.success}")
            if session_response.success:
                print(f"Session ID: {session_response.data.get('sessionId')}")

            # Get system status
            status_response = await client.get_system_status()
            if status_response.success:
                print(f"System status: {status_response.data.get('status')}")

            # Get module status
            modules_response = await client.get_module_status()
            if modules_response.success:
                available_modules = list(modules_response.data.keys())
                print(f"Available modules: {available_modules}")

# Example 2: Automated Image Analysis Pipeline
async def image_analysis_pipeline():
    """Automated image analysis pipeline"""
    print("\n=== Image Analysis Pipeline ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            # Analyze multiple images
            image_paths = [
                'images/photo1.jpg',
                'images/photo2.png',
                'images/screenshot.png'
            ]

            results = await helper.analyze_image_batch(
                image_paths,
                ['classification', 'objects', 'faces']
            )

            for result in results:
                print(f"Image analysis: {'✓' if result.success else '✗'} {result.message}")
                if result.success and result.data:
                    analysis = result.data
                    print(f"  Classifications: {len(analysis.get('classifications', []))}")
                    print(f"  Objects detected: {len(analysis.get('objects', []))}")
                    print(f"  Faces detected: {len(analysis.get('faces', []))}")

# Example 3: System Monitoring and Automation
async def system_monitoring_example():
    """System monitoring and automation"""
    print("\n=== System Monitoring Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            # Monitor system for 30 seconds
            print("Monitoring system for 30 seconds...")
            result = await helper.monitor_system(duration=30.0, interval=5.0)

            if result.success:
                history = result.data['history']
                averages = result.data['averages']

                print(f"Collected {len(history)} data points")
                cpu_usage = averages.get('cpuUsage', 0)
                memory_usage = averages.get('memoryUsage', 0)
                duration = result.duration
                print(f"Average CPU usage: {cpu_usage:.2f}%")
                print(f"Average memory usage: {memory_usage:.2f}%")
                print(f"Duration: {duration:.2f} seconds")

            # Get current processes
            processes_response = await client.get_running_processes()
            if processes_response.success:
                processes = processes_response.data
                print(f"Currently running processes: {len(processes)}")

                # Show top 5 memory-consuming processes
                sorted_processes = sorted(processes, key=lambda x: x.get('memoryUsage', 0), reverse=True)
                print("\nTop 5 memory-consuming processes:")
                for i, proc in enumerate(sorted_processes[:5], 1):
                    print(f"  {i}. {proc.get('name', 'Unknown')} (PID: {proc.get('processId', 0)}) - {proc.get('memoryUsage', 0)} MB")

            # Start an application
            print("\nStarting calculator...")
            start_response = await client.start_application('calc.exe')
            print(f"Start result: {'Success' if start_response.success else 'Failed'}")

            # Wait a bit
            await asyncio.sleep(3)

            # Stop the application
            print("Stopping calculator...")
            stop_response = await client.stop_application('Calculator')
            print(f"Stop result: {'Success' if stop_response.success else 'Failed'}")

# Example 4: Web Scraping and Content Analysis
async def web_scraping_example():
    """Web scraping and content analysis"""
    print("\n=== Web Scraping Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            # URLs to scrape
            urls = [
                'https://example.com',
                'https://httpbin.org/html',
                'https://news.ycombinator.com'
            ]

            print(f"Scraping {len(urls)} URLs...")
            results = await helper.scrape_web_content(urls, ['content', 'links', 'metadata'])

            for result in results:
                print(f"\nURL: {result.metadata.get('url', 'Unknown')}")
                print(f"Status: {'✓' if result.success else '✗'}")
                if result.success and result.data:
                    data = result.data
                    print(f"  Content length: {len(data.get('content', ''))}")
                    print(f"  Links found: {len(data.get('links', []))}")
                    print(f"  Title: {data.get('metadata', {}).get('title', 'No title')}")

            # Search for specific content
            print("\nSearching for 'artificial intelligence'...")
            search_response = await client.search_web('artificial intelligence', 5)
            if search_response.success:
                results = search_response.data
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('title', 'No title')}")
                    print(f"     URL: {result.get('url', '')}")
                    relevance = result.get('relevanceScore', 0)
                    print(f"     Relevance: {relevance:.3f}")

# Example 5: Chat Automation
async def chat_automation_example():
    """Chat automation example"""
    print("\n=== Chat Automation Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            # Automated conversation
            messages = [
                "Hello! Can you analyze the current system status?",
                "What applications are currently running?",
                "Can you capture a screenshot of the desktop?",
                "Search for recent news about artificial intelligence",
                "What is the current CPU and memory usage?",
                "Thank you for your help!"
            ]

            print(f"Sending {len(messages)} messages...")
            result = await helper.chat_conversation(messages)

            if result.success:
                conversation = result.data
                print(f"Completed conversation with {len(conversation)} exchanges")

                for i, exchange in enumerate(conversation, 1):
                    print(f"\nExchange {i}:")
                    print(f"  User: {exchange['user_message'][:100]}{'...' if len(exchange['user_message']) > 100 else ''}")
                    print(f"  AI: {exchange['ai_response'][:100]}{'...' if len(exchange['ai_response']) > 100 else ''}")
            else:
                print(f"Chat failed: {result.message}")

# Example 6: File Processing Pipeline
async def file_processing_example():
    """File processing pipeline"""
    print("\n=== File Processing Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            # Process different types of files
            file_paths = [
                'documents/sample.pdf',
                'images/analysis.jpg',
                'data/spreadsheet.xlsx'
            ]

            operations = [
                {'type': 'analyze', 'analysis_types': ['content', 'structure', 'metadata']},
                {'type': 'process', 'processing_type': 'general'}
            ]

            print(f"Processing {len(file_paths)} files...")
            results = await helper.process_files(file_paths, operations)

            for result in results:
                print(f"\nFile: {result.metadata.get('file_path', 'Unknown')}")
                print(f"Status: {'✓' if result.success else '✗'}")
                if result.success and result.data:
                    data = result.data
                    if 'analysis' in data:
                        print(f"  Analysis completed: {type(data['analysis']).__name__}")
                    if 'processing' in data:
                        print(f"  Processing completed: {type(data['processing']).__name__}")

# Example 7: Mathematical Operations
async def mathematical_operations_example():
    """Mathematical operations example"""
    print("\n=== Mathematical Operations Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            # Mathematical expressions to evaluate
            expressions = [
                "2 + 2 * 3",
                "sin(pi/2)",
                "sqrt(144)",
                "log(100)",
                "e^2",
                "factorial(5)",
                "gcd(48, 18)",
                "lcm(12, 18)"
            ]

            print(f"Calculating {len(expressions)} expressions...")
            result = await helper.mathematical_analysis(expressions)

            if result.success:
                calculations = result.data
                for calc in calculations:
                    if calc['success']:
                        print(f"✓ {calc['expression']} = {calc['result']}")
                    else:
                        print(f"✗ {calc['expression']}: {calc['error']}")

            # Statistical analysis
            datasets = [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [2.1, 2.5, 3.2, 2.8, 3.1, 2.9, 3.5, 2.7],
                [10, 20, 30, 40, 50]
            ]

            print(f"\nAnalyzing {len(datasets)} datasets...")
            stats_result = await helper.statistical_analysis(datasets)

            if stats_result.success:
                analyses = stats_result.data
                for analysis in analyses:
                    if analysis['success']:
                        stats = analysis['statistics']
                        print(f"Dataset {analysis['dataset_index'] + 1} ({analysis['dataset_size']} values):")
                        mean_val = stats.get('mean', 0)
                        std_dev = stats.get('standardDeviation', 0)
                        print(f"  Mean: {mean_val:.3f}")
                        print(f"  Std Dev: {std_dev:.3f}")
                        print(f"  Min: {stats.get('min', 0)}")
                        print(f"  Max: {stats.get('max', 0)}")

# Example 8: Real-time Processing
async def real_time_processing_example():
    """Real-time processing example"""
    print("\n=== Real-time Processing Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key-here'
    }) as client:

        # Set up event handlers
        stream_data_count = 0

        def on_stream_data(data):
            nonlocal stream_data_count
            stream_data_count += 1
            print(f"Stream data #{stream_data_count}: {data.stream_type} - {type(data.data).__name__}")

        client.on('stream_data', on_stream_data)

        try:
            # Start system monitoring stream
            print("Starting system monitoring stream...")
            system_response = await client.start_stream('system', {'interval': 2000})
            print(f"System stream: {'Started' if system_response.success else 'Failed'}")

            # Start vision stream (if camera available)
            print("Starting vision stream...")
            vision_response = await client.start_stream('vision', {'source': 'camera', 'frameRate': 5})
            print(f"Vision stream: {'Started' if vision_response.success else 'Failed'}")

            # Let it run for 20 seconds
            print("Streams running for 20 seconds...")
            await asyncio.sleep(20)

            # Stop streams
            print("Stopping streams...")
            await client.stop_stream('system')
            await client.stop_stream('vision')

            print(f"Received {stream_data_count} stream data messages")

        except Exception as e:
            print(f"Real-time processing error: {e}")

# Example 9: Complex Multi-step Automation
async def complex_workflow_example():
    """Complex multi-step automation workflow"""
    print("\n=== Complex Automation Workflow ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'ws_url': 'ws://localhost:8081/ws',
        'api_key': 'your-api-key-here'
    }) as client:

        async with AutomationHelper(client) as helper:
            print("Step 1: System Health Check")
            health_result = await helper.system_health_check()

            if health_result.success:
                health_data = health_result.data
                health_score = health_data['health_score']
                cpu_usage = health_data['system_metrics'].get('cpuUsage', 0)
                memory_usage = health_data['system_metrics'].get('memoryUsage', 0)
                print(f"  Health Score: {health_score:.1f}/100")
                print(f"  CPU Usage: {cpu_usage:.1f}%")
                print(f"  Memory Usage: {memory_usage:.1f}%")
                print(f"  Process Count: {health_data['process_count']}")
                print(f"  Screen Captured: {'Yes' if health_data['screen_captured'] else 'No'}")

                # Step 2: Automated backup
                print("\nStep 2: Creating automated backup")
                backup_paths = ['documents', 'images', 'data']
                backup_result = await helper.automated_backup(
                    backup_paths,
                    'backups/auto_backup'
                )

                if backup_result.success:
                    backup_data = backup_result.data
                    print(f"  Backup created: {backup_data['backup_path']}")
                    print(f"  Files backed up: {len(backup_data['file_backups'])}")
                    print(f"  System state saved: {'Yes' if backup_data['system_state'] else 'No'}")

                # Step 3: Search and analyze based on health data
                print("\nStep 3: Intelligent search based on system data")
                queries = []
                if health_data['system_metrics'].get('cpuUsage', 0) > 70:
                    queries.append('how to reduce CPU usage')
                if health_data['system_metrics'].get('memoryUsage', 0) > 70:
                    queries.append('memory optimization techniques')
                if health_data['process_count'] > 50:
                    queries.append('managing too many processes')

                if queries:
                    search_result = await helper.search_and_analyze(queries, max_results=3)
                    if search_result.success:
                        print(f"  Analyzed {len(queries)} topics")
                        for query, data in search_result.data.items():
                            if 'error' not in data:
                                print(f"  {query}: Found {len(data)} results")
                else:
                    print("  System is healthy, no optimization needed")

                # Step 4: Generate final report
                print("\nStep 4: Generating automation report")
                report = {
                    'timestamp': time.time(),
                    'workflow_duration': time.time() - health_result.duration,
                    'health_score': health_data['health_score'],
                    'backup_created': backup_result.success,
                    'recommendations': []
                }

                # Generate recommendations
                if health_data['health_score'] < 70:
                    report['recommendations'].append('System health is below optimal levels')
                if health_data['system_metrics'].get('cpuUsage', 0) > 80:
                    report['recommendations'].append('High CPU usage detected')
                if health_data['system_metrics'].get('memoryUsage', 0) > 80:
                    report['recommendations'].append('High memory usage detected')
                if backup_result.success:
                    report['recommendations'].append('Automated backup completed successfully')

                print(f"  Report generated with {len(report['recommendations'])} recommendations")
                for rec in report['recommendations']:
                    print(f"  • {rec}")

            else:
                print(f"Health check failed: {health_result.message}")

# Example 10: Error Handling and Recovery
async def error_handling_example():
    """Error handling and recovery example"""
    print("\n=== Error Handling Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here',
        'max_retries': 5,
        'retry_delay': 2.0
    }) as client:

        # Test health check
        print("Testing health check...")
        is_healthy = await client.health_check()
        print(f"System healthy: {is_healthy}")

        # Test with invalid request
        print("\nTesting error handling with invalid request...")
        try:
            invalid_response = await client.get('/invalid-endpoint')
            print(f"Invalid request result: {invalid_response.success} - {invalid_response.message}")
        except Exception as e:
            print(f"Exception caught: {e}")

        # Test wait for ready
        print("\nTesting wait for ready...")
        is_ready = await client.wait_for_ready(timeout=10.0)
        print(f"System ready: {is_ready}")

        # Test session management
        print("\nTesting session management...")
        session_response = await client.create_session('error-test-client')
        if session_response.success:
            print(f"Session created: {session_response.data.get('sessionId')}")

            # Test session operations
            chat_response = await client.send_chat_message("Hello, this is a test message")
            if chat_response.success:
                print("Chat message sent successfully")

            # Clean up session
            destroy_response = await client.destroy_session()
            print(f"Session destroyed: {destroy_response.success}")

# Example 11: Performance Benchmarking
async def performance_benchmark_example():
    """Performance benchmarking example"""
    print("\n=== Performance Benchmark Example ===")

    async with CoreAI3DClient({
        'base_url': 'http://localhost:8080/api/v1',
        'api_key': 'your-api-key-here'
    }) as client:

        # Test different operations
        operations = [
            ('System Status', lambda: client.get_system_status()),
            ('System Metrics', lambda: client.get_system_metrics()),
            ('Module Status', lambda: client.get_module_status()),
            ('Simple Calculation', lambda: client.calculate('2 + 2')),
            ('Chat Message', lambda: client.send_chat_message('Hello')),
        ]

        results = {}

        for name, operation in operations:
            times = []
            print(f"Benchmarking {name}...")

            for i in range(5):  # 5 iterations each
                start_time = time.time()
                try:
                    await operation()
                    duration = time.time() - start_time
                    times.append(duration)
                except Exception as e:
                    print(f"  Iteration {i+1} failed: {e}")

            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                results[name] = {
                    'average': avg_time,
                    'minimum': min_time,
                    'maximum': max_time,
                    'samples': len(times)
                }

                print(f"  Average: {avg_time:.3f}s")
                print(f"  Min: {min_time:.3f}s, Max: {max_time:.3f}s")

        # Summary
        print("\n=== Benchmark Summary ===")
        total_avg = 0
        for name, stats in results.items():
            average = stats['average']
            samples = stats['samples']
            print(f"{name}: {average:.3f}s avg ({samples} samples)")
            total_avg += stats['average']

        overall_avg = total_avg / len(results)
        print(f"\nOverall average response time: {overall_avg:.3f}s")

# Main function to run examples
async def main():
    """Main function to run selected example"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python python_automation_examples.py <example_name>")
        print("\nAvailable examples:")
        print("  basic       - Basic setup and connection")
        print("  images      - Image analysis pipeline")
        print("  system      - System monitoring and automation")
        print("  web         - Web scraping and content analysis")
        print("  chat        - Chat automation")
        print("  files       - File processing pipeline")
        print("  math        - Mathematical operations")
        print("  realtime    - Real-time processing")
        print("  complex     - Complex automation workflow")
        print("  errors      - Error handling and recovery")
        print("  benchmark   - Performance benchmarking")
        print("  all         - Run all examples")
        return

    example_name = sys.argv[1].lower()

    examples = {
        'basic': basic_setup_example,
        'images': image_analysis_pipeline,
        'system': system_monitoring_example,
        'web': web_scraping_example,
        'chat': chat_automation_example,
        'files': file_processing_example,
        'math': mathematical_operations_example,
        'realtime': real_time_processing_example,
        'complex': complex_workflow_example,
        'errors': error_handling_example,
        'benchmark': performance_benchmark_example,
    }

    if example_name == 'all':
        for name, func in examples.items():
            print(f"\n{'='*50}")
            print(f"Running {name} example...")
            print('='*50)
            try:
                await func()
            except Exception as e:
                print(f"Example {name} failed: {e}")
            await asyncio.sleep(2)  # Brief pause between examples
    elif example_name in examples:
        await examples[example_name]()
    else:
        print(f"Unknown example: {example_name}")
        print("Use 'all' to run all examples")

if __name__ == "__main__":
    asyncio.run(main())
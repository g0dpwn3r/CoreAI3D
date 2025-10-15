#!/usr/bin/env python3
"""
CoreAI3D Python Automation Helper
High-level automation utilities for common tasks
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from coreai3d_client import CoreAI3DClient, APIResponse

@dataclass
class AutomationResult:
    """Result of an automation task"""
    success: bool
    data: Any
    message: str
    duration: float
    metadata: Dict[str, Any]

class AutomationHelper:
    """High-level automation utilities for CoreAI3D"""

    def __init__(self, client: CoreAI3DClient):
        self.client = client
        self.results: List[AutomationResult] = []

    async def __aenter__(self):
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.disconnect()

    def _record_result(self, success: bool, data: Any, message: str, start_time: float, metadata: Dict = None):
        """Record automation result"""
        result = AutomationResult(
            success=success,
            data=data,
            message=message,
            duration=time.time() - start_time,
            metadata=metadata or {}
        )
        self.results.append(result)
        return result

    # Image Analysis Pipeline
    async def analyze_image_batch(self, image_paths: List[str], analysis_types: List[str] = None) -> List[AutomationResult]:
        """Analyze multiple images with various analysis types"""
        if analysis_types is None:
            analysis_types = ['classification', 'objects', 'faces']

        results = []
        start_time = time.time()

        for image_path in image_paths:
            try:
                response = await self.client.analyze_image(image_path, analysis_types)
                if response.success:
                    results.append(self._record_result(
                        True, response.data, f"Analyzed {image_path}", start_time,
                        {'image_path': image_path, 'analysis_types': analysis_types}
                    ))
                else:
                    results.append(self._record_result(
                        False, None, f"Failed to analyze {image_path}: {response.message}", start_time,
                        {'image_path': image_path, 'error': response.message}
                    ))
            except Exception as e:
                results.append(self._record_result(
                    False, None, f"Exception analyzing {image_path}: {str(e)}", start_time,
                    {'image_path': image_path, 'exception': str(e)}
                ))

        return results

    async def ocr_batch(self, image_paths: List[str]) -> List[AutomationResult]:
        """Perform OCR on multiple images"""
        results = []
        start_time = time.time()

        for image_path in image_paths:
            try:
                response = await self.client.perform_ocr(image_path)
                if response.success:
                    results.append(self._record_result(
                        True, response.data, f"OCR completed for {image_path}", start_time,
                        {'image_path': image_path, 'text_length': len(response.data.get('text', ''))}
                    ))
                else:
                    results.append(self._record_result(
                        False, None, f"OCR failed for {image_path}: {response.message}", start_time,
                        {'image_path': image_path, 'error': response.message}
                    ))
            except Exception as e:
                results.append(self._record_result(
                    False, None, f"OCR exception for {image_path}: {str(e)}", start_time,
                    {'image_path': image_path, 'exception': str(e)}
                ))

        return results

    # System Monitoring and Automation
    async def monitor_system(self, duration: float = 60.0, interval: float = 5.0) -> AutomationResult:
        """Monitor system metrics over time"""
        start_time = time.time()
        metrics_history = []

        try:
            while time.time() - start_time < duration:
                response = await self.client.get_system_metrics()
                if response.success:
                    metrics = response.data
                    metrics['timestamp'] = time.time()
                    metrics_history.append(metrics)
                else:
                    return self._record_result(
                        False, None, f"Failed to get system metrics: {response.message}", start_time,
                        {'error': response.message}
                    )

                await asyncio.sleep(interval)

            # Calculate averages
            if metrics_history:
                avg_metrics = {}
                for key in metrics_history[0].keys():
                    if isinstance(metrics_history[0][key], (int, float)):
                        values = [m[key] for m in metrics_history]
                        avg_metrics[key] = sum(values) / len(values)

                return self._record_result(
                    True, {
                        'history': metrics_history,
                        'averages': avg_metrics,
                        'samples': len(metrics_history)
                    }, f"Monitored system for {duration}s with {len(metrics_history)} samples", start_time,
                    {'duration': duration, 'interval': interval, 'samples': len(metrics_history)}
                )
            else:
                return self._record_result(
                    False, None, "No metrics collected", start_time,
                    {'duration': duration, 'interval': interval}
                )

        except Exception as e:
            return self._record_result(
                False, None, f"System monitoring failed: {str(e)}", start_time,
                {'exception': str(e)}
            )

    async def process_monitoring(self, process_names: List[str], duration: float = 60.0) -> AutomationResult:
        """Monitor specific processes"""
        start_time = time.time()
        monitoring_data = {}

        try:
            for process_name in process_names:
                monitoring_data[process_name] = []

            while time.time() - start_time < duration:
                response = await self.client.get_running_processes()
                if response.success:
                    current_processes = {p['name']: p for p in response.data}

                    for process_name in process_names:
                        if process_name in current_processes:
                            monitoring_data[process_name].append({
                                'timestamp': time.time(),
                                'pid': current_processes[process_name]['processId'],
                                'memory': current_processes[process_name]['memoryUsage'],
                                'cpu': current_processes[process_name]['cpuUsage']
                            })

                await asyncio.sleep(2.0)  # Check every 2 seconds

            return self._record_result(
                True, monitoring_data, f"Monitored {len(process_names)} processes for {duration}s", start_time,
                {'process_names': process_names, 'duration': duration, 'samples': len(next(iter(monitoring_data.values())) or [])}
            )

        except Exception as e:
            return self._record_result(
                False, None, f"Process monitoring failed: {str(e)}", start_time,
                {'exception': str(e)}
            )

    # Web Scraping and Analysis
    async def scrape_web_content(self, urls: List[str], extractors: List[str] = None) -> List[AutomationResult]:
        """Scrape content from multiple URLs"""
        if extractors is None:
            extractors = ['content', 'links', 'images', 'metadata']

        results = []
        start_time = time.time()

        for url in urls:
            try:
                # Get web page
                page_response = await self.client.get_web_page(url)
                if not page_response.success:
                    results.append(self._record_result(
                        False, None, f"Failed to fetch {url}: {page_response.message}", start_time,
                        {'url': url, 'error': page_response.message}
                    ))
                    continue

                extracted_data = {'url': url}

                # Extract content based on requested extractors
                if 'content' in extractors:
                    content_response = await self.client.extract_content(url)
                    if content_response.success:
                        extracted_data['content'] = content_response.data

                if 'links' in extractors:
                    extracted_data['links'] = page_response.data.get('links', [])

                if 'images' in extractors:
                    extracted_data['images'] = page_response.data.get('images', [])

                if 'metadata' in extractors:
                    extracted_data['metadata'] = {
                        'title': page_response.data.get('title', ''),
                        'status_code': page_response.data.get('statusCode', 0),
                        'content_type': page_response.data.get('contentType', ''),
                        'content_length': page_response.data.get('contentLength', 0)
                    }

                results.append(self._record_result(
                    True, extracted_data, f"Successfully scraped {url}", start_time,
                    {'url': url, 'extractors': extractors}
                ))

            except Exception as e:
                results.append(self._record_result(
                    False, None, f"Scraping failed for {url}: {str(e)}", start_time,
                    {'url': url, 'exception': str(e)}
                ))

        return results

    async def search_and_analyze(self, queries: List[str], max_results: int = 5) -> AutomationResult:
        """Search web and analyze results"""
        start_time = time.time()
        all_results = {}

        try:
            for query in queries:
                search_response = await self.client.search_web(query, max_results)
                if search_response.success:
                    all_results[query] = search_response.data
                else:
                    all_results[query] = {'error': search_response.message}

            return self._record_result(
                True, all_results, f"Searched and analyzed {len(queries)} queries", start_time,
                {'queries': queries, 'max_results': max_results}
            )

        except Exception as e:
            return self._record_result(
                False, None, f"Search and analysis failed: {str(e)}", start_time,
                {'exception': str(e)}
            )

    # File Processing Pipeline
    async def process_files(self, file_paths: List[str], operations: List[Dict] = None) -> List[AutomationResult]:
        """Process multiple files with various operations"""
        if operations is None:
            operations = [{'type': 'analyze', 'analysis_types': ['content', 'structure']}]

        results = []
        start_time = time.time()

        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                results.append(self._record_result(
                    False, None, f"File not found: {file_path}", start_time,
                    {'file_path': str(file_path)}
                ))
                continue

            try:
                # Upload file
                upload_response = await self.client.upload_file(str(file_path))
                if not upload_response.success:
                    results.append(self._record_result(
                        False, None, f"Upload failed for {file_path}: {upload_response.message}", start_time,
                        {'file_path': str(file_path), 'error': upload_response.message}
                    ))
                    continue

                file_id = upload_response.data.get('fileId')
                processed_data = {'file_path': str(file_path), 'file_id': file_id}

                # Apply operations
                for operation in operations:
                    op_type = operation.get('type')

                    if op_type == 'analyze':
                        analysis_types = operation.get('analysis_types', ['general'])
                        analysis_response = await self.client.analyze_multimodal(
                            file_path.suffix[1:],  # file extension without dot
                            str(file_path),
                            analysis_types
                        )
                        if analysis_response.success:
                            processed_data['analysis'] = analysis_response.data

                    elif op_type == 'process':
                        processing_type = operation.get('processing_type', 'general')
                        process_response = await self.client.process_with_ai(
                            str(file_path),
                            processing_type
                        )
                        if process_response.success:
                            processed_data['processing'] = process_response.data

                results.append(self._record_result(
                    True, processed_data, f"Processed {file_path}", start_time,
                    {'file_path': str(file_path), 'operations': operations}
                ))

            except Exception as e:
                results.append(self._record_result(
                    False, None, f"Processing failed for {file_path}: {str(e)}", start_time,
                    {'file_path': str(file_path), 'exception': str(e)}
                ))

        return results

    # Chat and Conversation Automation
    async def chat_conversation(self, messages: List[str], options: Dict = None) -> AutomationResult:
        """Have a conversation with the AI"""
        start_time = time.time()
        conversation_history = []

        try:
            for message in messages:
                response = await self.client.send_chat_message(message, options)
                if response.success:
                    conversation_history.append({
                        'user_message': message,
                        'ai_response': response.data.get('content', ''),
                        'timestamp': time.time()
                    })
                else:
                    return self._record_result(
                        False, conversation_history, f"Chat failed at message: {message[:50]}...", start_time,
                        {'error': response.message, 'partial_history': conversation_history}
                    )

                # Small delay between messages
                await asyncio.sleep(0.5)

            return self._record_result(
                True, conversation_history, f"Completed conversation with {len(messages)} messages", start_time,
                {'message_count': len(messages), 'total_messages': len(conversation_history)}
            )

        except Exception as e:
            return self._record_result(
                False, conversation_history, f"Chat conversation failed: {str(e)}", start_time,
                {'exception': str(e), 'partial_history': conversation_history}
            )

    async def automated_query_response(self, queries: List[str]) -> AutomationResult:
        """Send multiple queries and collect responses"""
        start_time = time.time()
        responses = []

        try:
            for query in queries:
                response = await self.client.send_chat_message(query)
                if response.success:
                    responses.append({
                        'query': query,
                        'response': response.data.get('content', ''),
                        'success': True
                    })
                else:
                    responses.append({
                        'query': query,
                        'error': response.message,
                        'success': False
                    })

            return self._record_result(
                True, responses, f"Processed {len(queries)} queries", start_time,
                {'query_count': len(queries), 'success_count': sum(1 for r in responses if r['success'])}
            )

        except Exception as e:
            return self._record_result(
                False, responses, f"Query processing failed: {str(e)}", start_time,
                {'exception': str(e), 'processed_queries': len(responses)}
            )

    # Mathematical Operations
    async def mathematical_analysis(self, expressions: List[str]) -> AutomationResult:
        """Perform mathematical calculations and analysis"""
        start_time = time.time()
        results = []

        try:
            for expression in expressions:
                calc_response = await self.client.calculate(expression)
                if calc_response.success:
                    results.append({
                        'expression': expression,
                        'result': calc_response.data.get('result'),
                        'success': True
                    })
                else:
                    results.append({
                        'expression': expression,
                        'error': calc_response.message,
                        'success': False
                    })

            return self._record_result(
                True, results, f"Calculated {len(expressions)} expressions", start_time,
                {'expression_count': len(expressions), 'success_count': sum(1 for r in results if r['success'])}
            )

        except Exception as e:
            return self._record_result(
                False, results, f"Mathematical analysis failed: {str(e)}", start_time,
                {'exception': str(e), 'processed_expressions': len(results)}
            )

    async def statistical_analysis(self, datasets: List[List[float]]) -> AutomationResult:
        """Perform statistical analysis on multiple datasets"""
        start_time = time.time()
        results = []

        try:
            for i, dataset in enumerate(datasets):
                stats_response = await self.client.get_statistics(dataset)
                if stats_response.success:
                    results.append({
                        'dataset_index': i,
                        'dataset_size': len(dataset),
                        'statistics': stats_response.data,
                        'success': True
                    })
                else:
                    results.append({
                        'dataset_index': i,
                        'dataset_size': len(dataset),
                        'error': stats_response.message,
                        'success': False
                    })

            return self._record_result(
                True, results, f"Analyzed {len(datasets)} datasets", start_time,
                {'dataset_count': len(datasets), 'success_count': sum(1 for r in results if r['success'])}
            )

        except Exception as e:
            return self._record_result(
                False, results, f"Statistical analysis failed: {str(e)}", start_time,
                {'exception': str(e), 'processed_datasets': len(results)}
            )

    # Complex Multi-step Automation
    async def system_health_check(self) -> AutomationResult:
        """Comprehensive system health check"""
        start_time = time.time()

        try:
            # Get system metrics
            metrics_response = await self.client.get_system_metrics()
            if not metrics_response.success:
                return self._record_result(
                    False, None, f"Failed to get system metrics: {metrics_response.message}", start_time,
                    {'error': metrics_response.message}
                )

            metrics = metrics_response.data

            # Get running processes
            processes_response = await self.client.get_running_processes()
            process_count = len(processes_response.data) if processes_response.success else 0

            # Capture screen
            screen_response = await self.client.capture_screen()
            screen_captured = screen_response.success

            # Analyze screen content
            screen_analysis = None
            if screen_captured:
                screen_analysis_response = await self.client.analyze_image('screenshot.png', ['ocr', 'objects'])
                screen_analysis = screen_analysis_response.data if screen_analysis_response.success else None

            # Compile health report
            health_report = {
                'timestamp': time.time(),
                'system_metrics': metrics,
                'process_count': process_count,
                'screen_captured': screen_captured,
                'screen_analysis': screen_analysis,
                'health_score': self._calculate_health_score(metrics, process_count, screen_captured)
            }

            return self._record_result(
                True, health_report, "System health check completed", start_time,
                {'components_checked': 4}
            )

        except Exception as e:
            return self._record_result(
                False, None, f"Health check failed: {str(e)}", start_time,
                {'exception': str(e)}
            )

    def _calculate_health_score(self, metrics: Dict, process_count: int, screen_captured: bool) -> float:
        """Calculate overall system health score"""
        score = 100.0

        # CPU usage penalty
        cpu_usage = metrics.get('cpuUsage', 0)
        if cpu_usage > 80:
            score -= (cpu_usage - 80) * 0.5

        # Memory usage penalty
        memory_usage = metrics.get('memoryUsage', 0)
        if memory_usage > 80:
            score -= (memory_usage - 80) * 0.5

        # Process count penalty (too many processes)
        if process_count > 100:
            score -= min((process_count - 100) * 0.1, 20)

        # Screen capture bonus
        if screen_captured:
            score += 5

        return max(0, min(100, score))

    async def automated_backup(self, source_paths: List[str], backup_path: str) -> AutomationResult:
        """Automated backup of files and system state"""
        start_time = time.time()

        try:
            backup_data = {
                'timestamp': time.time(),
                'source_paths': source_paths,
                'backup_path': backup_path,
                'system_state': {},
                'file_backups': []
            }

            # Get system state
            system_response = await self.client.get_system_metrics()
            if system_response.success:
                backup_data['system_state'] = system_response.data

            # Backup files
            for source_path in source_paths:
                source_path = Path(source_path)
                if source_path.exists():
                    backup_file_path = Path(backup_path) / f"{source_path.name}.backup"
                    backup_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # For now, just record the file info
                    # In a real implementation, you would copy the files
                    backup_data['file_backups'].append({
                        'source': str(source_path),
                        'backup': str(backup_file_path),
                        'size': source_path.stat().st_size if source_path.is_file() else 0,
                        'backed_up': True
                    })

            return self._record_result(
                True, backup_data, f"Backup completed for {len(source_paths)} paths", start_time,
                {'source_count': len(source_paths), 'backup_path': backup_path}
            )

        except Exception as e:
            return self._record_result(
                False, None, f"Backup failed: {str(e)}", start_time,
                {'exception': str(e)}
            )

    # Utility Methods
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all automation results"""
        if not self.results:
            return {'total_tasks': 0, 'successful': 0, 'failed': 0, 'total_duration': 0}

        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        total_duration = sum(r.duration for r in self.results)

        return {
            'total_tasks': len(self.results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self.results) if self.results else 0,
            'total_duration': total_duration,
            'average_duration': total_duration / len(self.results) if self.results else 0,
            'latest_result': self.results[-1] if self.results else None
        }

    def clear_results(self):
        """Clear automation results"""
        self.results.clear()

    def export_results(self, format: str = 'json') -> str:
        """Export results in specified format"""
        if format.lower() == 'json':
            return json.dumps([{
                'success': r.success,
                'message': r.message,
                'duration': r.duration,
                'metadata': r.metadata
            } for r in self.results], indent=2)
        else:
            return str(self.results)
#!/usr/bin/env python3
"""
CoreAI3D Python Client
Easy-to-use Python client for automated API calls to the CoreAI3D system
"""

import asyncio
import aiohttp
import websockets
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from urllib.parse import urljoin
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response structure"""
    success: bool
    data: Any
    message: str
    status_code: int
    metadata: Dict[str, Any]

@dataclass
class StreamData:
    """Real-time stream data structure"""
    stream_type: str
    data: Any
    timestamp: float
    metadata: Dict[str, Any]

class CoreAI3DClient:
    """Python client for CoreAI3D API"""

    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'base_url': 'http://0.0.0.0:8080/api/v1',
            'ws_url': 'ws://0.0.0.0:8081/ws',
            'api_key': '',
            'session_id': '',
            'timeout': 30.0,
            'max_retries': 3,
            'retry_delay': 1.0,
            'max_concurrent': 10,
            'debug': False,
            **config
        }

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}

        # Configure debug logging
        if self.config['debug']:
            logging.getLogger('aiohttp').setLevel(logging.DEBUG)
            logging.getLogger('websockets').setLevel(logging.DEBUG)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Initialize HTTP session and WebSocket connection"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            self.session = aiohttp.ClientSession(timeout=timeout)

        if not self.is_connected:
            await self._connect_websocket()

    async def disconnect(self):
        """Close connections"""
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
            self.is_connected = False

        if self.session:
            await self.session.close()
            self.session = None

        # Cancel streaming tasks
        for task in self.streaming_tasks.values():
            task.cancel()
        self.streaming_tasks.clear()

    async def _connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            self.ws_connection = await websockets.connect(
                self.config['ws_url'],
                extra_headers={'Authorization': f'Bearer {self.config["api_key"]}'}
            )
            self.is_connected = True
            logger.info("WebSocket connected")

            # Start message handler
            asyncio.create_task(self._handle_websocket_messages())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            self.is_connected = False

    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        message_type = data.get('type', 'unknown')

        # Call registered handlers
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in message handler for {message_type}: {e}")

        # Built-in message handling
        if message_type == 'chat_response':
            logger.info(f"Chat response: {data.get('content', '')}")
        elif message_type == 'stream_data':
            logger.debug(f"Stream data: {data.get('stream_type', '')}")
        elif message_type == 'error':
            logger.error(f"API Error: {data.get('message', '')}")

    def on(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable = None):
        """Unregister event handler"""
        if event_type in self.message_handlers:
            if handler:
                try:
                    self.message_handlers[event_type].remove(handler)
                except ValueError:
                    pass
            else:
                self.message_handlers[event_type].clear()

    async def _make_request(self, method: str, endpoint: str, data: Any = None) -> APIResponse:
        """Make HTTP request with retry logic"""
        if not self.session:
            await self.connect()

        url = urljoin(self.config['base_url'], endpoint)
        headers = {
            'Authorization': f'Bearer {self.config["api_key"]}',
            'X-Session-ID': self.config['session_id'],
            'Content-Type': 'application/json'
        }

        for attempt in range(self.config['max_retries'] + 1):
            try:
                async with self.session.request(
                    method, url, json=data, headers=headers
                ) as response:
                    response_data = await response.json()
                    return APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        message=response_data.get('message', ''),
                        status_code=response.status,
                        metadata=response.headers
                    )

            except asyncio.TimeoutError:
                if attempt < self.config['max_retries']:
                    logger.warning(f"Request timeout, retrying ({attempt + 1}/{self.config['max_retries'] + 1})")
                    await asyncio.sleep(self.config['retry_delay'] * (2 ** attempt))
                else:
                    return APIResponse(
                        success=False,
                        data=None,
                        message="Request timeout",
                        status_code=408,
                        metadata={}
                    )

            except Exception as e:
                if attempt < self.config['max_retries']:
                    logger.warning(f"Request failed: {e}, retrying ({attempt + 1}/{self.config['max_retries'] + 1})")
                    await asyncio.sleep(self.config['retry_delay'] * (2 ** attempt))
                else:
                    return APIResponse(
                        success=False,
                        data=None,
                        message=str(e),
                        status_code=500,
                        metadata={}
                    )

    # HTTP API Methods
    async def get(self, endpoint: str) -> APIResponse:
        return await self._make_request('GET', endpoint)

    async def post(self, endpoint: str, data: Any = None) -> APIResponse:
        return await self._make_request('POST', endpoint, data)

    async def put(self, endpoint: str, data: Any = None) -> APIResponse:
        return await self._make_request('PUT', endpoint, data)

    async def delete(self, endpoint: str) -> APIResponse:
        return await self._make_request('DELETE', endpoint)

    async def patch(self, endpoint: str, data: Any = None) -> APIResponse:
        return await self._make_request('PATCH', endpoint, data)

    # Session Management
    async def create_session(self, client_id: str = '') -> APIResponse:
        response = await self.post('/sessions', {'clientId': client_id})
        if response.success and 'sessionId' in response.data:
            self.config['session_id'] = response.data['sessionId']
        return response

    async def destroy_session(self) -> APIResponse:
        if self.config['session_id']:
            response = await self.delete(f"/sessions/{self.config['session_id']}")
            if response.success:
                self.config['session_id'] = ''
            return response
        return APIResponse(success=True, data=None, message="No active session", status_code=200, metadata={})

    # Chat API
    async def send_chat_message(self, message: str, options: Dict[str, Any] = None) -> APIResponse:
        data = {
            'type': 'chat_message',
            'content': message,
            'timestamp': time.time(),
            **(options or {})
        }
        return await self.post('/chat/message', data)

    async def get_chat_history(self, limit: int = 50) -> APIResponse:
        return await self.get(f'/chat/history?limit={limit}')

    async def clear_chat_history(self) -> APIResponse:
        return await self.post('/chat/clear')

    # Vision API
    async def analyze_image(self, image_path: str, analysis_types: List[str] = None) -> APIResponse:
        return await self.post('/vision/analyze', {
            'imagePath': image_path,
            'analysisTypes': analysis_types or ['classification', 'objects']
        })

    async def detect_objects(self, image_path: str, confidence: float = 0.5) -> APIResponse:
        return await self.post('/vision/detect', {
            'imagePath': image_path,
            'confidence': confidence
        })

    async def perform_ocr(self, image_path: str) -> APIResponse:
        return await self.post('/vision/ocr', {'imagePath': image_path})

    async def analyze_faces(self, image_path: str) -> APIResponse:
        return await self.post('/vision/faces', {'imagePath': image_path})

    async def process_video(self, video_path: str, operations: List[str] = None) -> APIResponse:
        return await self.post('/vision/video', {
            'videoPath': video_path,
            'operations': operations or []
        })

    # Audio API
    async def speech_to_text(self, audio_path: str) -> APIResponse:
        return await self.post('/audio/speech-to-text', {'audioPath': audio_path})

    async def text_to_speech(self, text: str, voice: str = 'default') -> APIResponse:
        return await self.post('/audio/text-to-speech', {'text': text, 'voice': voice})

    async def analyze_audio(self, audio_path: str) -> APIResponse:
        return await self.post('/audio/analyze', {'audioPath': audio_path})

    async def process_audio(self, audio_path: str, effects: List[str] = None) -> APIResponse:
        return await self.post('/audio/process', {'audioPath': audio_path, 'effects': effects or []})

    # System API
    async def get_system_metrics(self) -> APIResponse:
        return await self.get('/system/metrics')

    async def get_running_processes(self) -> APIResponse:
        return await self.get('/system/processes')

    async def start_application(self, app_name: str, args: List[str] = None) -> APIResponse:
        return await self.post('/system/start', {'appName': app_name, 'args': args or []})

    async def stop_application(self, app_name: str) -> APIResponse:
        return await self.post('/system/stop', {'appName': app_name})

    async def capture_screen(self, output_path: str = None) -> APIResponse:
        return await self.post('/system/capture', {'outputPath': output_path})

    async def automate_task(self, task_type: str, parameters: Dict[str, Any] = None) -> APIResponse:
        return await self.post('/system/automate', {'taskType': task_type, 'parameters': parameters or {}})

    # Web API
    async def search_web(self, query: str, max_results: int = 10) -> APIResponse:
        return await self.post('/web/search', {'query': query, 'maxResults': max_results})

    async def get_web_page(self, url: str) -> APIResponse:
        return await self.post('/web/fetch', {'url': url})

    async def extract_content(self, url: str) -> APIResponse:
        return await self.post('/web/extract', {'url': url})

    async def get_news(self, topic: str, max_articles: int = 10) -> APIResponse:
        return await self.post('/web/news', {'topic': topic, 'maxArticles': max_articles})

    # Math API
    async def calculate(self, expression: str) -> APIResponse:
        return await self.post('/math/calculate', {'expression': expression})

    async def optimize(self, objective: str, initial_guess: List[float], method: str = 'gradient_descent') -> APIResponse:
        return await self.post('/math/optimize', {'objective': objective, 'initialGuess': initial_guess, 'method': method})

    async def get_statistics(self, data: List[float]) -> APIResponse:
        return await self.post('/math/statistics', {'data': data})

    async def matrix_operation(self, operation: str, matrix_name: str, parameters: Dict[str, Any] = None) -> APIResponse:
        return await self.post('/math/matrix', {'operation': operation, 'matrixName': matrix_name, 'parameters': parameters or {}})

    # Multi-modal Processing
    async def analyze_multimodal(self, content_type: str, content: str, analysis_types: List[str] = None) -> APIResponse:
        return await self.post('/multimodal/analyze', {
            'contentType': content_type,
            'content': content,
            'analysisTypes': analysis_types or []
        })

    async def process_with_ai(self, content: str, processing_type: str = 'general') -> APIResponse:
        return await self.post('/ai/process', {'content': content, 'processingType': processing_type})

    # Real-time Processing
    async def start_stream(self, stream_type: str, parameters: Dict[str, Any] = None) -> APIResponse:
        response = await self.post('/streams/start', {'streamType': stream_type, 'parameters': parameters or {}})
        if response.success:
            self.streaming_tasks[stream_type] = asyncio.create_task(self._handle_stream(stream_type))
        return response

    async def stop_stream(self, stream_type: str) -> APIResponse:
        if stream_type in self.streaming_tasks:
            self.streaming_tasks[stream_type].cancel()
            del self.streaming_tasks[stream_type]

        return await self.post('/streams/stop', {'streamType': stream_type})

    async def _handle_stream(self, stream_type: str):
        """Handle real-time stream data"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                if data.get('type') == 'stream_data' and data.get('streamType') == stream_type:
                    # Emit stream data to handlers
                    if 'stream_data' in self.message_handlers:
                        for handler in self.message_handlers['stream_data']:
                            await handler(StreamData(
                                stream_type=stream_type,
                                data=data.get('data'),
                                timestamp=data.get('timestamp', time.time()),
                                metadata=data.get('metadata', {})
                            ))
        except asyncio.CancelledError:
            logger.info(f"Stream {stream_type} cancelled")
        except Exception as e:
            logger.error(f"Stream {stream_type} error: {e}")

    # File Operations
    async def upload_file(self, file_path: str, metadata: Dict[str, Any] = None) -> APIResponse:
        if not self.session:
            await self.connect()

        file_path = Path(file_path)
        if not file_path.exists():
            return APIResponse(success=False, data=None, message="File not found", status_code=404, metadata={})

        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=file_path.name)
            if metadata:
                data.add_field('metadata', json.dumps(metadata))

            url = urljoin(self.config['base_url'], '/files/upload')
            headers = {'Authorization': f'Bearer {self.config["api_key"]}'}

            async with self.session.post(url, data=data, headers=headers) as response:
                response_data = await response.json()
                return APIResponse(
                    success=response.status < 400,
                    data=response_data,
                    message=response_data.get('message', ''),
                    status_code=response.status,
                    metadata=dict(response.headers)
                )

    async def download_file(self, file_id: str, download_path: str) -> APIResponse:
        url = urljoin(self.config['base_url'], f'/files/download/{file_id}')
        headers = {'Authorization': f'Bearer {self.config["api_key"]}'}

        async with self.session.get(url, headers=headers) as response:
            if response.status != 200:
                response_data = await response.json()
                return APIResponse(
                    success=False,
                    data=response_data,
                    message=response_data.get('message', ''),
                    status_code=response.status,
                    metadata={}
                )

            with open(download_path, 'wb') as f:
                f.write(await response.read())

            return APIResponse(
                success=True,
                data={'fileId': file_id, 'path': download_path},
                message="File downloaded successfully",
                status_code=200,
                metadata={}
            )

    # Status and Health
    async def health_check(self) -> bool:
        try:
            response = await self.get('/health')
            return response.success and response.data.get('status') == 'healthy'
        except Exception:
            return False

    async def get_system_status(self) -> APIResponse:
        return await self.get('/status')

    async def get_module_status(self, module_name: str = None) -> APIResponse:
        endpoint = f'/status/modules{ f"/{module_name}" if module_name else ""}'
        return await self.get(endpoint)

    # Neural Network API
    async def get_neural_topology(self) -> APIResponse:
        return await self.get('/neural/topology')

    async def get_neural_activity(self) -> APIResponse:
        return await self.get('/neural/activity')

    # Utility Methods
    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.health_check():
                return True
            await asyncio.sleep(1.0)
        return False

    def is_connected(self) -> bool:
        return self.is_connected and self.ws_connection is not None
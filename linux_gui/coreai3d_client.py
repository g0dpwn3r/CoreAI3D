#!/usr/bin/env python3
"""
CoreAI3D Python Client
Easy-to-use Python client for automated API calls to the CoreAI3D system
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from urllib.parse import urljoin
import os
from pathlib import Path
import qasync

# Configure logging with reduced verbosity
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger(__name__)

# Reduce verbosity of third-party libraries
logging.getLogger('websocket').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

@dataclass
class APIResponse:
    """Standardized API response structure"""
    success: bool
    data: Optional[Any]
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
            'base_url': 'http://localhost:8080/api/v1',
            'ws_url': 'ws://localhost:8081',
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
        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._is_connected: bool = False
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 10
        self._reconnect_backoff_base: float = 1.0
        self._reconnect_backoff_max: float = 60.0
        self._connecting: bool = False
        self._shutdown: bool = False
        self._connection_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()

        # Configure debug logging
        if self.config['debug']:
            logging.getLogger('aiohttp').setLevel(logging.DEBUG)
            logging.getLogger('aiohttp').setLevel(logging.DEBUG)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def _ensure_http_session(self):
        """Ensure we have a valid HTTP session, create one if needed"""
        if self.session is None:
            try:
                # Check if we have a running event loop (important for qasync integration)
                asyncio.get_running_loop()
                logger.info("Creating new HTTP session")
                timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
                self.session = aiohttp.ClientSession(timeout=timeout)
            except RuntimeError:
                # No running event loop yet, defer session creation
                logger.debug("No running event loop, deferring HTTP session creation")
                return False
        return True

    async def connect(self):
        """Initialize HTTP session and WebSocket connection"""
        logger.info("Starting client connection process")

        # Check shutdown flag
        if self._shutdown:
            logger.warning("Client is shutting down, skipping connection")
            return

        async with self._connection_lock:
            if not self._connecting and not self._is_connected and not self._shutdown:
                self._connecting = True
                try:
                    # Ensure we have a valid session before attempting WebSocket connection
                    if not await self._ensure_session():
                        logger.error("Failed to establish session, cannot proceed with WebSocket connection")
                        return

                    await self._connect_websocket()
                    # Add reconnection logic for WebSocket
                    if self._is_connected and self.ws_connection and not self._shutdown:
                        # Set up automatic reconnection on connection loss
                        logger.info("Starting connection monitor task")
                        self._monitor_task = asyncio.create_task(self._monitor_connection())
                finally:
                    self._connecting = False

    async def disconnect(self):
        """Close connections"""
        logger.info("Starting client disconnect process")

        # Set shutdown flag to prevent new operations
        async with self._state_lock:
            self._shutdown = True
            self._is_connected = False

        # Stop monitor task first to prevent reconnection attempts
        if self._monitor_task is not None and not self._monitor_task.done():
            logger.info("Stopping monitor task")
            self._monitor_task.cancel()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._monitor_task = None

        # Stop streaming tasks
        for task_name, task in list(self.streaming_tasks.items()):
            if not task.done():
                logger.info(f"Stopping streaming task: {task_name}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        self.streaming_tasks.clear()

        # Stop WebSocket task
        if self._ws_task is not None and not self._ws_task.done():
            logger.info("Stopping WebSocket task")
            self._ws_task.cancel()
            try:
                await asyncio.wait_for(self._ws_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._ws_task = None

        # Close WebSocket connection
        if self.ws_connection:
            logger.info("Closing WebSocket connection")
            try:
                await self.ws_connection.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            self.ws_connection = None

        # Close HTTP session
        if self.session:
            logger.info("Closing HTTP session")
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            self.session = None

        # Reset connection state and reconnection counters
        self._is_connected = False
        self._reconnect_attempts = 0
        logger.info("Client disconnected successfully")

    async def _connect_websocket(self):
        """Establish WebSocket connection with improved error handling and diagnostics"""
        async with self._connection_lock:
            if self._connecting:
                logger.warning("Connection already in progress, skipping")
                return

            self._connecting = True
            try:
                # Check shutdown flag
                if self._shutdown:
                    logger.warning("Client is shutting down, skipping WebSocket connection")
                    return

                try:
                    logger.info(f"Attempting WebSocket connection to {self.config['ws_url']}")
                    logger.info(f"Connection parameters: timeout={self.config['timeout']}, debug={self.config.get('debug', False)}")

                    # Ensure we have a valid HTTP session before WebSocket connection
                    if not await self._ensure_http_session():
                        logger.error("Cannot establish WebSocket connection - HTTP session not available")
                        return

                    # Ensure any existing connection is properly closed
                    if self.ws_connection:
                        logger.info("Closing existing WebSocket connection")
                        try:
                            await self.ws_connection.close()
                        except Exception as e:
                            logger.warning(f"Error closing existing connection: {e}")
                        self.ws_connection = None

                    # Stop any existing WebSocket task
                    if self._ws_task is not None and not self._ws_task.done():
                        logger.info("Stopping existing WebSocket task")
                        self._ws_task.cancel()
                        try:
                            await asyncio.wait_for(self._ws_task, timeout=1.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        self._ws_task = None

                    # Connect to WebSocket with aiohttp
                    logger.info("Creating WebSocket connection...")
                    logger.info(f"WebSocket URL: {self.config['ws_url']}")

                    # Prepare authentication headers (same as HTTP requests)
                    api_key = self.config["api_key"].replace('\n', '').replace('\r', '')
                    session_id = self.config['session_id'].replace('\n', '').replace('\r', '')
                    headers = {
                        'Authorization': f'Bearer {api_key}',
                        'X-Session-ID': session_id
                    }

                    self.ws_connection = await self.session.ws_connect(
                        self.config['ws_url'],
                        headers=headers
                    )
                    logger.info("WebSocket connection established successfully")

                    async with self._state_lock:
                        if not self._shutdown:
                            self._is_connected = True
                            logger.info("WebSocket connected successfully")

                            # Start message handler as a task
                            self._ws_task = asyncio.create_task(self._handle_websocket_messages())
                            logger.info("WebSocket message handler task started")
                        else:
                            # Shutdown occurred during connection, close it
                            logger.warning("Shutdown occurred during connection, closing WebSocket")
                            try:
                                await self.ws_connection.close()
                            except Exception:
                                pass
                            self.ws_connection = None

                except asyncio.TimeoutError:
                    logger.error("WebSocket connection timed out")
                    logger.error("Possible causes: server not running, firewall blocking, network issues, or incorrect URL")
                    async with self._state_lock:
                        self._is_connected = False
                        self.ws_connection = None
                        self._ws_task = None
                    self._connecting = False
                except aiohttp.ClientConnectorError as e:
                    logger.error(f"WebSocket connection failed: {e}")
                    logger.error(f"URL provided: {self.config['ws_url']}")
                    async with self._state_lock:
                        self._is_connected = False
                        self.ws_connection = None
                        self._ws_task = None
                    self._connecting = False
                except Exception as e:
                    logger.error(f"WebSocket connection failed: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    async with self._state_lock:
                        self._is_connected = False
                        self.ws_connection = None
                        self._ws_task = None
                    # Don't re-raise the exception to allow retry logic to work
                    self._connecting = False
            finally:
                self._connecting = False

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            # Check if connection is still valid before starting
            if not self.ws_connection:
                logger.warning("WebSocket connection is None, exiting handler")
                return

            logger.info("WebSocket message handler started")

            async for msg in self.ws_connection:
                if self._shutdown:
                    break

                try:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        logger.debug(f"Received WebSocket message: {len(msg.data)} bytes")
                        data = json.loads(msg.data)
                        self._process_message(data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {self.ws_connection.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.info("WebSocket connection closed")
                        break

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {msg.data}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    break

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            # Ensure connection state is updated
            async with self._state_lock:
                self._is_connected = False
            logger.info("WebSocket message handler exited")
            # Clean up connection reference
            if self.ws_connection:
                self.ws_connection = None

    def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        message_type = data.get('type', 'unknown')
        logger.info(f"Received WebSocket message: type={message_type}, data={data}")

        # Call registered handlers (single invocation for each message type)
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in message handler for {message_type}: {e}")

        # Built-in message handling for logging
        if message_type == 'stream_data':
            logger.debug(f"Stream data: {data.get('stream_type', '')}")
        elif message_type == 'error':
            logger.error(f"API Error: {data.get('message', '')}")

    def on(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Optional[Callable] = None):
        """Unregister event handler"""
        if event_type in self.message_handlers:
            if handler:
                try:
                    self.message_handlers[event_type].remove(handler)
                except ValueError:
                    pass
            else:
                self.message_handlers[event_type].clear()

    def register_chat_handler(self, handler: Callable):
        """Register a chat response handler"""
        self.on('chat_response', handler)

    async def _make_request(self, method: str, endpoint: str, data: Optional[Any] = None) -> APIResponse:
        """Make HTTP request with retry logic"""
        # Ensure we have a session
        if not await self._ensure_http_session():
            return APIResponse(
                success=False,
                data=None,
                message="HTTP session not available - event loop not running",
                status_code=503,
                metadata={}
            )

        # Sanitize api_key and session_id to prevent header injection
        api_key = self.config["api_key"].replace('\n', '').replace('\r', '')
        session_id = self.config['session_id'].replace('\n', '').replace('\r', '')

        url = urljoin(self.config['base_url'], endpoint)
        headers = {
            'Authorization': f'Bearer {api_key}',
            'X-Session-ID': session_id,
            'Content-Type': 'application/json'
        }

        for attempt in range(self.config['max_retries'] + 1):
            try:
                async with self.session.request(method.upper(), url, json=data, headers=headers) as response:
                    response_data = await response.json()
                    return APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        message=response_data.get('message', ''),
                        status_code=response.status,
                        metadata=dict(response.headers)
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
                error_msg = str(e)
                if attempt < self.config['max_retries']:
                    # Reduce verbosity of retry messages - only log every 3rd attempt
                    if (attempt + 1) % 3 == 0:
                        logger.warning(f"Request failed: {error_msg}, retrying ({attempt + 1}/{self.config['max_retries'] + 1})")
                    await asyncio.sleep(self.config['retry_delay'] * (2 ** attempt))
                else:
                    return APIResponse(
                        success=False,
                        data=None,
                        message=error_msg,
                        status_code=500,
                        metadata={}
                    )

        # This should never be reached, but just in case
        return APIResponse(
            success=False,
            data=None,
            message="Request failed after all retries",
            status_code=500,
            metadata={}
        )

    async def _ensure_session(self):
        """Ensure we have a valid session, create one if needed"""
        if not self.config['session_id']:
            logger.info("No session ID, creating new session")
            # First ensure we have an HTTP session available
            if not await self._ensure_http_session():
                logger.error("Cannot create session - HTTP session not available")
                return False

            session_response = await self.create_session()
            if not session_response.success:
                logger.error(f"Failed to create session: {session_response.message}")
                return False
            logger.info(f"Created new session: {self.config['session_id']}")
        return True

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
            session_id = response.data['sessionId']
            if session_id and session_id.strip():  # Check if not empty or whitespace-only
                self.config['session_id'] = session_id
            else:
                # Server returned empty or invalid session ID
                logger.error("Server returned empty or invalid session ID")
                return APIResponse(
                    success=False,
                    data=None,
                    message="Server returned empty or invalid session ID",
                    status_code=response.status_code,
                    metadata=response.metadata
                )
        return response

    async def destroy_session(self) -> APIResponse:
        if self.config['session_id']:
            response = await self.delete(f"/sessions/{self.config['session_id']}")
            if response.success:
                self.config['session_id'] = ''
            return response
        return APIResponse(success=True, data=None, message="No active session", status_code=200, metadata={})

    # Chat API
    async def send_chat_message(self, message: str, options: Optional[Dict[str, Any]] = None) -> APIResponse:
        data = {
            'type': 'chat',
            'message': message,
            'messageType': "text",
            **(options or {})
        }
        return await self.post('/chat/message', data)

    async def get_chat_history(self, limit: int = 50) -> APIResponse:
        return await self.get(f'/chat/history?limit={limit}')

    async def clear_chat_history(self) -> APIResponse:
        return await self.post('/chat/clear')

    # Vision API
    async def analyze_image(self, image_path: str, analysis_types: Optional[List[str]] = None) -> APIResponse:
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

    async def process_video(self, video_path: str, operations: Optional[List[str]] = None) -> APIResponse:
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

    async def process_audio(self, audio_path: str, effects: Optional[List[str]] = None) -> APIResponse:
        return await self.post('/audio/process', {'audioPath': audio_path, 'effects': effects or []})

    # System API
    async def get_system_metrics(self) -> APIResponse:
        return await self.get('/system/metrics')

    async def get_running_processes(self) -> APIResponse:
        return await self.get('/system/processes')

    async def start_application(self, app_name: str, args: Optional[List[str]] = None) -> APIResponse:
        return await self.post('/system/start', {'appName': app_name, 'args': args or []})

    async def stop_application(self, app_name: str) -> APIResponse:
        return await self.post('/system/stop', {'appName': app_name})

    async def capture_screen(self, output_path: Optional[str] = None) -> APIResponse:
        return await self.post('/system/capture', {'outputPath': output_path})

    async def automate_task(self, task_type: str, parameters: Optional[Dict[str, Any]] = None) -> APIResponse:
        return await self.post('/system/automate', {'taskType': task_type, 'parameters': parameters or {}})

    # Web API
    async def search_web(self, query: str, max_results: int = 10) -> APIResponse:
        return await self.post('/web/search', {'query': query, 'maxResults': max_results})

    # Backward compatibility alias
    async def web_search(self, query: str, max_results: int = 10) -> APIResponse:
        """Alias for search_web for backward compatibility"""
        return await self.search_web(query, max_results)

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

    async def matrix_operation(self, operation: str, matrix_name: str, parameters: Optional[Dict[str, Any]] = None) -> APIResponse:
        return await self.post('/math/matrix', {'operation': operation, 'matrixName': matrix_name, 'parameters': parameters or {}})

    # Multi-modal Processing
    async def analyze_multimodal(self, content_type: str, content: str, analysis_types: Optional[List[str]] = None) -> APIResponse:
        return await self.post('/multimodal/analyze', {
            'contentType': content_type,
            'content': content,
            'analysisTypes': analysis_types or []
        })

    async def process_with_ai(self, content: str, processing_type: str = 'general') -> APIResponse:
        return await self.post('/ai/process', {'content': content, 'processingType': processing_type})

    # Real-time Processing
    async def start_stream(self, stream_type: str, parameters: Optional[Dict[str, Any]] = None) -> APIResponse:
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
            # Note: This method should not consume the WebSocket connection directly
            # as it's already being consumed by _handle_websocket_messages
            # Instead, rely on message handlers registered for stream_data
            logger.info(f"Stream handler for {stream_type} started - waiting for messages via WebSocket")
            # Keep the task alive until cancelled
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info(f"Stream {stream_type} cancelled")
        except Exception as e:
            logger.error(f"Stream {stream_type} error: {e}")

    # File Operations
    async def upload_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> APIResponse:
        if not self.session:
            await self.connect()

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return APIResponse(success=False, data=None, message="File not found", status_code=404, metadata={})

        async with aiohttp.ClientSession() as temp_session:
            with open(file_path_obj, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=file_path_obj.name)
                if metadata:
                    data.add_field('metadata', json.dumps(metadata))

                url = urljoin(self.config['base_url'], '/files/upload')
                headers = {'Authorization': f'Bearer {self.config["api_key"]}'}

                async with temp_session.post(url, data=data, headers=headers) as response:
                    response_data = await response.json()
                    return APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        message=response_data.get('message', ''),
                        status_code=response.status,
                        metadata=dict(response.headers)
                    )

    async def download_file(self, file_id: str, download_path: str) -> APIResponse:
        async with aiohttp.ClientSession() as temp_session:
            url = urljoin(self.config['base_url'], f'/files/download/{file_id}')
            headers = {'Authorization': f'Bearer {self.config["api_key"]}'}

            async with temp_session.get(url, headers=headers) as response:
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
    async def health_check(self) -> APIResponse:
        try:
            # Ensure we have a valid session before health check
            if not await self._ensure_session():
                return APIResponse(
                    success=False,
                    data=None,
                    message="Failed to establish session",
                    status_code=503,
                    metadata={}
                )

            response = await self.get('/health')
            return response
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message=str(e),
                status_code=500,
                metadata={}
            )

    async def get_system_status(self) -> APIResponse:
        return await self.get('/status')

    async def get_module_status(self, module_name: Optional[str] = None) -> APIResponse:
        endpoint = f'/status/modules{ f"/{module_name}" if module_name else ""}'
        return await self.get(endpoint)

    # Prediction API
    async def run_prediction(self, model_name: str, input_data: Any, input_type: str,
                            contains_header: bool = False, contains_text: bool = False) -> APIResponse:
        """Run prediction with specified model and input data"""
        data = {
            'modelName': model_name,
            'inputData': input_data,
            'inputType': input_type,
            'containsHeader': contains_header,
            'containsText': contains_text
        }
        return await self.post('/prediction/run', data)

    # Performance Testing API
    async def run_performance_test(self, config: Dict[str, Any]) -> APIResponse:
        """Run performance benchmarking test"""
        return await self.post('/performance/test', config)

    # Training API
    async def start_training(self, config: Dict[str, Any]) -> APIResponse:
        """Start neural network training with specified configuration"""
        return await self.post('/training/start', config)

    async def initialize_neural_network(self, config: Dict[str, Any]) -> APIResponse:
        """Initialize neural network with specified configuration"""
        return await self.post('/neural/initialize', config)

    async def train_neural_network(self, config: Dict[str, Any]) -> APIResponse:
        """Train neural network with specified configuration"""
        return await self.post('/neural/train', config)

    async def stop_training(self) -> APIResponse:
        """Stop current training session"""
        return await self.post('/training/stop')

    async def get_training_status(self) -> APIResponse:
        """Get current training session status"""
        return await self.get('/training/status')

    async def load_model(self, model_identifier: str, model_type: Optional[str] = None) -> APIResponse:
        """Load a model by dataset ID or name"""
        data = {
            'modelIdentifier': model_identifier,
            'modelType': model_type
        }
        return await self.post('/models/load', data)

    async def load_dataset(self, filename: str, dataset_name: str, output_size: int = 1, has_header: bool = True,
                          contains_text: bool = False, min_range: float = -1.0, max_range: float = 1.0) -> APIResponse:
        """Load dataset for training with normalization parameters"""
        data = {
            'filename': filename,
            'dataset_name': dataset_name,
            'output_size': output_size,
            'has_header': has_header,
            'contains_text': contains_text,
            'min_range': min_range,
            'max_range': max_range
        }
        return await self.post('/training/load_dataset', data)

    # Neural Network API
    async def get_neural_topology(self) -> APIResponse:
        return await self.get('/neural/topology')

    async def get_neural_activity(self) -> APIResponse:
        return await self.get('/neural/activity')

    # Utility Methods
    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            health_response = await self.health_check()
            if health_response.success and health_response.data and health_response.data.get('status') == 'healthy':
                return True
            await asyncio.sleep(1.0)
        return False

    def is_connected(self) -> bool:
        return self._is_connected and self.ws_connection is not None

    async def _monitor_connection(self):
        """Monitor WebSocket connection and handle reconnections with exponential backoff"""
        try:
            logger.info("WebSocket connection monitor started")
            while True:
                # Check shutdown flag
                if self._shutdown:
                    logger.info("Connection monitor stopping due to shutdown")
                    break

                await asyncio.sleep(10.0)  # Check every 10 seconds

                async with self._state_lock:
                    connection_lost = not self._is_connected or self.ws_connection is None

                if connection_lost:
                    if self._shutdown:
                        break
                    if self._reconnect_attempts >= self._max_reconnect_attempts:
                        logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached, giving up")
                        break

                    logger.info(f"Connection lost, attempting to reconnect (attempt {self._reconnect_attempts + 1}/{self._max_reconnect_attempts})...")
                    try:
                        await self._connect_websocket()
                        async with self._state_lock:
                            if self._is_connected and not self._shutdown:
                                logger.info("Successfully reconnected")
                                # Preserve session state during reconnection - don't reset session_id
                                self._reconnect_attempts = 0  # Reset counter on success
                            elif not self._shutdown:
                                logger.warning("Reconnection failed")
                                self._reconnect_attempts += 1
                                # Exponential backoff with jitter
                                backoff_time = min(self._reconnect_backoff_base * (2 ** self._reconnect_attempts), self._reconnect_backoff_max)
                                logger.info(f"Waiting {backoff_time:.1f} seconds before next reconnection attempt")
                                await asyncio.sleep(backoff_time)
                    except Exception as e:
                        logger.error(f"Reconnection attempt failed: {e}")
                        if not self._shutdown:
                            self._reconnect_attempts += 1
                            # Exponential backoff with jitter
                            backoff_time = min(self._reconnect_backoff_base * (2 ** self._reconnect_attempts), self._reconnect_backoff_max)
                            logger.info(f"Waiting {backoff_time:.1f} seconds before next reconnection attempt")
                            await asyncio.sleep(backoff_time)
                else:
                    # Connection is active, check if it's still healthy
                    try:
                        logger.debug("Performing connection health check (ping)")
                        # Send a ping to check connection health
                        await self.ws_connection.ping()
                        logger.debug("Connection health check passed")
                    except Exception as e:
                        logger.warning(f"Connection health check failed: {e}")
                        async with self._state_lock:
                            self._is_connected = False
                            if self.ws_connection:
                                try:
                                    await self.ws_connection.close()
                                except Exception:
                                    pass
                                self.ws_connection = None
                            if self._ws_task and not self._ws_task.done():
                                self._ws_task.cancel()
                                try:
                                    await asyncio.wait_for(self._ws_task, timeout=1.0)
                                except (asyncio.TimeoutError, asyncio.CancelledError):
                                    pass
                                self._ws_task = None
        except asyncio.CancelledError:
            logger.info("Connection monitor cancelled")
            raise  # Re-raise to properly cancel the task
        except Exception as e:
            logger.error(f"Connection monitor error: {e}")
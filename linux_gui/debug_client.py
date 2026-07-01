#!/usr/bin/env python3
"""
CoreAI3D Client Debug Script
Standalone script to test CoreAI3D client connection logic without Qt GUI.
Provides detailed logging to identify connection failures.
"""

import logging
import json
import sys
import time
import threading
import websocket
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import CoreAI3DClient
sys.path.insert(0, str(Path(__file__).parent))

from coreai3d_client import CoreAI3DClient

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_client.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

class DebugClientTester:
    """Debug tester for CoreAI3D client connections"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / 'config.json'
        self.client = None
        self.config = {}
        self.progress_lock = threading.Lock()
        self.start_time = None
        self.current_phase = ""
        self.total_phases = 0
        self.completed_phases = 0

    def display_progress(self, phase: str, completed: int, total: int):
        """Display progress bar with elapsed time and estimated remaining time"""
        with self.progress_lock:
            if self.start_time is None:
                self.start_time = time.time()

            elapsed = time.time() - self.start_time
            progress = completed / total if total > 0 else 0
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Estimate remaining time
            if completed > 0:
                avg_time_per_phase = elapsed / completed
                remaining_phases = total - completed
                estimated_remaining = avg_time_per_phase * remaining_phases
            else:
                estimated_remaining = 0

            # Format time display
            elapsed_str = f"{elapsed:.1f}s"
            remaining_str = f"{estimated_remaining:.1f}s" if estimated_remaining > 0 else "N/A"

            # Clear previous line and print new progress
            print(f"\r[{bar}] {completed}/{total} phases | {phase} | Elapsed: {elapsed_str} | Remaining: {remaining_str}", end='', flush=True)

            if completed >= total:
                print()  # New line when complete

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded config from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")

        # Default configuration
        return {
            'base_url': 'http://localhost:8080/api/v1',
            'ws_url': 'ws://localhost:8081',
            'api_key': '',
            'session_id': '',
            'timeout': 30.0,
            'max_retries': 3,
            'retry_delay': 1.0,
            'max_concurrent': 10,
            'debug': True
        }

    def test_connection_initialization(self):
        """Test client initialization and basic setup"""
        logger.info("=== Testing Client Initialization ===")

        try:
            self.config = self.load_config()
            logger.info(f"Client config: {json.dumps(self.config, indent=2)}")

            # Initialize client
            logger.info("Initializing CoreAI3DClient...")
            self.client = CoreAI3DClient(self.config)
            logger.info("Client initialized successfully")

            # Check initial state
            logger.info(f"Session ID: {self.client.config.get('session_id', 'None')}")
            logger.info(f"API Key configured: {'Yes' if self.client.config.get('api_key') else 'No'}")

            return True

        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def test_websocket_connection(self):
        """Test WebSocket connection establishment with synchronous websocket library"""
        logger.info("=== Testing WebSocket Connection ===")

        # Initialize client first if not done
        if not self.client:
            print("Initializing client...")
            success = self.test_connection_initialization()
            if not success:
                logger.error("Failed to initialize client for WebSocket test")
                return False

        max_retries = self.client.config.get('max_retries', 3)
        retry_delay = self.client.config.get('retry_delay', 1.0)

        def on_message(ws, message):
            print(f"Received: {message}")
            logger.info(f"WebSocket message received: {message}")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"Connection closed: {close_status_code}, {close_msg}")
            logger.info(f"WebSocket connection closed: {close_status_code}, {close_msg}")

        def on_open(ws):
            print("WebSocket connection opened successfully")
            logger.info("WebSocket connection opened successfully")
            # Send a test message
            ws.send(json.dumps({
                'type': 'ping',
                'timestamp': time.time()
            }))

        for attempt in range(max_retries + 1):
            try:
                print(f"Attempting WebSocket connection (attempt {attempt + 1}/{max_retries + 1})...")
                logger.info(f"Attempting WebSocket connection (attempt {attempt + 1}/{max_retries + 1})...")
                start_time = time.time()

                # Create WebSocket connection
                ws_url = self.client.config['ws_url']
                print(f"Connecting to: {ws_url}")
                logger.info(f"Connecting to WebSocket URL: {ws_url}")

                ws = websocket.WebSocketApp(ws_url,
                                          on_message=on_message,
                                          on_error=on_error,
                                          on_close=on_close)
                ws.on_open = on_open

                # Run the WebSocket connection in a separate thread
                ws_thread = threading.Thread(target=ws.run_forever)
                ws_thread.daemon = True
                ws_thread.start()

                # Wait for connection to establish
                connection_timeout = 10.0
                connected = False
                for _ in range(int(connection_timeout * 10)):
                    if ws.sock and ws.sock.connected:
                        connected = True
                        break
                    time.sleep(0.1)

                connection_time = time.time() - start_time

                if connected:
                    print(f"Connection established in {connection_time:.2f} seconds")
                    logger.info(f"WebSocket connection established in {connection_time:.2f} seconds")

                    # Test connection health
                    try:
                        print("Testing connection health...")
                        ws.send(json.dumps({'type': 'ping'}))
                        print("Connection health check passed")
                        logger.info("WebSocket connection health check passed")

                        # Close connection after successful test
                        ws.close()
                        ws_thread.join(timeout=2.0)
                        return True

                    except Exception as e:
                        print(f"Connection health check failed: {e}")
                        logger.warning(f"WebSocket health check failed: {e}")
                        ws.close()
                        ws_thread.join(timeout=2.0)
                        return False
                else:
                    print(f"Connection attempt {attempt + 1} timed out after {connection_timeout} seconds")
                    logger.warning(f"WebSocket connection timeout (attempt {attempt + 1}/{max_retries + 1})")
                    ws.close()
                    ws_thread.join(timeout=2.0)

                    if attempt < max_retries:
                        delay = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        print("Max retries exceeded due to timeouts")
                        return False

            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                logger.error(f"WebSocket connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries exceeded due to errors")
                    logger.error(f"WebSocket connection test failed after {max_retries + 1} attempts")
                    logger.error(f"Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return False

        return False

    def test_http_connection(self):
        """Test HTTP connection and basic API calls"""
        logger.info("=== Testing HTTP Connection ===")

        if not self.client:
            logger.error("Client not initialized")
            return False

        try:
            # Test health check using requests library
            logger.info("Performing health check...")
            base_url = self.client.config['base_url']
            health_url = f"{base_url}/health"

            headers = {
                'Authorization': f'Bearer {self.client.config["api_key"]}',
                'X-Session-ID': self.client.config['session_id'],
                'Content-Type': 'application/json'
            }

            response = requests.get(health_url, headers=headers, timeout=30.0)
            response_data = response.json()

            logger.info(f"Health check response: success={response.status_code < 400}, status={response.status_code}")
            logger.info(f"Health check message: {response_data.get('message', '')}")
            logger.info(f"Health check data: {response_data}")

            if response.status_code < 400:
                logger.info("HTTP connection and health check successful")
                return True
            else:
                logger.error("Health check failed")
                return False

        except Exception as e:
            logger.error(f"HTTP connection test failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def test_session_management(self):
        """Test session creation and management"""
        logger.info("=== Testing Session Management ===")

        if not self.client:
            logger.error("Client not initialized")
            return False

        try:
            # Check if we have a session
            current_session = self.client.config.get('session_id', '')
            logger.info(f"Current session ID: '{current_session}'")

            if not current_session:
                logger.info("No session ID, creating new session...")
                # Create session using requests
                base_url = self.client.config['base_url']
                session_url = f"{base_url}/sessions"

                headers = {
                    'Authorization': f'Bearer {self.client.config["api_key"]}',
                    'Content-Type': 'application/json'
                }

                response = requests.post(session_url, headers=headers, json={'clientId': ''}, timeout=30.0)
                response_data = response.json()

                logger.info(f"Session creation response: success={response.status_code < 400}, status={response.status_code}")
                logger.info(f"Session creation message: {response_data.get('message', '')}")
                logger.info(f"Session creation data: {response_data}")

                # Check for sessionId in the data field
                if response.status_code < 400 and response_data.get('data', {}).get('sessionId'):
                    self.client.config['session_id'] = response_data['data']['sessionId']
                    new_session = self.client.config.get('session_id', '')
                    logger.info(f"New session ID: '{new_session}'")
                    return True
                else:
                    logger.error("Session creation failed")
                    return False
            else:
                logger.info("Session already exists")
                return True

        except Exception as e:
            logger.error(f"Session management test failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def test_basic_api_calls(self):
        """Test basic API functionality"""
        logger.info("=== Testing Basic API Calls ===")

        if not self.client:
            logger.error("Client not initialized")
            return False

        try:
            # Test system status using requests
            logger.info("Testing system status API...")
            base_url = self.client.config['base_url']
            status_url = f"{base_url}/status"

            headers = {
                'Authorization': f'Bearer {self.client.config["api_key"]}',
                'X-Session-ID': self.client.config['session_id'],
                'Content-Type': 'application/json'
            }

            response = requests.get(status_url, headers=headers, timeout=30.0)
            response_data = response.json()

            logger.info(f"System status response: success={response.status_code < 400}, status={response.status_code}")
            logger.info(f"System status message: {response_data.get('message', '')}")
            logger.info(f"System status data: {response_data}")

            # Test module status
            logger.info("Testing module status API...")
            module_url = f"{base_url}/status/modules"

            response2 = requests.get(module_url, headers=headers, timeout=30.0)
            response2_data = response2.json()

            logger.info(f"Module status response: success={response2.status_code < 400}, status={response2.status_code}")
            logger.info(f"Module status message: {response2_data.get('message', '')}")
            logger.info(f"Module status data: {response2_data}")

            return (response.status_code < 400) or (response2.status_code < 400)

        except Exception as e:
            logger.error(f"Basic API calls test failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def run_full_test_suite(self):
        """Run complete test suite with progress tracking"""
        logger.info("=== Starting CoreAI3D Client Debug Test Suite ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {Path.cwd()}")
        logger.info(f"Config path: {self.config_path}")

        # Initialize progress tracking
        self.start_time = time.time()
        test_phases = [
            ("Client Initialization", self.test_connection_initialization),
            ("WebSocket Connection", self.test_websocket_connection),
            ("HTTP Connection", self.test_http_connection),
            ("Session Management", self.test_session_management),
            ("Basic API Calls", self.test_basic_api_calls)
        ]
        self.total_phases = len(test_phases)
        self.completed_phases = 0

        results = {}

        # Run tests with progress tracking
        for phase_name, test_func in test_phases:
            self.current_phase = phase_name
            self.display_progress(phase_name, self.completed_phases, self.total_phases)

            try:
                result = test_func()
                results[phase_name.lower().replace(' ', '_')] = result
                self.completed_phases += 1
                self.display_progress(phase_name, self.completed_phases, self.total_phases)
            except Exception as e:
                logger.error(f"Test {phase_name} failed with exception: {e}")
                results[phase_name.lower().replace(' ', '_')] = False
                self.completed_phases += 1
                self.display_progress(phase_name, self.completed_phases, self.total_phases)

        # Final progress update
        self.display_progress("Test Suite Complete", self.total_phases, self.total_phases)

        # Summary
        logger.info("=== Test Results Summary ===")
        for test_name, success in results.items():
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name.upper()}: {status}")

        total_passed = sum(results.values())
        total_tests = len(results)
        logger.info(f"Overall: {total_passed}/{total_tests} tests passed")

        # Cleanup
        if self.client:
            logger.info("Cleaning up client connection...")
            try:
                # Since we removed async, we can't call disconnect directly
                # The client will clean up when it goes out of scope
                logger.info("Client cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

        return results

def main():
    """Main debug function"""
    import argparse

    parser = argparse.ArgumentParser(description='Debug CoreAI3D Client Connections')
    parser.add_argument('--config', type=str, help='Path to config.json file')
    parser.add_argument('--test', type=str, choices=['init', 'ws', 'http', 'session', 'api', 'all'],
                        default='all', help='Specific test to run')

    args = parser.parse_args()

    tester = DebugClientTester(args.config)

    try:
        if args.test == 'all':
            tester.run_full_test_suite()
        else:
            # Run specific test - initialize client first for tests that need it
            if args.test == 'init':
                tester.test_connection_initialization()
            elif args.test in ['ws', 'http', 'session', 'api']:
                # Initialize client first for tests that require it
                init_success = tester.test_connection_initialization()
                if not init_success:
                    logger.error(f"Failed to initialize client for {args.test} test")
                    return

                if args.test == 'ws':
                    tester.test_websocket_connection()
                elif args.test == 'http':
                    tester.test_http_connection()
                elif args.test == 'session':
                    tester.test_session_management()
                elif args.test == 'api':
                    tester.test_basic_api_calls()

    except KeyboardInterrupt:
        logger.info("Debug script interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in debug script: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("Debug script completed")

if __name__ == "__main__":
    # Run the main function
    main()
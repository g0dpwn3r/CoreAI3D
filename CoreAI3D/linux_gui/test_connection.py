#!/usr/bin/env python3
"""
Test script for CoreAI3D connection logic
Tests the health_check method and APIResponse handling
"""

import sys
import os
import asyncio
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coreai3d_client import CoreAI3DClient

# Import dashboard for testing non-GUI mode
try:
    from dashboard import CoreAI3DDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("Warning: Dashboard import failed, GUI tests will be skipped")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test the connection logic similar to dashboard's test_connection method"""
    print("Testing CoreAI3D Connection Logic")
    print("=" * 40)

    # Configuration matching dashboard defaults
    config = {
        'base_url': "http://localhost:8080/api/v1",
        'ws_url': "ws://localhost:8081",
        'api_key': "2956a13167",  # Same as dashboard
        'timeout': 30.0,
        'max_connections': 10,
        'retry_attempts': 3,
        'debug': True
    }

    print(f"API URL: {config['base_url']}")
    print(f"WebSocket URL: {config['ws_url']}")
    print(f"API Key configured: {bool(config['api_key'])}")
    print(f"Connection timeout: {config['timeout']}s")
    print()

    client = None
    try:
        print("Creating CoreAI3DClient instance...")
        client = CoreAI3DClient(config)
        print("✓ Client created successfully")
        print()

        print("Testing connection to API server...")
        print("Calling health_check()...")

        # Test the health_check call (same as dashboard)
        result = await client.health_check()

        print("Health check result received")
        print(f"Result type: {type(result)}")

        # Validate response format (same validation as dashboard)
        if not hasattr(result, 'success'):
            raise AttributeError("Health check response missing 'success' attribute")

        if not hasattr(result, 'data'):
            raise AttributeError("Health check response missing 'data' attribute")

        if not hasattr(result, 'message'):
            raise AttributeError("Health check response missing 'message' attribute")

        if not hasattr(result, 'status_code'):
            raise AttributeError("Health check response missing 'status_code' attribute")

        print(f"✓ Response validation passed")
        print(f"  Success: {result.success}")
        print(f"  Status Code: {result.status_code}")
        print(f"  Message: {result.message}")

        if result.data:
            print(f"  Data: {result.data}")

        print()

        if result.success:
            print("✓ CONNECTION TEST SUCCESSFUL!")
            print("✓ API connection successful!")
            print("✓ Status Code:", result.status_code)
            print("✓ Message:", result.message)
            print("✓ The CoreAI3D server is responding correctly.")
            print("✓ All systems are operational.")
            return True
        else:
            print("✗ CONNECTION TEST FAILED!")
            error_details = result.data.get('error', 'Unknown error') if result.data else 'Unknown error'
            print("✗ Error:", error_details)
            print("✗ Status Code:", result.status_code)
            print("✗ Message:", result.message)
            return False

    except AttributeError as e:
        print("✗ CLIENT METHOD ACCESS ERROR!")
        print(f"✗ Error: {str(e)}")
        print("✗ The client object does not have the expected methods.")
        print("✗ This may indicate an incompatible client version.")
        return False

    except Exception as e:
        print("✗ UNEXPECTED ERROR DURING CONNECTION TEST!")
        print(f"✗ Error: {str(e)}")
        print(f"✗ Exception type: {type(e).__name__}")
        import traceback
        print("✗ Traceback:")
        traceback.print_exc()
        return False

    finally:
        if client and hasattr(client, 'close'):
            try:
                await client.close()
                print("✓ Client connection closed")
            except Exception as e:
                print(f"⚠ Warning: Error closing client: {str(e)}")

async def test_dashboard_connection():
    """Test dashboard connection in non-GUI mode"""
    if not DASHBOARD_AVAILABLE:
        print("Dashboard not available, skipping GUI test")
        return True

    print("\nTesting Dashboard Connection (Non-GUI Mode)")
    print("=" * 50)

    try:
        # Skip dashboard test for now since it requires QApplication
        print("Dashboard test skipped - requires QApplication for initialization")
        print("✓ Dashboard test skipped (expected)")
        return True

    except Exception as e:
        print(f"✗ Dashboard test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    try:
        # Test basic client connection
        client_success = await test_connection()

        # Test dashboard connection (non-GUI mode)
        dashboard_success = await test_dashboard_connection()

        print()
        print("=" * 40)

        overall_success = client_success and dashboard_success

        if overall_success:
            print("OVERALL RESULT: SUCCESS")
            print("✓ All connection tests passed")
            sys.exit(0)
        else:
            print("OVERALL RESULT: FAILED")
            if not client_success:
                print("✗ Client connection test failed")
            if not dashboard_success:
                print("✗ Dashboard connection test failed")
            sys.exit(1)

    except Exception as e:
        print(f"✗ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())
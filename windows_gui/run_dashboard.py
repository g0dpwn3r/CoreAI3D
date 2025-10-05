#!/usr/bin/env python3
"""
CoreAI3D Dashboard Startup Script
Simple script to launch the Windows GUI Dashboard with proper error handling.
"""

import sys
import os
import logging
from pathlib import Path

def setup_environment():
    """Setup environment for the dashboard"""
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

        # Add python directory to Python path (for coreai3d_client)
        python_dir = current_dir.parent / 'python'
        if str(python_dir) not in sys.path:
            sys.path.insert(0, str(python_dir))

        # Create logs directory
        logs_dir = current_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / 'dashboard_startup.log'),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info("Environment setup completed")
        logger.info(f"Python path includes: {sys.path[:3]}...")  # Log first 3 paths

        return True

    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        required_modules = [
            'PyQt6',
            'psutil',
            'aiohttp',
            'websockets'
        ]

        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            print("Missing required modules:")
            for module in missing_modules:
                print(f"  - {module}")
            print("\nPlease install them using:")
            print("pip install -r requirements.txt")
            return False

        print("All dependencies are installed [OK]")
        return True

    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False

def main():
    """Main startup function"""
    print("CoreAI3D Dashboard Startup")
    print("=" * 30)

    # Setup environment
    if not setup_environment():
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    try:
        # Import and run dashboard
        from dashboard import main as dashboard_main

        print("Starting CoreAI3D Dashboard...")
        print("Press Ctrl+C to exit")

        dashboard_main()

    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)

    except Exception as e:
        print(f"Error starting dashboard: {e}")
        logging.error(f"Dashboard startup error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
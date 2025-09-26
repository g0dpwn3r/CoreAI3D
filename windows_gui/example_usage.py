#!/usr/bin/env python3
"""
CoreAI3D Dashboard Example Usage
Demonstrates how to use the dashboard programmatically and integrate it with other systems.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from coreai3d_client import CoreAI3DClient
from automation_helper import AutomationHelper
from dashboard import DashboardConfig, TrainingDataManager, SandboxManager

class DashboardExample:
    """Example class demonstrating dashboard usage"""

    def __init__(self):
        self.config = DashboardConfig()
        self.client = None
        self.training_manager = TrainingDataManager(self.config)
        self.sandbox_manager = SandboxManager(self.config)

    async def setup_client(self):
        """Setup CoreAI3D client"""
        try:
            self.client = CoreAI3DClient({
                'base_url': self.config.api_url,
                'ws_url': self.config.ws_url,
                'api_key': self.config.api_key,
                'timeout': 30.0,
                'max_connections': 10,
                'retry_attempts': 3
            })

            print("✓ Client initialized successfully")
            return True

        except Exception as e:
            print(f"✗ Failed to initialize client: {e}")
            return False

    async def example_health_check(self):
        """Example: Perform health check"""
        print("\n=== Health Check Example ===")

        if not self.client:
            await self.setup_client()

        try:
            result = await self.client.health_check()
            if result.success:
                print("✓ Health check passed")
                print(f"  Response time: {result.response_time:.2f}s")
                print(f"  Server status: {result.data.get('status', 'unknown')}")
            else:
                print("✗ Health check failed")
                print(f"  Error: {result.message}")

        except Exception as e:
            print(f"✗ Health check error: {e}")

    async def example_system_monitoring(self):
        """Example: Monitor system metrics"""
        print("\n=== System Monitoring Example ===")

        if not self.client:
            await self.setup_client()

        try:
            result = await self.client.get_system_metrics()
            if result.success:
                print("✓ System metrics retrieved")
                metrics = result.data

                print(f"  CPU Usage: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"  Memory Usage: {metrics.get('memory_percent', 0):.1f}%")
                print(f"  Disk Usage: {metrics.get('disk_percent', 0):.1f}%")
                print(f"  Active Processes: {metrics.get('process_count', 0)}")
            else:
                print("✗ Failed to get system metrics")
                print(f"  Error: {result.message}")

        except Exception as e:
            print(f"✗ System monitoring error: {e}")

    async def example_training_data_management(self):
        """Example: Manage training data"""
        print("\n=== Training Data Management Example ===")

        try:
            # Create a sample dataset
            dataset_name = f"example_dataset_{int(datetime.now().timestamp())}"
            success = self.training_manager.create_dataset(
                name=dataset_name,
                description="Example dataset created by automation script",
                data_type="mixed"
            )

            if success:
                print(f"✓ Created dataset: {dataset_name}")

                # Add a sample file (create a dummy file for demo)
                sample_file = Path("sample_data") / "example.txt"
                sample_file.parent.mkdir(exist_ok=True)
                sample_file.write_text("This is a sample training data file.")

                success = self.training_manager.add_file_to_dataset(
                    dataset_name, str(sample_file)
                )

                if success:
                    print(f"✓ Added sample file to dataset: {dataset_name}")
                else:
                    print(f"✗ Failed to add file to dataset: {dataset_name}")

            else:
                print(f"✗ Failed to create dataset: {dataset_name}")

        except Exception as e:
            print(f"✗ Training data management error: {e}")

    async def example_sandbox_testing(self):
        """Example: Test Windows Sandbox functionality"""
        print("\n=== Windows Sandbox Testing Example ===")

        try:
            # Check if sandbox is available
            if not self._is_sandbox_available():
                print("⚠ Windows Sandbox is not available on this system")
                print("  This feature requires:")
                print("  - Windows 10/11 Pro, Enterprise, or Education")
                print("  - Virtualization enabled in BIOS")
                print("  - Windows Sandbox feature enabled")
                return

            # Configure sandbox
            sandbox_config = {
                'host_folder': str(Path.home()),
                'startup_command': 'echo "Sandbox test completed successfully"',
                'networking': 'enable',
                'vgpu': 'enable'
            }

            print("Starting Windows Sandbox test...")

            # Start sandbox
            success = self.sandbox_manager.start_sandbox(sandbox_config)

            if success:
                print("✓ Windows Sandbox started successfully")

                # Wait a moment for sandbox to initialize
                await asyncio.sleep(5)

                # Stop sandbox
                success = self.sandbox_manager.stop_sandbox()
                if success:
                    print("✓ Windows Sandbox stopped successfully")
                else:
                    print("⚠ Failed to stop Windows Sandbox gracefully")

            else:
                print("✗ Failed to start Windows Sandbox")

        except Exception as e:
            print(f"✗ Sandbox testing error: {e}")

    def _is_sandbox_available(self) -> bool:
        """Check if Windows Sandbox is available"""
        try:
            # Check if Windows Sandbox executable exists
            import subprocess
            result = subprocess.run(
                ['where', 'WindowsSandbox.exe'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    async def example_vision_module(self):
        """Example: Use vision module"""
        print("\n=== Vision Module Example ===")

        if not self.client:
            await self.setup_client()

        try:
            # Create a sample image for testing
            sample_image = Path("sample_data") / "test_image.jpg"
            sample_image.parent.mkdir(exist_ok=True)

            # Create a simple test image (1x1 pixel red image)
            with open(sample_image, 'wb') as f:
                # Minimal JPEG header for a 1x1 red pixel
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\x00\x00\xff\xd9')

            result = await self.client.analyze_image(str(sample_image))

            if result.success:
                print("✓ Vision analysis completed")
                print(f"  Analysis: {result.data.get('description', 'No description')}")
                print(f"  Confidence: {result.data.get('confidence', 0):.2f}")
            else:
                print("✗ Vision analysis failed")
                print(f"  Error: {result.message}")

            # Clean up
            sample_image.unlink(missing_ok=True)

        except Exception as e:
            print(f"✗ Vision module error: {e}")

    async def example_web_search(self):
        """Example: Perform web search"""
        print("\n=== Web Search Example ===")

        if not self.client:
            await self.setup_client()

        try:
            query = "artificial intelligence latest developments"
            result = await self.client.web_search(query)

            if result.success:
                print("✓ Web search completed")
                print(f"  Query: {query}")
                print(f"  Results found: {len(result.data.get('results', []))}")

                # Show first result
                results = result.data.get('results', [])
                if results:
                    first_result = results[0]
                    print(f"  First result: {first_result.get('title', 'No title')}")
                    print(f"  URL: {first_result.get('url', 'No URL')}")
            else:
                print("✗ Web search failed")
                print(f"  Error: {result.message}")

        except Exception as e:
            print(f"✗ Web search error: {e}")

    async def run_all_examples(self):
        """Run all examples"""
        print("CoreAI3D Dashboard - Example Usage")
        print("=" * 40)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run examples
        await self.example_health_check()
        await self.example_system_monitoring()
        await self.example_training_data_management()
        await self.example_sandbox_testing()
        await self.example_vision_module()
        await self.example_web_search()

        print("\n" + "=" * 40)
        print("All examples completed!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.close()
            print("✓ Client connection closed")

async def main():
    """Main function"""
    example = DashboardExample()

    try:
        await example.run_all_examples()

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

    finally:
        await example.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
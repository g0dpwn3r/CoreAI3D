#!/usr/bin/env python3
"""
CoreAI3D Windows Operations Training Script
Trains AI to operate Windows systems safely within Docker containers
"""

import asyncio
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/windows_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WindowsOperationTrainer:
    """Trains AI to perform Windows operations safely"""

    def __init__(self):
        self.training_data = []
        self.sandbox_type = os.environ.get('SANDBOX_TYPE', 'windows')
        self.workspace = Path('C:\\workspace')
        self.training_dir = Path('C:\\training_data')

    async def initialize_training_environment(self):
        """Initialize the training environment"""
        logger.info(f"Initializing Windows training environment for {self.sandbox_type}")

        # Create necessary directories
        self.workspace.mkdir(exist_ok=True)
        self.training_dir.mkdir(exist_ok=True)

        # Set up training scenarios
        await self.setup_training_scenarios()

        logger.info("Training environment initialized")

    async def setup_training_scenarios(self):
        """Set up various Windows operation training scenarios"""
        scenarios = [
            {
                "name": "file_system_operations",
                "description": "Basic file system operations",
                "commands": [
                    "Get-ChildItem -Path C:\\workspace -Force",
                    "Get-Location",
                    "New-Item -ItemType Directory -Path 'C:\\workspace\\test_directory' -Force",
                    "New-Item -ItemType File -Path 'C:\\workspace\\test_directory\\test_file.txt' -Force",
                    "Copy-Item 'C:\\workspace\\test_directory\\test_file.txt' 'C:\\workspace\\test_directory\\backup.txt'",
                    "Move-Item 'C:\\workspace\\test_directory\\backup.txt' 'C:\\workspace\\test_directory\\renamed.txt'",
                    "Remove-Item 'C:\\workspace\\test_directory\\renamed.txt'",
                    "Remove-Item 'C:\\workspace\\test_directory' -Recurse"
                ]
            },
            {
                "name": "process_management",
                "description": "Process monitoring and management",
                "commands": [
                    "Get-Process | Select-Object -First 10",
                    "Get-Service | Where-Object {$_.Status -eq 'Running'} | Select-Object -First 10",
                    "Get-NetTCPConnection | Select-Object -First 10",
                    "Get-EventLog -LogName System -Newest 10"
                ]
            },
            {
                "name": "network_operations",
                "description": "Network configuration and testing",
                "commands": [
                    "Get-NetIPAddress",
                    "Test-NetConnection google.com -Count 3",
                    "Resolve-DnsName google.com",
                    "Get-NetRoute",
                    "Invoke-WebRequest -Uri 'https://google.com' -Method Head"
                ]
            },
            {
                "name": "package_management",
                "description": "Package installation and management",
                "commands": [
                    "choco list --local-only",
                    "choco search python",
                    "python --version"
                ]
            },
            {
                "name": "registry_operations",
                "description": "Windows registry operations",
                "commands": [
                    "Get-ChildItem -Path 'HKLM:\\SOFTWARE' -ErrorAction SilentlyContinue | Select-Object -First 5",
                    "Get-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'",
                    "Get-Help about_Registry"
                ]
            },
            {
                "name": "system_monitoring",
                "description": "System monitoring and diagnostics",
                "commands": [
                    "Get-ComputerInfo",
                    "Get-Disk",
                    "Get-Volume",
                    "Get-EventLog -LogName System -Newest 1",
                    "Get-WmiObject -Class Win32_OperatingSystem"
                ]
            }
        ]

        # Save scenarios to training data
        for scenario in scenarios:
            await self.save_training_scenario(scenario)

    async def save_training_scenario(self, scenario: Dict[str, Any]):
        """Save a training scenario to the training data"""
        scenario_file = self.training_dir / f"{scenario['name']}.json"

        scenario_data = {
            "scenario_name": scenario["name"],
            "description": scenario["description"],
            "sandbox_type": self.sandbox_type,
            "commands": scenario["commands"],
            "created_at": datetime.now().isoformat(),
            "difficulty": self._calculate_difficulty(scenario)
        }

        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)

        logger.info(f"Saved training scenario: {scenario['name']}")

    def _calculate_difficulty(self, scenario: Dict[str, Any]) -> str:
        """Calculate difficulty level of a scenario"""
        command_count = len(scenario["commands"])
        if command_count <= 3:
            return "beginner"
        elif command_count <= 6:
            return "intermediate"
        else:
            return "advanced"

    async def execute_training_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Execute a training scenario and record results"""
        scenario_file = self.training_dir / f"{scenario_name}.json"

        if not scenario_file.exists():
            return {"error": f"Scenario {scenario_name} not found"}

        with open(scenario_file, 'r') as f:
            scenario = json.load(f)

        logger.info(f"Executing training scenario: {scenario_name}")

        results = {
            "scenario_name": scenario_name,
            "start_time": datetime.now().isoformat(),
            "commands": [],
            "success_count": 0,
            "total_count": len(scenario["commands"])
        }

        for i, command in enumerate(scenario["commands"]):
            command_result = await self.execute_command(command, i + 1)
            results["commands"].append(command_result)

            if command_result["success"]:
                results["success_count"] += 1

        results["end_time"] = datetime.now().isoformat()
        results["success_rate"] = results["success_count"] / results["total_count"]

        # Save execution results
        results_file = self.training_dir / f"{scenario_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Scenario {scenario_name} completed. Success rate: {results['success_rate']:.2%}")
        return results

    async def execute_command(self, command: str, command_id: int) -> Dict[str, Any]:
        """Execute a single command and return results"""
        logger.info(f"Executing command {command_id}: {command}")

        try:
            start_time = time.time()

            # Execute PowerShell command
            process = await asyncio.create_subprocess_shell(
                f"powershell.exe -Command {command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
            )

            stdout, stderr = await process.communicate()

            end_time = time.time()
            execution_time = end_time - start_time

            result = {
                "command_id": command_id,
                "command": command,
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Command {command_id} completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error executing command {command_id}: {e}")
            return {
                "command_id": command_id,
                "command": command,
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Main training function"""
    trainer = WindowsOperationTrainer()

    try:
        await trainer.initialize_training_environment()
        logger.info("Windows operations training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
CoreAI3D Linux Operations Training Script
Trains AI to operate Linux systems safely within Docker containers
"""

import asyncio
import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/linux_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LinuxOperationTrainer:
    """Trains AI to perform Linux operations safely"""

    def __init__(self):
        self.training_data = []
        self.sandbox_type = os.environ.get('SANDBOX_TYPE', 'ubuntu')
        self.workspace = Path('/workspace')
        self.training_dir = Path('/training_data')

    async def initialize_training_environment(self):
        """Initialize the training environment"""
        logger.info(f"Initializing Linux training environment for {self.sandbox_type}")

        # Create necessary directories
        self.workspace.mkdir(exist_ok=True)
        self.training_dir.mkdir(exist_ok=True)

        # Set up training scenarios
        await self.setup_training_scenarios()

        logger.info("Training environment initialized")

    async def setup_training_scenarios(self):
        """Set up various Linux operation training scenarios"""
        scenarios = [
            {
                "name": "file_system_operations",
                "description": "Basic file system operations",
                "commands": [
                    "ls -la",
                    "pwd",
                    "mkdir test_directory",
                    "cd test_directory && touch test_file.txt",
                    "cp test_file.txt backup_file.txt",
                    "mv backup_file.txt renamed_file.txt",
                    "cat renamed_file.txt",
                    "rm renamed_file.txt && cd .. && rmdir test_directory"
                ]
            },
            {
                "name": "process_management",
                "description": "Process monitoring and management",
                "commands": [
                    "ps aux",
                    "top -n 1",
                    "htop",
                    "pgrep python",
                    "pkill -f python",
                    "jobs",
                    "bg",
                    "fg"
                ]
            },
            {
                "name": "network_operations",
                "description": "Network configuration and testing",
                "commands": [
                    "ip addr show",
                    "ip route show",
                    "ping -c 3 google.com",
                    "traceroute google.com",
                    "netstat -tuln",
                    "ss -tuln",
                    "curl -I https://google.com",
                    "wget --spider https://google.com"
                ]
            },
            {
                "name": "package_management",
                "description": "Package installation and management",
                "commands": [
                    "apt update" if self.sandbox_type == "ubuntu" else "yum update",
                    "apt list --installed" if self.sandbox_type == "ubuntu" else "yum list installed",
                    "apt search python" if self.sandbox_type == "ubuntu" else "yum search python",
                    "apt show python3" if self.sandbox_type == "ubuntu" else "yum info python3",
                    "which python3",
                    "python3 --version"
                ]
            },
            {
                "name": "text_processing",
                "description": "Text processing and manipulation",
                "commands": [
                    "echo 'Hello World' > hello.txt",
                    "cat hello.txt",
                    "grep 'Hello' hello.txt",
                    "sed 's/Hello/Hi/' hello.txt",
                    "awk '{print $1}' hello.txt",
                    "sort hello.txt",
                    "uniq hello.txt",
                    "wc -l hello.txt",
                    "rm hello.txt"
                ]
            },
            {
                "name": "permissions_security",
                "description": "File permissions and security",
                "commands": [
                    "touch secret.txt",
                    "chmod 600 secret.txt",
                    "ls -l secret.txt",
                    "chmod 755 secret.txt",
                    "chown sandbox:sandbox secret.txt",
                    "ls -l secret.txt",
                    "groups",
                    "whoami",
                    "id",
                    "rm secret.txt"
                ]
            },
            {
                "name": "system_monitoring",
                "description": "System monitoring and diagnostics",
                "commands": [
                    "df -h",
                    "du -sh",
                    "free -h",
                    "uptime",
                    "date",
                    "uname -a",
                    "lscpu",
                    "lsmem",
                    "vmstat 1 3",
                    "iostat 1 3"
                ]
            },
            {
                "name": "scripting_automation",
                "description": "Shell scripting and automation",
                "commands": [
                    "echo '#!/bin/bash\necho \"Hello from script\"' > hello.sh",
                    "chmod +x hello.sh",
                    "./hello.sh",
                    "bash -c 'for i in {1..5}; do echo $i; done'",
                    "find . -name '*.sh' -exec ls -l {} \\;",
                    "crontab -l",
                    "which cron",
                    "rm hello.sh"
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

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
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

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics during training"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "sandbox_type": self.sandbox_type,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": dict(psutil.disk_usage('/')._asdict()),
                "network": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
                "processes": len(psutil.pids()),
                "threads": sum(p.num_threads() for p in psutil.process_iter() if p.num_threads())
            }

            # Save metrics
            metrics_file = self.training_dir / "system_metrics.json"
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

            return metrics

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    async def run_comprehensive_training(self):
        """Run comprehensive Linux operations training"""
        logger.info("Starting comprehensive Linux operations training")

        # Get all available scenarios
        scenario_files = list(self.training_dir.glob("*_operations.json"))
        scenario_names = [f.stem.replace('_operations', '') for f in scenario_files]

        training_results = {
            "training_session_id": f"session_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "sandbox_type": self.sandbox_type,
            "scenarios": [],
            "overall_success_rate": 0
        }

        total_success = 0
        total_commands = 0

        for scenario_name in scenario_names:
            scenario_result = await self.execute_training_scenario(scenario_name)
            training_results["scenarios"].append(scenario_result)

            total_success += scenario_result["success_count"]
            total_commands += scenario_result["total_count"]

        training_results["end_time"] = datetime.now().isoformat()
        training_results["overall_success_rate"] = total_success / total_commands if total_commands > 0 else 0

        # Save comprehensive results
        results_file = self.training_dir / "comprehensive_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)

        logger.info(f"Training completed. Overall success rate: {training_results['overall_success_rate']:.2%}")
        return training_results

    async def generate_training_report(self) -> str:
        """Generate a comprehensive training report"""
        report = f"""
# CoreAI3D Linux Operations Training Report
## Sandbox Type: {self.sandbox_type}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Summary
"""

        # Load comprehensive results if available
        results_file = self.training_dir / "comprehensive_training_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)

            report += f"""
- Total scenarios executed: {len(results['scenarios'])}
- Overall success rate: {results['overall_success_rate']:.2%}
- Training duration: {results.get('end_time', 'Unknown')}

## Scenario Details
"""

            for scenario in results['scenarios']:
                report += f"""
### {scenario['scenario_name'].replace('_', ' ').title()}
- Commands executed: {scenario['total_count']}
- Successful commands: {scenario['success_count']}
- Success rate: {scenario['success_rate']:.2%}
"""

        # Add system metrics
        metrics = await self.collect_system_metrics()
        report += f"""
## System Performance
- CPU Usage: {metrics.get('cpu_percent', 'Unknown')}%
- Memory Usage: {metrics.get('memory', {}).get('percent', 'Unknown')}%
- Disk Usage: {metrics.get('disk', {}).get('percent', 'Unknown')}%
- Active Processes: {metrics.get('processes', 'Unknown')}
"""

        return report

async def main():
    """Main training function"""
    trainer = LinuxOperationTrainer()

    try:
        await trainer.initialize_training_environment()

        # Run comprehensive training
        results = await trainer.run_comprehensive_training()

        # Generate and save report
        report = await trainer.generate_training_report()
        report_file = Path("/training_data/training_report.md")
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info("Linux operations training completed successfully")
        logger.info(f"Report saved to: {report_file}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
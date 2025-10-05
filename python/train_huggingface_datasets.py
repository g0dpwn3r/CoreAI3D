#!/usr/bin/env python3
"""
Train CoreAI3D models on Hugging Face datasets
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from huggingface_dataset_manager import HuggingFaceDatasetManager
from coreai3d_client import CoreAI3DClient

class HuggingFaceTrainer:
    """Trainer for Hugging Face datasets using CoreAI3D"""

    def __init__(self, client: CoreAI3DClient, data_dir: str = "training_data"):
        self.client = client
        self.hf_manager = HuggingFaceDatasetManager(data_dir)
        self.logger = logging.getLogger(__name__)

    async def train_on_dataset(self, dataset_name: str,
                             model_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train CoreAI3D model on Hugging Face dataset"""

        # Download and convert dataset
        self.logger.info(f"Preparing dataset: {dataset_name}")
        dataset_path = self.hf_manager.download_dataset(dataset_name)
        coreai3d_path = self.hf_manager.convert_to_coreai3d_format(dataset_path)

        # Load dataset info
        dataset_info = self.hf_manager.get_dataset_info(coreai3d_path)

        # Prepare training configuration
        config = {
            'dataset_path': coreai3d_path,
            'dataset_info': dataset_info,
            'model_type': model_config.get('model_type', 'auto'),
            'epochs': model_config.get('epochs', 10),
            'batch_size': model_config.get('batch_size', 32),
            'learning_rate': model_config.get('learning_rate', 0.001),
        }

        # Start training
        self.logger.info(f"Starting training on {dataset_name}")
        result = await self.client.post('/training/start', {
            'config': config,
            'dataset_type': 'huggingface'
        })

        if result.success:
            training_id = result.data.get('training_id')
            self.logger.info(f"Training started with ID: {training_id}")

            # Monitor training
            return await self._monitor_training(training_id)
        else:
            raise Exception(f"Training failed to start: {result.data}")

    async def _monitor_training(self, training_id: str) -> Dict[str, Any]:
        """Monitor training progress"""
        while True:
            result = await self.client.get(f'/training/status/{training_id}')

            if result.success:
                status = result.data.get('status')
                progress = result.data.get('progress', 0)

                self.logger.info(f"Training progress: {progress}% - Status: {status}")

                if status == 'completed':
                    return result.data
                elif status == 'failed':
                    raise Exception(f"Training failed: {result.data.get('error')}")

            await asyncio.sleep(5)  # Check every 5 seconds

async def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train CoreAI3D on Hugging Face datasets')
    parser.add_argument('dataset', help='Hugging Face dataset name')
    parser.add_argument('--model-type', default='auto', help='Model type')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--api-url', default='http://localhost:8080/api/v1', help='CoreAI3D API URL')
    parser.add_argument('--api-key', help='API key')

    args = parser.parse_args()

    # Initialize client
    client = CoreAI3DClient({
        'base_url': args.api_url,
        'api_key': args.api_key
    })

    # Initialize trainer
    trainer = HuggingFaceTrainer(client)

    # Train model
    try:
        result = await trainer.train_on_dataset(
            args.dataset,
            {
                'model_type': args.model_type,
                'epochs': args.epochs,
                'batch_size': args.batch_size
            }
        )

        print("Training completed successfully!")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Training failed: {e}")
        return 1

    return 0

if __name__ == '__main__':
    asyncio.run(main())
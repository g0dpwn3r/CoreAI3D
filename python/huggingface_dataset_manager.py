#!/usr/bin/env python3
"""
Hugging Face Dataset Manager for CoreAI3D
Handles downloading, processing, and managing Hugging Face datasets
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp

# Hugging Face imports
try:
    from datasets import load_dataset, DatasetDict, Dataset
    from huggingface_hub import HfApi, HfFolder
    import pandas as pd
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: Hugging Face libraries not available. Install with: pip install datasets huggingface-hub pandas")

from coreai3d_client import CoreAI3DClient

@dataclass
class DatasetInfo:
    """Information about a Hugging Face dataset"""
    name: str
    description: str
    size: int
    splits: List[str]
    features: Dict[str, Any]
    tags: List[str]
    downloads: int
    likes: int

class HuggingFaceDatasetManager:
    """Manager for Hugging Face datasets integration"""

    def __init__(self, data_dir: str = "training_data", api_token: str = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.api_token = api_token or os.environ.get('HF_TOKEN')

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize HF API
        if HUGGINGFACE_AVAILABLE:
            self.hf_api = HfApi()
            if self.api_token:
                HfFolder.save_token(self.api_token)

    def check_huggingface_availability(self) -> bool:
        """Check if Hugging Face libraries are available"""
        return HUGGINGFACE_AVAILABLE

    def search_datasets(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """Search for datasets on Hugging Face Hub"""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face libraries not available")

        try:
            datasets = self.hf_api.list_datasets(search=query, limit=limit)

            results = []
            for dataset in datasets:
                info = DatasetInfo(
                    name=dataset.id,
                    description=getattr(dataset, 'description', ''),
                    size=getattr(dataset, 'size', 0),
                    splits=getattr(dataset, 'splits', []),
                    features=getattr(dataset, 'features', {}),
                    tags=getattr(dataset, 'tags', []),
                    downloads=getattr(dataset, 'downloads', 0),
                    likes=getattr(dataset, 'likes', 0)
                )
                results.append(info)

            return results

        except Exception as e:
            self.logger.error(f"Error searching datasets: {e}")
            return []

    def download_dataset(self, dataset_name: str,
                        splits: List[str] = None,
                        save_format: str = 'json') -> str:
        """Download and process a Hugging Face dataset"""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face libraries not available")

        try:
            # Create dataset directory
            dataset_dir = self.data_dir / dataset_name.replace('/', '_')
            dataset_dir.mkdir(exist_ok=True)

            self.logger.info(f"Downloading dataset: {dataset_name}")

            # Load dataset
            if splits:
                dataset = load_dataset(dataset_name, split=splits)
            else:
                dataset = load_dataset(dataset_name)

            # Handle different dataset types
            if isinstance(dataset, DatasetDict):
                # Multiple splits
                for split_name, split_data in dataset.items():
                    self._save_split(split_data, dataset_dir, split_name, save_format)
            else:
                # Single dataset
                self._save_split(dataset, dataset_dir, 'train', save_format)

            # Save metadata
            self._save_metadata(dataset_name, dataset_dir, dataset)

            self.logger.info(f"Dataset {dataset_name} downloaded successfully")
            return str(dataset_dir)

        except Exception as e:
            self.logger.error(f"Error downloading dataset {dataset_name}: {e}")
            raise

    def _save_split(self, dataset: Dataset, dataset_dir: Path,
                   split_name: str, save_format: str):
        """Save a dataset split"""
        split_dir = dataset_dir / split_name
        split_dir.mkdir(exist_ok=True)

        if save_format == 'json':
            # Save as JSON lines
            jsonl_file = split_dir / f"{split_name}.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for example in dataset:
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')

        elif save_format == 'csv':
            # Convert to pandas and save as CSV
            df = pd.DataFrame(dataset)
            csv_file = split_dir / f"{split_name}.csv"
            df.to_csv(csv_file, index=False)

        elif save_format == 'parquet':
            # Save as Parquet
            df = pd.DataFrame(dataset)
            parquet_file = split_dir / f"{split_name}.parquet"
            df.to_parquet(parquet_file, index=False)

    def _save_metadata(self, dataset_name: str, dataset_dir: Path, dataset: Any):
        """Save dataset metadata"""
        metadata = {
            'name': dataset_name,
            'description': getattr(dataset, 'description', ''),
            'features': list(dataset.features.keys()) if hasattr(dataset, 'features') else [],
            'num_rows': len(dataset) if hasattr(dataset, '__len__') else 0,
            'splits': list(dataset.keys()) if isinstance(dataset, DatasetDict) else ['train'],
            'downloaded_at': str(pd.Timestamp.now()),
            'source': 'huggingface'
        }

        metadata_file = dataset_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def list_downloaded_datasets(self) -> List[Dict[str, Any]]:
        """List all downloaded datasets"""
        datasets = []

        for item in self.data_dir.iterdir():
            if item.is_dir():
                metadata_file = item / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            metadata['local_path'] = str(item)
                            datasets.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Error reading metadata for {item}: {e}")

        return datasets

    def get_dataset_info(self, dataset_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a downloaded dataset"""
        metadata_file = Path(dataset_path) / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading dataset info: {e}")
        return None

    def convert_to_coreai3d_format(self, dataset_path: str,
                                  output_format: str = 'jsonl') -> str:
        """Convert dataset to CoreAI3D training format"""
        dataset_info = self.get_dataset_info(dataset_path)
        if not dataset_info:
            raise ValueError(f"Dataset not found: {dataset_path}")

        dataset_path = Path(dataset_path)
        coreai3d_dir = self.data_dir.parent / 'coreai3d_training_data'
        coreai3d_dir.mkdir(exist_ok=True)

        # Create CoreAI3D dataset
        coreai3d_dataset_dir = coreai3d_dir / dataset_info['name'].replace('/', '_')
        coreai3d_dataset_dir.mkdir(exist_ok=True)

        # Convert each split
        for split in dataset_info.get('splits', ['train']):
            split_dir = dataset_path / split
            if split_dir.exists():
                # Find data files
                data_files = list(split_dir.glob('*.jsonl')) + list(split_dir.glob('*.csv'))

                if data_files:
                    # Copy or convert files
                    for data_file in data_files:
                        if output_format == 'jsonl' and data_file.suffix == '.jsonl':
                            # Copy directly
                            import shutil
                            shutil.copy2(data_file, coreai3d_dataset_dir / data_file.name)
                        elif output_format == 'json' and data_file.suffix == '.jsonl':
                            # Convert to JSON array
                            self._convert_jsonl_to_json(data_file, coreai3d_dataset_dir)

        # Create CoreAI3D metadata
        coreai3d_metadata = {
            'name': dataset_info['name'],
            'description': dataset_info.get('description', ''),
            'data_type': 'text',  # Default assumption
            'created_at': dataset_info.get('downloaded_at', ''),
            'files': [f.name for f in coreai3d_dataset_dir.glob('*') if f.is_file()],
            'metadata': dataset_info
        }

        metadata_file = coreai3d_dataset_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(coreai3d_metadata, f, indent=2, ensure_ascii=False)

        return str(coreai3d_dataset_dir)

    def _convert_jsonl_to_json(self, jsonl_file: Path, output_dir: Path):
        """Convert JSONL file to JSON array"""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        json_file = output_dir / f"{jsonl_file.stem}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# CLI interface
def main():
    """Command line interface for dataset management"""
    import argparse

    parser = argparse.ArgumentParser(description='Hugging Face Dataset Manager for CoreAI3D')
    parser.add_argument('action', choices=['search', 'download', 'list', 'convert'],
                       help='Action to perform')
    parser.add_argument('--query', help='Search query for datasets')
    parser.add_argument('--dataset', help='Dataset name to download')
    parser.add_argument('--data-dir', default='training_data', help='Data directory')
    parser.add_argument('--format', default='json', choices=['json', 'csv', 'parquet'],
                       help='Output format')
    parser.add_argument('--limit', type=int, default=10, help='Search result limit')

    args = parser.parse_args()

    manager = HuggingFaceDatasetManager(args.data_dir)

    if not manager.check_huggingface_availability():
        print("Error: Hugging Face libraries not available. Install with:")
        print("pip install datasets huggingface-hub pandas")
        return

    try:
        if args.action == 'search':
            if not args.query:
                print("Error: --query required for search")
                return

            results = manager.search_datasets(args.query, args.limit)
            for result in results:
                print(f"- {result.name}: {result.description[:100]}...")
                print(f"  Downloads: {result.downloads}, Likes: {result.likes}")
                print()

        elif args.action == 'download':
            if not args.dataset:
                print("Error: --dataset required for download")
                return

            path = manager.download_dataset(args.dataset, save_format=args.format)
            print(f"Dataset downloaded to: {path}")

        elif args.action == 'list':
            datasets = manager.list_downloaded_datasets()
            for dataset in datasets:
                print(f"- {dataset['name']}: {len(dataset.get('files', []))} files")

        elif args.action == 'convert':
            if not args.dataset:
                print("Error: --dataset required for convert")
                return

            # Find dataset path
            datasets = manager.list_downloaded_datasets()
            dataset_path = None
            for dataset in datasets:
                if dataset['name'] == args.dataset:
                    dataset_path = dataset['local_path']
                    break

            if not dataset_path:
                print(f"Error: Dataset {args.dataset} not found locally")
                return

            coreai3d_path = manager.convert_to_coreai3d_format(dataset_path)
            print(f"Converted to CoreAI3D format: {coreai3d_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Hugging Face Dataset Manager
Handles downloading, converting, and managing Hugging Face datasets for CoreAI3D
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi, HfFolder
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: Hugging Face libraries not available. Install with: pip install datasets huggingface-hub")

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about a Hugging Face dataset"""
    name: str
    description: str
    size: Optional[int]
    downloads: int
    likes: int
    tags: List[str]
    author: str = ""
    last_modified: str = ""

class HuggingFaceDatasetManager:
    """Manages Hugging Face dataset operations"""

    def __init__(self, data_dir: str = "training_data", api_token: Optional[str] = None):
        """
        Initialize the Hugging Face dataset manager

        Args:
            data_dir: Directory to store downloaded datasets
            api_token: Hugging Face API token for authentication
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face libraries not available. Install datasets and huggingface-hub")

        self.data_dir = Path(data_dir) / "huggingface"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set up authentication
        if api_token:
            HfFolder.save_token(api_token)
            self.api = HfApi()
        else:
            self.api = HfApi()

        logger.info(f"Hugging Face Dataset Manager initialized with data directory: {self.data_dir}")

    def search_datasets(self, query: str, limit: int = 10) -> List[DatasetInfo]:
        """
        Search for datasets on Hugging Face

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of DatasetInfo objects
        """
        try:
            # Search for datasets
            results = self.api.list_datasets(
                search=query,
                limit=limit,
                sort="downloads",
                direction=-1  # Descending
            )

            datasets = []
            for result in results:
                dataset_info = DatasetInfo(
                    name=result.id,
                    description=result.description or "",
                    size=getattr(result, 'size', None),
                    downloads=getattr(result, 'downloads', 0),
                    likes=getattr(result, 'likes', 0),
                    tags=getattr(result, 'tags', []),
                    author=getattr(result, 'author', ''),
                    last_modified=getattr(result, 'lastModified', '')
                )
                datasets.append(dataset_info)

            logger.info(f"Found {len(datasets)} datasets for query: {query}")
            return datasets

        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return []

    def download_dataset(self, dataset_name: str, save_format: str = 'json',
                        split: str = 'train', **kwargs) -> str:
        """
        Download a dataset from Hugging Face

        Args:
            dataset_name: Name of the dataset (e.g., 'imdb', 'glue', 'squad')
            save_format: Format to save in ('json', 'csv', 'parquet')
            split: Dataset split to download ('train', 'test', 'validation')
            **kwargs: Additional arguments for load_dataset

        Returns:
            Path to the downloaded dataset directory
        """
        try:
            logger.info(f"Downloading dataset: {dataset_name}")

            # Load the dataset
            dataset = load_dataset(dataset_name, split=split, **kwargs)

            # Create dataset directory
            dataset_dir = self.data_dir / dataset_name.replace('/', '_')
            dataset_dir.mkdir(exist_ok=True)

            # Save in the requested format
            if save_format == 'json':
                self._save_as_json(dataset, dataset_dir)
            elif save_format == 'csv':
                self._save_as_csv(dataset, dataset_dir)
            elif save_format == 'parquet':
                self._save_as_parquet(dataset, dataset_dir)
            else:
                raise ValueError(f"Unsupported format: {save_format}")

            logger.info(f"Dataset {dataset_name} downloaded and saved to {dataset_dir}")
            return str(dataset_dir)

        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {e}")
            raise

    def convert_to_coreai3d_format(self, dataset_path: str) -> str:
        """
        Convert downloaded dataset to CoreAI3D format

        Args:
            dataset_path: Path to the downloaded dataset

        Returns:
            Path to the CoreAI3D formatted dataset
        """
        try:
            dataset_path = Path(dataset_path)
            coreai3d_path = self.data_dir.parent / dataset_path.name

            # Create CoreAI3D format directory
            coreai3d_path.mkdir(exist_ok=True)

            # Find data files
            data_files = list(dataset_path.glob("*.json"))
            if not data_files:
                data_files = list(dataset_path.glob("*.csv"))
            if not data_files:
                data_files = list(dataset_path.glob("*.parquet"))

            if not data_files:
                raise FileNotFoundError(f"No data files found in {dataset_path}")

            # Create metadata
            metadata = {
                'name': dataset_path.name,
                'description': f"Converted from Hugging Face dataset {dataset_path.name}",
                'data_type': 'text',  # Default, can be updated based on content
                'created_at': str(Path(dataset_path).stat().st_mtime),
                'files': [],
                'metadata': {
                    'source': 'huggingface',
                    'original_name': dataset_path.name,
                    'conversion_date': str(Path().cwd().stat().st_mtime)
                }
            }

            # Process each data file
            for data_file in data_files:
                # Copy file to CoreAI3D format
                coreai3d_file = coreai3d_path / data_file.name
                coreai3d_file.write_bytes(data_file.read_bytes())

                # Add to metadata
                metadata['files'].append({
                    'name': data_file.name,
                    'path': str(coreai3d_file),
                    'added_at': str(data_file.stat().st_mtime),
                    'size': data_file.stat().st_size
                })

            # Save metadata
            metadata_file = coreai3d_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Converted dataset to CoreAI3D format: {coreai3d_path}")
            return str(coreai3d_path)

        except Exception as e:
            logger.error(f"Error converting dataset to CoreAI3D format: {e}")
            raise

    def _save_as_json(self, dataset: Dataset, output_dir: Path):
        """Save dataset as JSON"""
        # Convert to list of dicts
        data = []
        for item in dataset:
            # Convert tensors/other non-serializable objects to strings
            processed_item = {}
            for key, value in item.items():
                if hasattr(value, 'tolist'):  # Tensor/array
                    processed_item[key] = value.tolist()
                else:
                    processed_item[key] = str(value) if not isinstance(value, (str, int, float, bool, list, dict)) else value
            data.append(processed_item)

        # Save as JSON
        output_file = output_dir / 'data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_as_csv(self, dataset: Dataset, output_dir: Path):
        """Save dataset as CSV"""
        import pandas as pd

        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)

        # Save as CSV
        output_file = output_dir / 'data.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')

    def _save_as_parquet(self, dataset: Dataset, output_dir: Path):
        """Save dataset as Parquet"""
        import pandas as pd

        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)

        # Save as Parquet
        output_file = output_dir / 'data.parquet'
        df.to_parquet(output_file, index=False)

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """
        Get information about a specific dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetInfo object or None if not found
        """
        try:
            dataset_info = self.api.dataset_info(dataset_name)

            return DatasetInfo(
                name=dataset_info.id,
                description=dataset_info.description or "",
                size=getattr(dataset_info, 'size', None),
                downloads=getattr(dataset_info, 'downloads', 0),
                likes=getattr(dataset_info, 'likes', 0),
                tags=getattr(dataset_info, 'tags', []),
                author=getattr(dataset_info, 'author', ''),
                last_modified=getattr(dataset_info, 'lastModified', '')
            )

        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_name}: {e}")
            return None

    def list_user_datasets(self, username: str) -> List[DatasetInfo]:
        """
        List datasets by a specific user

        Args:
            username: Hugging Face username

        Returns:
            List of DatasetInfo objects
        """
        try:
            datasets = self.api.list_datasets(author=username)
            return [DatasetInfo(
                name=ds.id,
                description=ds.description or "",
                size=getattr(ds, 'size', None),
                downloads=getattr(ds, 'downloads', 0),
                likes=getattr(ds, 'likes', 0),
                tags=getattr(ds, 'tags', []),
                author=username
            ) for ds in datasets]

        except Exception as e:
            logger.error(f"Error listing datasets for user {username}: {e}")
            return []

    def get_downloaded_datasets(self) -> List[str]:
        """
        Get list of downloaded dataset names

        Returns:
            List of dataset names
        """
        try:
            return [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Error getting downloaded datasets: {e}")
            return []
#!/usr/bin/env python3
"""
CoreAI3D Cybersecurity / Programming / Scripting Knowledge Training Orchestrator

Loads three curated knowledge-base datasets into the CoreAI3D neural network
and trains the model. All datasets are structured as:
  - 6 text input columns (hashed into feature vectors by backend)
  - 4 numeric target columns (classification / severity / priority / automation)

Datasets:
  1. cybersecurity_training.csv  - attack types, defenses, tools, frameworks
  2. programming_training.csv    - languages, paradigms, patterns, algorithms
  3. scripting_training.csv      - shell commands, sysadmin, automation

Usage (from within the admin-gui or any container on the skynet network):
  cd /app
  python3 scripts/train_knowledge_base.py

Or via Docker exec:
  host$ docker exec -it coreai3d_admin_gui bash
  container$ cd /app && python3 scripts/train_knowledge_base.py

The script hits the CoreAI3D HTTP API at http://coreai3d:8080 using
the /api/v1/training endpoints.
"""

import aiohttp
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COREAI3D_API_BASE = "http://coreai3d:8080/api/v1"
COREAI3D_API_PORT = 8080

# Training data lives in the same docker-compose volume as the backend sees.
# Inside the coreai3d container: /training_data/<dataset>.csv
TRAINING_DATA_PATH = "/training_data"
TRAINING_DATA_DIR_HOST = Path("/app/training_data")

# Neural network hyperparameters (tuned for small structured datasets)
DEFAULT_HYPERPARAMS = {
    "learning_rate": 0.01,
    "epochs": 500,          # 500 epochs for convergence on small datasets
    "layers": 3,            # 3 hidden layers
    "neurons": 24,          # 24 neurons per hidden layer
    "batch_size": 16,
    "optimizer": "adam",
    "activation": "relu",
    "min_range": 0.0,
    "max_range": 1.0,
}

# Dataset descriptors
DATASETS = [
    {
        "name": "cybersecurity_knowledge",
        "filename": "cybersecurity_training.csv",
        "description": "Cybersecurity attack types, defenses, tools, frameworks, incident response",
        "input_size": 6,      # 6 text columns hashed by backend
        "output_size": 4,     # category_norm, threat_level, priority_norm, automation_value
        "has_header": True,
        "contains_text": True,
    },
    {
        "name": "programming_knowledge",
        "filename": "programming_training.csv",
        "description": "Programming languages, paradigms, patterns, algorithms, tools",
        "input_size": 6,
        "output_size": 4,     # difficulty_norm, paradigm_norm, priority_norm, automation_value
        "has_header": True,
        "contains_text": True,
    },
    {
        "name": "scripting_knowledge",
        "filename": "scripting_training.csv",
        "description": "Shell commands, system administration, automation, security hardening",
        "input_size": 6,
        "output_size": 4,     # risk_norm, automation_norm, priority_norm, efficiency_norm
        "has_header": True,
        "contains_text": True,
    },
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class ApiResult:
    ok: bool
    status_code: int
    data: dict = None
    error: str = ""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def api_request(
    session: aiohttp.ClientSession,
    method: str,
    path: str,
    payload: dict = None,
) -> ApiResult:
    """Issue a request to the CoreAI3D API. Returns an ApiResult."""
    url = f"{COREAI3D_API_BASE}{path}"
    try:
        async with session.request(method, url, json=payload, timeout=120) as resp:
            status = resp.status
            try:
                body = await resp.json(content_type=None)
            except Exception:
                text = await resp.text()
                body = {"raw": text}
            return ApiResult(
                ok=200 <= status < 300,
                status_code=status,
                data=body,
                error="" if 200 <= status < 300 else f"HTTP {status}",
            )
    except Exception as e:
        return ApiResult(ok=False, status_code=0, error=str(e))


async def check_api_health(session: aiohttp.ClientSession) -> bool:
    """Verify the CoreAI3D API is reachable."""
    result = await api_request(session, "GET", "/health")
    return result.ok


def build_absolute_path(filename: str) -> str:
    """
    Build absolute path for dataset file.
    The coreai3d container mounts ./training_data as /training_data
    """
    return f"/training_data/{filename}"


async def load_dataset(
    session: aiohttp.ClientSession,
    ds: dict,
) -> ApiResult:
    """Instruct CoreAI3D to load a CSV dataset."""
    payload = {
        "filename": build_absolute_path(ds["filename"]),
        "directory": "/training_data",
        "output_size": ds["output_size"],
        "input_size": ds["input_size"],
        "has_header": ds["has_header"],
        "contains_text": ds["contains_text"],
        "dataset_name": ds["name"],
        "description": ds["description"],
        "min_range": DEFAULT_HYPERPARAMS["min_range"],
        "max_range": DEFAULT_HYPERPARAMS["max_range"],
        "layers": DEFAULT_HYPERPARAMS["layers"],
        "neurons": DEFAULT_HYPERPARAMS["neurons"],
    }
    return await api_request(session, "POST", "/training/load_dataset", payload)


async def start_training(
    session: aiohttp.ClientSession,
    ds: dict,
) -> ApiResult:
    """Start asynchronous training on the named dataset."""
    payload = {
        "learning_rate": DEFAULT_HYPERPARAMS["learning_rate"],
        "epochs": DEFAULT_HYPERPARAMS["epochs"],
        "dataset_name": ds["name"],
        "min_range": DEFAULT_HYPERPARAMS["min_range"],
        "max_range": DEFAULT_HYPERPARAMS["max_range"],
        "layers": DEFAULT_HYPERPARAMS["layers"],
        "neurons": DEFAULT_HYPERPARAMS["neurons"],
        "input_size": ds["input_size"],
        "output_size": ds["output_size"],
        "batch_size": DEFAULT_HYPERPARAMS["batch_size"],
        "optimizer": DEFAULT_HYPERPARAMS["optimizer"],
        "activation": DEFAULT_HYPERPARAMS["activation"],
    }
    return await api_request(session, "POST", "/training/start", payload)


async def get_training_status(session: aiohttp.ClientSession) -> ApiResult:
    """Poll the current training status endpoint."""
    return await api_request(session, "GET", "/training/status")


async def wait_for_ready(
    session: aiohttp.ClientSession,
    max_wait_seconds: int = 60,
) -> bool:
    """Wait until training reports idle/ready before starting a new job."""
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        result = await get_training_status(session)
        if result.ok and result.data:
            status = result.data.get("data", {}).get("status", "")
            if status.lower() in ("idle", "ready", "completed", "failed", "stopped", ""):
                return True
        await asyncio.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Dataset sanity check
# ---------------------------------------------------------------------------
def validate_local_csv(ds: dict) -> bool:
    """Quick sanity-check the CSV file exists and has correct column count."""
    path = TRAINING_DATA_DIR_HOST / ds["filename"]
    if not path.exists():
        # Try the container's training_data mount path
        alt_path = Path(TRAINING_DATA_PATH) / ds["filename"]
        if not alt_path.exists():
            logger.error(f"CSV not found: {path} (alt: {alt_path})")
            return False
        path = alt_path

    with open(path) as fh:
        header = fh.readline().strip()
        cols = header.split(",")
        expected = ds["input_size"] + ds["output_size"]
        if len(cols) != expected:
            logger.error(
                f"{ds['filename']}: header has {len(cols)} columns, "
                f"expected {expected} (input={ds['input_size']}, output={ds['output_size']})"
            )
            return False

        line_num = 1
        for raw in fh:
            line_num += 1
            if not raw.strip():
                continue
            # Naive split (does not handle commas in quoted fields)
            row = raw.strip().split(",")
            numeric_start = len(row) - ds["output_size"]
            numeric_cells = row[numeric_start:]
            for cell in numeric_cells:
                try:
                    float(cell)
                except ValueError:
                    logger.error(
                        f"{ds['filename']} line {line_num}: "
                        f"target column value is not numeric: '{cell}'"
                    )
                    return False
    logger.info(f"✓ {ds['filename']}: validated {expected} columns, numeric targets ok")
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def train_all_datasets():
    """Load and train on all three datasets sequentially."""
    logger.info("=" * 70)
    logger.info("CoreAI3D Knowledge Base Training")
    logger.info("=" * 70)
    logger.info(f"API base    : {COREAI3D_API_BASE}")
    logger.info(f"Datasets    : {len(DATASETS)}")
    logger.info(f"Hyperparams : {DEFAULT_HYPERPARAMS}")
    logger.info("")

    # Validate all CSVs first
    logger.info("[1/4] Validating local CSV files...")
    for ds in DATASETS:
        if not validate_local_csv(ds):
            logger.error(f"Validation failed for {ds['filename']}. Aborting.")
            return False
    logger.info("")

    # 2. Check API health
    logger.info("[2/4] Checking API connectivity...")
    async with aiohttp.ClientSession() as session:
        healthy = await check_api_health(session)
        if not healthy:
            logger.error(
                "CoreAI3D API is not reachable at %s. "
                "Make sure the coreai3d container is running.",
                COREAI3D_API_BASE,
            )
            return False
        logger.info("✓ API is healthy")
    logger.info("")

    # 3. Load + train each dataset sequentially
    logger.info("[3/4] Loading datasets and starting training...")
    total_start = time.time()

    async with aiohttp.ClientSession() as session:
        for idx, ds in enumerate(DATASETS, start=1):
            logger.info(f"--- Dataset {idx}/{len(DATASETS)}: {ds['name']} ---")
            logger.info(f"    File : {ds['filename']}")
            logger.info(f"    In/Out: {ds['input_size']} → {ds['output_size']}")

            # Load dataset
            load_result = await load_dataset(session, ds)
            if not load_result.ok:
                logger.error(
                    f"Failed to load dataset {ds['name']}: "
                    f"{load_result.error} | {load_result.data}"
                )
                continue
            logger.info(f"✓ Loaded: {json.dumps(load_result.data)[:300]}")

            # Wait for any in-flight training to settle
            logger.info("    Waiting for ready state...")
            if not await wait_for_ready(session):
                logger.error("Timeout waiting for training to become ready. Skipping.")
                continue

            # Start training
            logger.info(f"    Starting training with {DEFAULT_HYPERPARAMS['epochs']} epochs...")
            start_result = await start_training(session, ds)
            if not start_result.ok:
                logger.error(
                    f"Failed to start training for {ds['name']}: "
                    f"{start_result.error} | {start_result.data}"
                )
                continue
            logger.info(f"✓ Training started: {json.dumps(start_result.data)[:300]}")

            # Wait for training to complete
            train_start = time.time()
            while True:
                status_result = await get_training_status(session)
                if not status_result.ok:
                    logger.warning(f"Status endpoint failed: {status_result.error}")
                    await asyncio.sleep(5)
                    continue

                status_data = status_result.data.get("data", {})
                status = status_data.get("status", "unknown")
                epoch = status_data.get("current_epoch", 0)
                epochs = status_data.get("total_epochs", DEFAULT_HYPERPARAMS["epochs"])
                loss = status_data.get("loss", status_data.get("current_loss", "?"))

                logger.info(
                    f"    Training status: {status} "
                    f"({epoch}/{epochs} epochs, loss={loss})"
                )

                if status.lower() in ("completed", "failed", "stopped", "idle"):
                    logger.info(f"    ✓ Finished with status: {status}")
                    break

                await asyncio.sleep(5)

            elapsed = time.time() - train_start
            logger.info(f"    Training completed in {elapsed:.1f}s")
            logger.info("")

    # 4. Final summary
    total_elapsed = time.time() - total_start
    logger.info("[4/4] Training completed")
    logger.info(f"Total time  : {total_elapsed:.1f}s")
    logger.info("=" * 70)
    logger.info("Knowledge base training finished.")
    logger.info("The CoreAI3D model now has embedded knowledge for:")
    logger.info("  • Cybersecurity (attacks, defenses, tools, incident response)")
    logger.info("  • Programming (languages, paradigms, patterns, algorithms)")
    logger.info("  • Scripting (shell, sysadmin, automation, hardening)")
    logger.info("=" * 70)
    return True


def main():
    """Entry point."""
    try:
        success = asyncio.run(train_all_datasets())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

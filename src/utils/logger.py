import json
import logging
import os
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # File handler
    log_file = LOG_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def log_benchmark_result(result: dict) -> None:
    """
    Appends a benchmark result to a structured JSONL file.
    JSONL (one JSON object per line) is trivial to load into pandas later.
    """
    result["timestamp"] = datetime.utcnow().isoformat()
    results_file = LOG_DIR / "benchmark_results.jsonl"

    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")

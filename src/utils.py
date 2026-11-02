"""Shared utilities: paths, configuration, runtime setup, and JSON helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a path anchored at the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration, resolving relative data paths from root."""
    cfg_path = Path(path) if path is not None else project_path("configs", "default.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration at {cfg_path} did not parse to a mapping")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (including parents) and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_cache_env() -> None:
    """Keep torch/Hugging Face caches inside the workspace (sandbox-friendly)."""
    os.environ.setdefault("TORCH_HOME", str(project_path(".cache", "torch")))
    os.environ.setdefault("HF_HOME", str(project_path(".cache", "huggingface")))
    os.environ.setdefault("MPLCONFIGDIR", str(project_path(".cache", "matplotlib")))
    # Some MPS ops lack kernels; fall back to CPU rather than crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------------
# Runtime / device
# ---------------------------------------------------------------------------
def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference into a torch.device without surprises."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference in {"mps", "auto"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(
    device: torch.device | str = "auto",
    mps_memory_fraction: float = 0.55,
    torch_threads: int = 6,
) -> torch.device:
    """Set threading and (on Apple Silicon) a hard MPS allocator cap.

    The cap is the main laptop-safety measure: feature extraction and detection
    cannot consume more than ``mps_memory_fraction`` of unified memory.
    """
    configure_cache_env()
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(max(1, min(4, torch_threads // 2)))
        except RuntimeError:
            # set_num_interop_threads can only be called once per process.
            pass

    resolved = get_device(str(device))
    if resolved.type == "mps":
        try:
            torch.mps.set_per_process_memory_fraction(float(mps_memory_fraction))
        except (AttributeError, RuntimeError):
            pass
        torch.mps.empty_cache()
    return resolved


def empty_device_cache(device: torch.device | str | None = None) -> None:
    """Release cached accelerator memory if applicable."""
    device = get_device("auto") if device is None else torch.device(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    """Atomically write a JSON file, converting NumPy/torch scalars."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Logging / timing
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure stdout logging and optionally mirror to a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def human_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


class Timer:
    """Simple context manager that records elapsed wall-clock seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start

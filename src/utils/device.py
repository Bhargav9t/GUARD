# ==============================================================================
# File: src/utils/device.py
# Description: Hardware device detection, configuration, and telemetry for GUARD.
# Automatically detects CUDA GPUs, Apple Silicon MPS, or falls back to CPU.
# ==============================================================================

from typing import Dict, Any
import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Determines and returns the optimal available PyTorch computing device.

    Parameters:
        prefer_cuda (bool): Whether to prioritize CUDA if available. Default is True.

    Returns:
        torch.device: Configured PyTorch device object ('cuda', 'mps', or 'cpu').
    """
    # 1. Check for NVIDIA CUDA availability
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    # 2. Check for Apple Silicon GPU acceleration (Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 3. Fallback to CPU execution
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Collects detailed hardware device telemetry for debugging, logging, and performance auditing.

    Returns:
        Dict[str, Any]: Dictionary containing device name, type, memory capacity,
                        and acceleration capabilities.
    """
    info: Dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": "CPU",
        "total_memory_gb": None,
        "allocated_memory_gb": None,
    }

    # If CUDA GPU is present, query detailed NVIDIA hardware specifications
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device_index)

        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = device_props.name
        info["cuda_version"] = torch.version.cuda
        info["compute_capability"] = f"{device_props.major}.{device_props.minor}"
        # Convert memory from bytes to Gigabytes (1 GB = 1024^3 bytes)
        info["total_memory_gb"] = round(device_props.total_memory / (1024 ** 3), 2)
        info["allocated_memory_gb"] = round(torch.cuda.memory_allocated(device_index) / (1024 ** 3), 2)
        info["cached_memory_gb"] = round(torch.cuda.memory_reserved(device_index) / (1024 ** 3), 2)
        info["device_type"] = "cuda"

    # If Apple MPS is available
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device_name"] = "Apple Silicon (MPS)"
        info["device_type"] = "mps"

    # Standard CPU fallback
    else:
        info["device_type"] = "cpu"

    return info


def print_device_summary() -> None:
    """
    Prints a formatted summary of the active computing environment to stdout.
    """
    info = get_device_info()
    print("=" * 60)
    print(" GUARD COMPUTE ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f" PyTorch Version    : {info['pytorch_version']}")
    print(f" Target Device Type : {info['device_type'].upper()}")
    print(f" Device Hardware    : {info['device_name']}")
    if info.get("cuda_available"):
        print(f" Total VRAM Capacity: {info['total_memory_gb']} GB")
        print(f" CUDA Compute Cap.  : {info['compute_capability']}")
    print("=" * 60)

from __future__ import annotations

from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """Return CUDA readiness information with safe fallback when torch is unavailable."""
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
            "selected_device": "cpu",
            "cuda_version": None,
            "gpu_name": None,
            "note": "PyTorch not installed; using CPU fallback.",
        }

    cuda_available = bool(torch.cuda.is_available())
    selected_device = "cuda" if cuda_available else "cpu"
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None

    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "selected_device": selected_device,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "note": "CUDA available." if cuda_available else "CUDA unavailable.",
    }


def get_best_device() -> object:
    import torch

    info = get_device_info()
    if info.get("cuda_available"):
        try:
            # quick kernel test
            t = torch.tensor([1.0], device="cuda")
            t += 1
            return torch.device("cuda")
        except Exception:
            print("CUDA detected but kernel image unsupported; falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")

"""Device and environment utilities."""


def is_colab() -> bool:
    """Check if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def has_cuda() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Get the best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_gpu_info() -> dict:
    """Get GPU information if available."""
    info = {"available": False, "name": "CPU", "memory_total": 0, "memory_used": 0}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_total"] = torch.cuda.get_device_properties(0).total_mem / 1e9
            info["memory_used"] = torch.cuda.memory_allocated(0) / 1e9
    except ImportError:
        pass
    return info


def get_colab_runtime_type() -> str:
    """Detect Colab runtime type."""
    if not is_colab():
        return "local"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "T4" in name:
                return "T4"
            elif "A100" in name:
                return "A100"
            elif "V100" in name:
                return "V100"
            return f"GPU ({name})"
    except ImportError:
        pass
    return "CPU"

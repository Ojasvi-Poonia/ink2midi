"""Device selection with NVIDIA CUDA and Apple MPS support."""

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return best available device. Priority: CUDA > MPS > CPU.

    On NVIDIA GPUs (e.g. RTX 3070 Ti), CUDA provides significant speedup.
    Falls back to MPS on Apple Silicon, then CPU if neither is available.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except RuntimeError:
            pass
    # Auto-detect: if preferred device isn't available, try others
    if prefer != "cpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.zeros(1, device="mps")
                return torch.device("mps")
            except RuntimeError:
                pass
    return torch.device("cpu")


def safe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device with fallback to CPU."""
    try:
        return tensor.to(device)
    except RuntimeError:
        return tensor.to("cpu")


def get_device_info() -> dict:
    """Return device availability info for logging."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["cuda_arch"] = f"{props.major}.{props.minor}"
    return info


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage. Useful during training."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    msg = f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total"
    if prefix:
        msg = f"{prefix} | {msg}"
    return msg

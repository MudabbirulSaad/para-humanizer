"""
Hardware utility functions for the UltimateParaphraser.
Provides functions for GPU detection and hardware optimization.
"""
import torch
from typing import Dict, Optional, Union, Any

def detect_gpu() -> bool:
    """
    Detect if a CUDA-capable GPU is available.
    
    Returns:
        Boolean indicating whether GPU is available
    """
    return torch.cuda.is_available()

def check_gpu_availability(use_gpu: bool = True) -> str:
    """
    Check if GPU is available and should be used.
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    return "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

def get_gpu_info() -> Optional[Dict[str, Union[str, float]]]:
    """
    Get information about the available GPU.
    
    Returns:
        Dictionary with GPU name and memory, or None if no GPU is available
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        return {"name": gpu_name, "memory_gb": gpu_memory}
    return None

def get_optimal_batch_size(device: str, default_size: int = 8) -> int:
    """
    Determine optimal batch size based on hardware.
    
    Args:
        device: The device being used ('cuda' or 'cpu')
        default_size: Default batch size if no optimizations are available
        
    Returns:
        Recommended batch size
    """
    if device != "cuda":
        return max(1, default_size // 2)  # Smaller batch for CPU
        
    try:
        # For CUDA, try to estimate based on available memory
        gpu_info = get_gpu_info()
        if gpu_info:
            # Very simple heuristic - adjust batch size based on available GPU memory
            memory_gb = gpu_info["memory_gb"]
            
            if memory_gb < 4:
                return 1  # Very limited memory
            elif memory_gb < 8:
                return 2  # Limited memory
            elif memory_gb < 16:
                return 4  # Moderate memory
            else:
                return default_size  # Sufficient memory
    except Exception:
        pass
        
    # Default fallback
    return default_size

def optimize_torch_settings(device: str) -> None:
    """
    Apply performance optimizations for PyTorch based on the device.
    
    Args:
        device: The device being used ('cuda' or 'cpu')
    """
    if device == "cuda":
        # Enable TF32 for better performance on Ampere GPUs
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.allow_tf32 = True
            
        # Set CUDA memory allocation settings for better performance
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    # Set other general optimizations
    if hasattr(torch, 'set_grad_enabled'):
        torch.set_grad_enabled(False)  # Disable gradient calculation for inference

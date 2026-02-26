"""GPU memory management utilities.

Provides helpers for monitoring, clearing, and managing GPU memory
during YOLOv11 training and inference.
"""

import gc

import torch


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared.")
    else:
        print("No GPU available.")


def show_gpu_memory_status(title: str = "GPU Memory Status") -> None:
    """Print current GPU memory allocation and availability.

    Args:
        title: Header for the status printout.
    """
    if not torch.cuda.is_available():
        print("No GPU available.")
        return

    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
    free = total - allocated

    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")
    print(f"  Allocated : {allocated:.2f} GB")
    print(f"  Reserved  : {reserved:.2f} GB")
    print(f"  Free      : {free:.2f} GB")
    print(f"  Total     : {total:.2f} GB")
    print(f"{'=' * 40}\n")


def manage_gpu_memory() -> None:
    """Show GPU status, clear cache, then show status again."""
    show_gpu_memory_status("Before Cleanup")
    clear_gpu_memory()
    show_gpu_memory_status("After Cleanup")

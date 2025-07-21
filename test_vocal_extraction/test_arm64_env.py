#!/usr/bin/env python3
"""
Test script to verify the arm64 environment is working correctly.
"""

import os
import sys
import platform
import torch
import numpy as np

def main():
    """Print environment information to verify setup."""
    print("\n=== Python Environment Information ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"System: {platform.system()} {platform.version()}")
    
    print("\n=== Package Information ===")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
    
    print("\n=== Demucs Test ===")
    try:
        from demucs.pretrained import get_model
        model = get_model('htdemucs')
        print(f"Demucs model loaded successfully: {model.sources}")
        print("Demucs is working correctly!")
    except Exception as e:
        print(f"Error loading Demucs: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

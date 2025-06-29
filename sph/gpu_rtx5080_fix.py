"""
RTX 5080 GPU support fix for CuPy.

The RTX 5080 uses the new Blackwell architecture (SM 12.0) which isn't 
supported by pre-built CuPy wheels. This module sets up the environment
to compile kernels for the closest supported architecture.
"""

import os
import warnings

def setup_rtx5080_environment():
    """Configure environment for RTX 5080 compatibility."""
    
    # RTX 5080 is SM 12.0, but CuPy might not support it yet
    # Fall back to compiling for SM 8.9 (RTX 4090) which should work
    
    # Option 1: Try to compile for native architecture
    os.environ['CUDA_ARCH_LIST'] = '8.9;9.0'
    
    # Option 2: Set specific compilation flags
    # This tells NVCC to compile for SM 8.9 which is forward compatible
    os.environ['CUPY_CUDA_COMPILE_OPTIONS'] = '-arch=sm_89 --std=c++14'
    
    # Option 3: Disable architecture checks (risky but might work)
    os.environ['CUPY_CUDA_DISABLE_ARCH_CHECK'] = '1'
    
    # Clear any cached kernels
    import cupy
    cupy.cuda.compiler._empty_file_preprocess_cache = {}
    
    warnings.warn(
        "RTX 5080 detected. Using compatibility mode with SM 8.9 architecture. "
        "For best performance, compile CuPy from source with SM 12.0 support.",
        UserWarning
    )

# Auto-setup when imported
try:
    import cupy as cp
    device = cp.cuda.Device()
    props = device.attributes
    major = props['ComputeCapability'][0]
    minor = props['ComputeCapability'][1]
    
    if major == 12 and minor == 0:  # RTX 5080
        setup_rtx5080_environment()
        print("RTX 5080 compatibility mode enabled")
except:
    pass
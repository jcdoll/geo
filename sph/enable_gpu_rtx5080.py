#!/usr/bin/env python3
"""
Enable GPU support for RTX 5080 by configuring CuPy to use sm_89 architecture.
Run this before importing CuPy or using SPH.
"""

import os
import sys

def configure_rtx5080_gpu():
    """Configure environment for RTX 5080 GPU support."""
    
    print("Configuring CuPy for RTX 5080...")
    
    # Set CUDA paths
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Force CuPy to compile for sm_89 (RTX 4090) which should work on RTX 5080
    # This tells NVCC to generate code for compute capability 8.9
    os.environ['CUPY_NVCC_GENERATE_CODE'] = 'arch=compute_89,code=sm_89'
    
    # Alternative: Try multiple architectures
    # os.environ['CUPY_NVCC_GENERATE_CODE'] = 'arch=compute_89,code=sm_89;arch=compute_90,code=sm_90'
    
    # Disable architecture checks
    os.environ['CUPY_CUDA_DISABLE_ARCH_CHECK'] = '1'
    
    # Clear CuPy cache to force recompilation
    os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy_rtx5080_cache'
    
    # Set build options
    os.environ['CUPY_CUDA_COMPILE_WITH_DEBUG'] = '0'
    
    print("Environment configured. Testing CuPy...")
    
    try:
        import cupy as cp
        
        # Simple test
        a = cp.array([1, 2, 3], dtype=cp.float32)
        b = cp.array([4, 5, 6], dtype=cp.float32)
        c = a + b
        result = cp.asnumpy(c)
        
        print(f"✅ CuPy test successful: {a} + {b} = {c}")
        print(f"✅ GPU acceleration is working!")
        
        # Show GPU info
        device = cp.cuda.Device()
        print(f"\nGPU Info:")
        print(f"  Name: {device.name}")
        print(f"  Compute Capability: {device.compute_capability}")
        print(f"  Memory: {device.mem_info[1] / 1e9:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Try clearing CuPy cache: rm -rf ~/.cupy/kernel_cache")
        print("2. Build CuPy from source (see rtx5080_gpu_solution.sh)")
        print("3. Use Numba backend for now (still very fast!)")
        return False


if __name__ == "__main__":
    # Configure environment
    success = configure_rtx5080_gpu()
    
    if success:
        print("\n✅ RTX 5080 GPU support enabled!")
        print("\nTo use in SPH:")
        print("1. Run this script first: python enable_gpu_rtx5080.py")
        print("2. Then run SPH: python main.py")
        print("3. Or import this module before using SPH:")
        print("   import sph.enable_gpu_rtx5080")
        print("   import sph")
        print("   sph.set_backend('gpu')")
    else:
        print("\n❌ GPU setup failed. Using Numba backend is recommended.")
        
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Fix CuPy for RTX 5080 (compute capability 12.0)
"""

import os
import cupy as cp

# Set environment variable to compile for SM 120 (RTX 5080)
os.environ['CUPY_CUDA_COMPILE_WITH_DEBUG'] = '0'
os.environ['CUDA_ARCH'] = '120'  # RTX 5080 compute capability

# Clear CuPy's kernel cache to force recompilation
print("Clearing CuPy kernel cache...")
cp.cuda.compiler._empty_file_preprocess_cache = {}

# Test if it works now
try:
    print("Testing CuPy with RTX 5080...")
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b
    print(f"Success! {a} + {b} = {c}")
    
    # Test a more complex operation
    x = cp.random.random((1000, 1000))
    y = cp.dot(x, x.T)
    print(f"Matrix multiplication successful: {y.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative fix...")
    
    # Alternative: Force compilation for multiple architectures
    os.environ['CUPY_CUDA_COMPILE_OPTIONS'] = '-arch=sm_89 -arch=sm_90'
    
    # Try again
    try:
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        print(f"Alternative fix worked! {a} + {b} = {c}")
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        print("\nYou may need to compile CuPy from source for RTX 5080 support.")
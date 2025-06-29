#!/bin/bash
# Solution for getting CuPy working with RTX 5080 (Blackwell sm_120)

echo "=== RTX 5080 GPU Solution for CuPy ==="
echo ""
echo "The RTX 5080 uses the new Blackwell architecture (sm_120) which requires:"
echo "- CUDA 12.8 or newer"
echo "- Building CuPy from source with sm_120 support"
echo ""

# Check current CUDA version
echo "Current CUDA version:"
nvcc --version | grep "release" || echo "NVCC not found"
echo ""

# Option 1: Build CuPy from source
echo "=== Option 1: Build CuPy from Source ==="
echo "This is the most reliable solution but takes ~30 minutes"
echo ""
echo "Steps:"
echo "1. Uninstall current CuPy:"
echo "   pip uninstall cupy-cuda12x cupy"
echo ""
echo "2. Install build dependencies:"
echo "   sudo apt-get install -y gcc g++ make"
echo ""
echo "3. Clone and build CuPy with sm_120 support:"
echo "   git clone https://github.com/cupy/cupy.git"
echo "   cd cupy"
echo "   git submodule update --init"
echo "   export CUPY_NVCC_GENERATE_CODE=\"arch=compute_89,code=sm_89;arch=compute_90,code=sm_90;arch=compute_120,code=sm_120\""
echo "   export CUPY_NUM_BUILD_JOBS=8"
echo "   pip install -e . --no-build-isolation -v"
echo ""

# Option 2: Use PyTorch's CUDA runtime
echo "=== Option 2: Use PyTorch with CUDA 12.8 ==="
echo "PyTorch nightly builds are starting to support RTX 5080:"
echo ""
echo "1. Install PyTorch nightly with CUDA 12.8:"
echo "   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
echo ""
echo "2. This might help CuPy find the right CUDA libraries"
echo ""

# Option 3: Use JAX instead
echo "=== Option 3: Use JAX (Alternative to CuPy) ==="
echo "JAX may have better RTX 5080 support:"
echo ""
echo "1. Install JAX with CUDA 12:"
echo "   pip install --upgrade \"jax[cuda12]\""
echo ""

# Option 4: Workaround with environment variables
echo "=== Option 4: Force CuPy to use closest architecture ==="
echo "This is a temporary workaround that may work:"
echo ""
echo "Add to your .bashrc:"
echo "export CUPY_NVCC_GENERATE_CODE=\"arch=compute_89,code=sm_89;arch=compute_90,code=sm_90\""
echo "export CUDA_VISIBLE_DEVICES=0"
echo "export CUPY_CACHE_DIR=/tmp/cupy_cache_sm90"
echo ""

echo "=== Recommendation ==="
echo "For immediate use: Continue using the Numba backend (20-50x faster than CPU)"
echo "For GPU acceleration: Build CuPy from source (Option 1)"
echo ""
echo "The Numba backend is already very fast for most use cases!"
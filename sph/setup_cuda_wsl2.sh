#!/bin/bash
# Setup CUDA for WSL2 to enable GPU acceleration in SPH

echo "=== Setting up CUDA for WSL2 ==="
echo "This script will install CUDA runtime libraries needed for CuPy"
echo ""

# Check if running in WSL
if ! grep -q microsoft /proc/version; then
    echo "Warning: This script is designed for WSL2"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if nvidia-smi works
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Make sure you have:"
    echo "1. NVIDIA GPU drivers installed on Windows"
    echo "2. WSL2 (not WSL1)"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name --format=csv,noheader

# Remove old CUDA installations if any
echo ""
echo "Cleaning up old CUDA installations..."
sudo apt-get remove --purge -y cuda* nvidia-cuda-toolkit 2>/dev/null || true
sudo apt-get autoremove -y

# Add NVIDIA package repository
echo ""
echo "Adding NVIDIA package repository..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Update package list
echo ""
echo "Updating package list..."
sudo apt-get update

# Install CUDA toolkit
echo ""
echo "Installing CUDA toolkit 12.6..."
sudo apt-get install -y cuda-toolkit-12-6

# Set up environment variables
echo ""
echo "Setting up environment variables..."

# Add to current session
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Add to .bashrc if not already there
if ! grep -q "cuda-12.6" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# CUDA environment variables' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Test CUDA installation
echo ""
echo "Testing CUDA installation..."
if [ -f /usr/local/cuda-12.6/lib64/libnvrtc.so.12 ]; then
    echo "✓ CUDA runtime libraries found"
else
    echo "✗ CUDA runtime libraries not found"
    exit 1
fi

# Test nvcc
if command -v nvcc &> /dev/null; then
    echo "✓ NVCC compiler found: $(nvcc --version | head -n1)"
else
    echo "✗ NVCC compiler not found"
fi

echo ""
echo "=== CUDA Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Restart your terminal or run: source ~/.bashrc"
echo "2. Test CuPy with: python -c \"import cupy; print(cupy.cuda.is_available())\""
echo "3. Run SPH with GPU: python main.py (will auto-detect GPU)"
echo ""
echo "To manually select GPU backend in SPH:"
echo "  import sph"
echo "  sph.set_backend('gpu')"
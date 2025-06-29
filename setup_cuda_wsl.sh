#\!/bin/bash
# Setup CUDA for WSL2

echo "Setting up CUDA for WSL2..."

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt-get update

# Install CUDA toolkit (runtime libraries only for WSL)
sudo apt-get install -y cuda-toolkit-12-6

# Set up environment variables
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "CUDA setup complete. Please run: source ~/.bashrc"

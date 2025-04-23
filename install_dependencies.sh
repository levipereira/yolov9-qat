#!/bin/bash

# Function to display usage message
usage() {
    echo "Usage: $0 [--no-trex]" 1>&2
    exit 1
}

# Set default flags
install_trex=true  # TREx installation enabled by default
install_base=true  # Base dependencies always installed by default

# Shared paths
TENSORRT_REPO_PATH="/opt/nvidia/TensorRT"
DOWNLOADS_PATH="/yolov9-qat/downloads"

# Check command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-trex)
            install_trex=false
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Function to install system dependencies
install_system_dependencies() {
    echo "Installing system dependencies..."
    apt-get update || return 1
    apt-get install -y zip htop screen libgl1-mesa-glx libfreetype6-dev || return 1
    return 0
}

# Function to upgrade TensorRT
upgrade_tensorrt() {
    echo "Upgrading TensorRT..."
    local os="ubuntu2204"
    local trt_version="10.9.0"
    local cuda="cuda-12.8"
    local tensorrt_package="nv-tensorrt-local-repo-${os}-${trt_version}-${cuda}_1.0-1_amd64.deb"
    local download_path="${DOWNLOADS_PATH}/${tensorrt_package}"
    
    # Create downloads directory if it doesn't exist
    mkdir -p "$DOWNLOADS_PATH" || return 1
    
    # Check if the package already exists
    if [ ! -f "$download_path" ]; then
        echo "Downloading TensorRT package..."
        wget "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_version}/local_repo/${tensorrt_package}" -O "$download_path" || return 1
    else
        echo "TensorRT package already exists at $download_path. Reusing existing file."
    fi
    
    # Install the package
    dpkg -i "$download_path" || return 1
    cp /var/nv-tensorrt-local-repo-${os}-${trt_version}-${cuda}/*keyring.gpg /usr/share/keyrings/ || return 1
    apt-get update || return 1
    apt-get install -y tensorrt || return 1
    apt-get purge "nv-tensorrt-local-repo*" -y || return 1
    
    # Keep the downloaded file for potential reuse
    echo "TensorRT package kept at $download_path for future use"
    return 0
}

# Function to install Python packages
install_python_packages() {
    echo "Installing Python packages..."
    pip install --upgrade pip || return 1
    pip install --upgrade tensorrt==10.9.0.34 || return 1
    
    pip install seaborn \
                thop \
                "markdown-it-py>=2.2.0" \
                "onnx-simplifier>=0.4.35" \
                "onnxsim>=0.4.35" \
                "onnxruntime>=1.16.3" \
                "ujson>=5.9.0" \
                "pycocotools>=2.0.7" \
                "pycuda>=2025.1" || return 1
    
    pip install --upgrade onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com || return 1
    pip install pillow==9.5.0 --no-cache-dir --force-reinstall || return 1
    return 0
}

# Function to clone TensorRT repository once
clone_tensorrt_repo() {
    echo "Cloning NVIDIA TensorRT repository..."
    
    if [ ! -d "$TENSORRT_REPO_PATH" ]; then
        # Create directory and clone repository
        mkdir -p "$(dirname "$TENSORRT_REPO_PATH")" || return 1
        git clone https://github.com/NVIDIA/TensorRT.git "$TENSORRT_REPO_PATH" || return 1
        cd "$TENSORRT_REPO_PATH" || return 1
        git checkout release/10.9 || return 1
        echo "TensorRT repository cloned successfully to $TENSORRT_REPO_PATH"
    else
        echo "TensorRT repository already exists at $TENSORRT_REPO_PATH"
    fi
    
    return 0
}

# Function to install PyTorch Quantization
install_pytorch_quantization() {
    echo "Installing PyTorch Quantization..."
    
    # Navigate to PyTorch Quantization directory in TensorRT repo
    cd "$TENSORRT_REPO_PATH/tools/pytorch-quantization" || return 1
    
    # Install requirements and setup
    pip install -r requirements.txt || return 1
    python setup.py install || return 1
    
    echo "PyTorch Quantization installed successfully"
    return 0
}

# Function to install TREx
install_trex_environment() {
    echo "Installing NVIDIA TREx environment..."
    # Check if TREx is not already installed
    if [ ! -d "/opt/nvidia_trex/env_trex" ]; then
        apt-get install -y graphviz || return 1
        pip install virtualenv "widgetsnbextension>=4.0.9" || return 1
        
        mkdir -p /opt/nvidia_trex || return 1
        cd /opt/nvidia_trex/ || return 1
        python3 -m virtualenv env_trex || return 1
        source env_trex/bin/activate || return 1
        pip install "Werkzeug>=2.2.2" "graphviz>=0.20.1" || return 1
        
        # Navigate to TREx directory in TensorRT repo
        cd "$TENSORRT_REPO_PATH/tools/experimental/trt-engine-explorer" || return 1
        
        source /opt/nvidia_trex/env_trex/bin/activate || return 1
        pip install -e . || return 1
        pip install jupyter_nbextensions_configurator notebook==6.4.12 ipywidgets || return 1
        jupyter nbextension enable widgetsnbextension --user --py || return 1
        deactivate || return 1
    else
        echo "NVIDIA TREx virtual environment already exists. Skipping installation."
    fi
    return 0
}

# Function to cleanup
cleanup() {
    echo "Cleaning up..."
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

# Main installation process
main() {
    # Install base dependencies (always)
    if $install_base; then
        install_system_dependencies || { echo "Failed to install system dependencies"; exit 1; }
        upgrade_tensorrt || { echo "Failed to upgrade TensorRT"; exit 1; }
        install_python_packages || { echo "Failed to install Python packages"; exit 1; }
    fi

    # Clone TensorRT repository once
    clone_tensorrt_repo || { echo "Failed to clone TensorRT repository"; exit 1; }

    # Install TREx by default unless --no-trex flag is provided
    if $install_trex; then
        install_trex_environment || { echo "Failed to install TREx environment"; exit 1; }
    fi
    
    # Always install PyTorch Quantization
    install_pytorch_quantization || { echo "Failed to install PyTorch Quantization"; exit 1; }

    # Final cleanup
    cleanup

    echo "Installation completed successfully."
    return 0
}

# Execute main function
main



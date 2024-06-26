#!/bin/bash

# Function to display usage message
usage() {
    echo "Usage: $0 [--defaults] [--trex]" 1>&2
    exit 1
}

# Set flags to false by default
defaults=false
install_trex=false

# Check command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --defaults)
            defaults=true
            ;;
        --trex)
            trex=true
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Ensure at least one option is provided
if ! $defaults || ! $trex; then
    usage
fi



# If --defaults flag is provided
if $defaults; then
    # Update package lists
    apt-get update || exit 1

    # Install required system dependencies
    apt-get install -y zip htop screen libgl1-mesa-glx libfreetype6-dev || exit 1


    # upgrade TensorRT from 8.5.3 cuda 11.8 to 10.0 Cuda 11.2
    
    wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.1.0/local_repo/nv-tensorrt-local-repo-ubuntu2004-10.1.0-cuda-12.4_1.0-1_amd64.deb || exit 1
    dpkg -i nv-tensorrt-local-repo-ubuntu2004-10.1.0-cuda-12.4_1.0-1_amd64.deb || exit 1
    cp /var/nv-tensorrt-local-repo-ubuntu2004-10.1.0-cuda-12.4/*keyring.gpg /usr/share/keyrings/ || exit 1
    apt-get update || exit 1
    apt-get install -y tensorrt || exit 1
    apt-get purge "nv-tensorrt-local-repo*" -y || exit 1
    rm -f nv-tensorrt-local-repo-ubuntu2004-10.1.0-cuda-12.4_1.0-1_amd64.deb
    # Upgrade pip
    pip install --upgrade pip || exit 1
    pip install --upgrade tensorrt==10.1.0 || exit 1
    # Install necessary Python packages
    
    pip install seaborn \
                thop \
                markdown-it-py==2.2.0 \
                onnx-simplifier==0.4.35 \
                onnxsim==0.4.35 \
                onnxruntime==1.16.3 \
                ujson==5.9.0 \
                pycocotools==2.0.7 \
                pycuda==2024.1  || exit 1
    pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com  || exit 1
    pip install pillow==9.5.0 --no-cache-dir --force-reinstall || exit 1
fi

# If --trex flag is provided
if $trex; then
    # Check if Trex is not installed
    if [ ! -d "/opt/nvidia_trex/env_trex" ]; then
        # Install Trex
        apt-get install -y graphviz || exit 1
        pip install virtualenv \
                    widgetsnbextension==4.0.9 || exit 1
        mkdir -p /opt/nvidia_trex || exit 1
        cd /opt/nvidia_trex/ || exit 1
        python3 -m virtualenv env_trex || exit 1
        source env_trex/bin/activate || exit 1
        pip install Werkzeug==2.2.2 graphviz==0.20.1 || exit 1
        cd /opt/nvidia_trex/ || exit 1
        git clone https://github.com/NVIDIA/TensorRT.git || exit 1
        cd TensorRT || exit 1
        git checkout release/9.3 || exit 1
        cd tools/experimental/trt-engine-explorer || exit 1
        source /opt/nvidia_trex/env_trex/bin/activate || exit 1
        pip install -e . || exit 1
        jupyter nbextension enable widgetsnbextension --user --py || exit 1
        deactivate || exit 1
    else
        echo "Nvidia Trex virtual environment already exists. Skipping installation."
    fi
fi

apt-get clean
rm -rf /var/lib/apt/lists/*
# Check if all packages were installed successfully
if [ $? -eq 0 ]; then
    echo "All packages installed successfully."
else
    echo "Some packages failed to install."
fi



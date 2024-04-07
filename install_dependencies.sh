#!/bin/bash

# Update package lists
apt-get update

# Install required system dependencies
apt-get install -y zip htop screen libgl1-mesa-glx libfreetype6-dev

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*

# Upgrade pip
pip install --upgrade pip

# Install necessary Python packages
pip install seaborn \
            thop \
            markdown-it-py==2.2.0 \
            onnx-simplifier==0.4.35 \
            onnxsim==0.4.35 \
            onnxruntime==1.16.3 \
            ujson==5.9.0 \
            pycocotools==2.0.7 \
            pycuda==2024.1     

pip install pillow==9.5.0 --no-cache-dir --force-reinstall

#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_onnx_file> <image_size> <stride>" 
    exit 1
fi

# Extract input file name without extension
onnx="$1"
image_size="$2"
stride="$3"
network_size=$((image_size + stride))
shape=1x3x${network_size}x${network_size}
file_no_ext="${onnx%.*}"

# Generate engine and graph file paths
trt_engine="$file_no_ext.engine"
graph="$file_no_ext.graph"


# Run trtexec command to generate engine and graph files
trtexec --onnx="${onnx}" \
        --saveEngine="${trt_engine}" \
        --fp16 --int8 \
        --buildOnly \
        --minShapes=images:$shape \
        --optShapes=images:$shape \
        --maxShapes=images:$shape  \
        --memPoolSize=workspace:1024MiB \
        --dumpLayerInfo --exportLayerInfo="${graph}" --profilingVerbosity=detailed

# Check if trtexec command was successful
if [ $? -eq 0 ]; then
    echo "Engine and graph files generated successfully:"
    echo "Engine file: ${trt_engine}"
    echo "Graph file: ${graph}"
else
    echo "Failed to generate engine and graph files."
    exit 1 
fi

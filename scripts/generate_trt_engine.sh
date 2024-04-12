#!/bin/bash

# Function to display usage message
usage() {
    echo "Usage: $0 <input_onnx_file> <image_size> [--generate-graph]" 
    exit 1
}

# Check if the correct number of arguments are provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
fi

function get_free_gpu_memory() {
    # Get the total memory and used memory from nvidia-smi for GPU 0
    local total_memory=$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
    local used_memory=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')

    # Calculate free memory
    local free_memory=$((total_memory - used_memory))
    echo "$free_memory"
}

workspace=$(get_free_gpu_memory)

# Set default values
generate_graph=false

# Parse command line arguments
onnx="$1"
image_size="$2"
stride=32
network_size=$((image_size + stride))
shape=1x3x${network_size}x${network_size}

file_no_ext="${onnx%.*}"

# Generate engine and graph file paths
trt_engine="$file_no_ext.engine"
graph="$trt_engine.layer.json"
profile="$trt_engine.profile.json"
timing="$trt_engine.timing.json"
timing_cache="$trt_engine.timing.cache"


# Check if optional flag --generate-graph is provided
if [ "$3" == "--generate-graph" ]; then
    generate_graph=true
fi

# Run trtexec command to generate engine and graph files
if [ "$generate_graph" = true ]; then
    trtexec --onnx="${onnx}" \
            --saveEngine="${trt_engine}" \
            --fp16 --int8 \
            --useCudaGraph \
            --separateProfileRun \
            --useSpinWait \
            --profilingVerbosity=detailed \
            --minShapes=images:$shape \
            --optShapes=images:$shape \
            --maxShapes=images:$shape  \
            --memPoolSize=workspace:${workspace}MiB \
            --dumpLayerInfo \
            --exportTimes="${timing}" \
            --exportLayerInfo="${graph}" \
            --exportProfile="${profile}" \
            --timingCacheFile="${timing_cache}"

        # Profiling affects the performance of your kernel!
        # Always run and time without profiling.

else
    trtexec --onnx="${onnx}" \
            --saveEngine="${trt_engine}" \
            --fp16 --int8 \
            --useCudaGraph \
            --minShapes=images:$shape \
            --optShapes=images:$shape \
            --maxShapes=images:$shape  \
            --memPoolSize=workspace:${workspace}MiB \
            --timingCacheFile="${timing_cache}"
fi

# Check if trtexec command was successful
if [ $? -eq 0 ]; then
    echo "Engine file generated successfully: ${trt_engine}"
    if [ "$generate_graph" = true ]; then
        echo "Graph file generated successfully: ${graph}"
        echo "Profile file generated successfully: ${profile}"
    fi
else
    echo "Failed to generate engine file."
    exit 1 
fi


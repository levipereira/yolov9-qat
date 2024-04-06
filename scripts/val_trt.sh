#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <weight_file> <dataset.yaml> <img_size>"
    exit 1
fi

weight=$1
data=$2
img_size=$3

# Extract filename without path and extension
prefix=$(basename -- "$weight")
prefix="${prefix%.*}"

file_no_ext="${weight%.*}"

onnx_file="$file_no_ext.onnx"
trt_file="$file_no_ext.engine"
graph_file="$file_no_ext.graph"

# Check if weight file exists
if [ ! -f "$weight" ]; then
    echo "Error: Weight file '$weight' not found."
    exit 1
fi

# Run the script
python3 export_qat.py --weights "$weight" --include onnx --inplace --dynamic --simplify 

echo -e "\n"

# Check if ONNX file was successfully generated
if [ ! -f $onnx_file ]; then
    echo "Error: ONNX file $onnx_file not generated."
    exit 1
fi

# Run the script
bash scripts/generate_trt_engine.sh $onnx_file $img_size 32

if [ $? -ne 0 ]; then
    exit 1 
fi

# Check if Graph file exists
#if [ ! -f "${experiment_dir}/${prefix}.graph" ]; then
#    echo "Error: Graph file '${experiment_dir}/${prefix}.graph' not found."
#    exit 1
#fi

# Run the script
#/bin/bash -c "source /opt/nvidia_trex/env_trex/bin/activate && python3 scripts/draw-engine.py --layer ${experiment_dir}/${prefix}.graph "  

# Run the script
python3 val_trt.py --engine-file $trt_file --data $data


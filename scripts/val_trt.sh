#!/bin/bash


usage() {
    echo "Usage: $0 <weight_file> <dataset.yaml> <img_size> --generate-graph"
    exit 1
}

# Check if all required arguments are provided
if [ "$#" -lt 3 -o "$#" -gt 4 ]; then 
    usage
    exit 1
fi

weight=$1
data=$2
img_size=$3
generate_graph=false

file_no_ext="${weight%.*}"

# Generate engine and graph file paths
onnx_file="$file_no_ext.onnx"
trt_engine="$file_no_ext.engine"
graph="$trt_engine.layer.json"
profile="$trt_engine.profile.json"
 

# Check if optional flag --generate-graph is provided
if [ "$4" == "--generate-graph" ]; then
    generate_graph=true
fi

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
if [ "$generate_graph" = true ]; then
    bash scripts/generate_trt_engine.sh $onnx_file $img_size --generate-graph
else
    bash scripts/generate_trt_engine.sh $onnx_file $img_size
fi

if [ $? -ne 0 ]; then
    exit 1 
fi

if [ "$generate_graph" = true ]; then
    # Check if Graph file exists
    if [ ! -f "$graph" ]; then
        echo "Error: Graph file $graph not found."
        exit 1
    fi

    # Check if Graph file exists
    if [ ! -f "$profile" ]; then
        echo "Error: Graph file $profile not found."
        exit 1
    fi

    # Run the script
    /bin/bash -c "source /opt/nvidia_trex/env_trex/bin/activate && python3 draw-engine.py --layer $graph --profile $profile"  
fi

# Run the script
python3 val_trt.py --engine-file $trt_engine --data $data



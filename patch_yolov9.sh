#!/bin/bash

# Check if the correct number of arguments were provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <yolov9_directory>"
    exit 1
fi

# YOLOv9 directory
yolov9_dir="$1"

# Check if the YOLOv9 directory exists
if [ ! -d "$yolov9_dir" ]; then
    echo "Error: Directory '$yolov9_dir' not found."
    exit 1
fi

if [ ! -f "$yolov9_dir/models/experimental.py" ]; then
    echo "Error: '$yolov9_dir' does not appear to contain a valid YOLOv9 repository."
    echo "Please make sure '$yolov9_dir' is the root directory of a YOLOv9 repository."
    exit 1
fi

# Copy files to the YOLOv9 directory
cp val_trt.py "$yolov9_dir/val_trt.py" && echo "qat.py patched successfully."
cp qat.py "$yolov9_dir/qat.py" && echo "qat.py patched successfully."
cp export_qat.py "$yolov9_dir/export_qat.py" && echo "export_qat.py patched successfully."
cp models/quantize_rules.py "$yolov9_dir/models/quantize_rules.py" && echo "quantize_rules.py patched successfully."
cp models/quantize.py "$yolov9_dir/models/quantize.py" && echo "quantize.py patched successfully."
cp scripts/generate_trt_engine.sh "$yolov9_dir/scripts/generate_trt_engine.sh" && echo "generate_trt_engine.sh patched successfully."
cp scripts/val_trt.sh "$yolov9_dir/scripts/val_trt.sh" && echo "val_trt.sh patched successfully."
cp draw-engine.py "$yolov9_dir/draw-engine.py" && echo "draw-engine.py patched successfully."

cp models/experimental_trt.py "$yolov9_dir/models/experimental_trt.py" && echo "experimental_trt.py patched successfully."
cp segment/qat_seg.py "$yolov9_dir/segment/qat_seg.py" && echo "qat_seg.py patched successfully."

echo "Patch applied successfully to YOLOv9 directory: $yolov9_dir"

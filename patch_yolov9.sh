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

# Copy files to the YOLOv9 directory
cp qat.py "$yolov9_dir/qat.py" && echo "qat.py patched successfully."
cp export.py "$yolov9_dir/export_qat.py" && echo "export_qat.py patched successfully."
cp models/quantize_rules.py "$yolov9_dir/models/quantize_rules.py" && echo "quantize_rules.py patched successfully."
cp models/quantize.py "$yolov9_dir/models/quantize.py" && echo "quantize.py patched successfully."

echo "Patch applied successfully to YOLOv9 directory: $yolov9_dir"

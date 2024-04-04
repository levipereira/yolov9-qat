# YOLOv9 QAT
This implementation of YOLOv9 QAT is specifically tailored for deployment on TensorRT platforms. If you do not intend to deploy your model using TensorRT, it is recommended not to proceed with this implementation.

## Introduction

This repository contains scripts for quantization-aware training (QAT) and sensitive layer analysis for YOLOv9 models.

We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov9 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

## How To QAT Training (Finetune)
### 1.Setup

Suggest to use docker environment.
NVIDIA PyTorch image (`nvcr.io/nvidia/pytorch:23.02-py3`)

Release 23.02 is based on CUDA 12.0.1, which requires NVIDIA Driver release 525 or later. 


## Installation
```bash

docker pull nvcr.io/nvidia/pytorch:23.02-py3

git clone https://github.com/WongKinYiu/yolov9.git

docker run --gpus all  \
 -it \
 --net host  \
 --ipc=host \
 -v $(pwd)/yolov9:/yolov9 \
 -v $(pwd)/coco/:/yolov9/coco \
 -v $(pwd)/runs:/yolov9/runs \
 nvcr.io/nvidia/pytorch:23.02-py3

```

1. Clone and apply patch (Inside Docker)
```bash
cd /
git clone https://github.com/levipereira/yolov9-qat.git
cd /yolov9-qat
./patch_yolov9.sh /yolov9
cd /yolov9
```

2. Install dependencies
```bash
cd /yolov9-qat
./install_dependencies.sh
cd /yolov9
```

3. Download dataset and pretrained model
```bash
$ cd /yolov9
$ bash scripts/get_coco.sh
$ wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
```

## Usage

## Quantize Model

To quantize a YOLOv9 model, run:

```bash
python3 qat.py quantize --weights yolov9-c-converted.pt  --name yolov9_qat --exist-ok

python qat.py quantize --weights <weights_path> --data <data_path> --hyp <hyp_path> ...
```
## Quantize Command Arguments

### Description
This command is used to perform PTQ/QAT finetuning.

### Arguments

- `--weights`: Path to the model weights (.pt). Default: ROOT/runs/models_original/yolov9-c.pt.
- `--data`: Path to the dataset configuration file (data.yaml). Default: ROOT/data/coco.yaml.
- `--hyp`: Path to the hyperparameters file (hyp.yaml). Default: ROOT/data/hyps/hyp.scratch-high.yaml.
- `--device`: Device to use for training/evaluation (e.g., "cuda:0"). Default: "cuda:0".
- `--batch-size`: Total batch size for training/evaluation. Default: 10.
- `--imgsz`, `--img`, `--img-size`: Train/val image size (pixels). Default: 640.
- `--project`: Directory to save the training/evaluation outputs. Default: ROOT/runs/qat.
- `--name`: Name of the training/evaluation experiment. Default: 'exp'.
- `--exist-ok`: Flag to indicate if existing project/name should be overwritten.
- `--iters`: Iterations per epoch. Default: 200.
- `--seed`: Global training seed. Default: 57.
- `--no-last-layer`: Disable QAT on Last Layer to improve mAP but also increase Latency.
- `--supervision-stride`: Supervision stride. Default: 1.
- `--no-eval-origin`: Disable eval for origin model.
- `--no-eval-ptq`: Disable eval for ptq model.
- `--eval-pycocotools`: Evaluation using Pycocotools. Valid only for COCO Dataset. (have bug dont use)

## Sensitive Layer Analysis
```bash
python qat.py sensitive --weights yolov9-c.pt --data coco.yaml --hyp hyp.scratch-high.yaml ...
```

## Sensitive Command Arguments

### Description
This command is used for sensitive layer analysis.

### Arguments

- `--weights`: Path to the model weights (.pt). Default: ROOT/runs/models_original/yolov9-c.pt.
- `--device`: Device to use for training/evaluation (e.g., "cuda:0"). Default: "cuda:0".
- `--data`: Path to the dataset configuration file (data.yaml). Default: data/coco.yaml.
- `--batch-size`: Total batch size for training/evaluation. Default: 10.
- `--imgsz`, `--img`, `--img-size`: Train/val image size (pixels). Default: 640.
- `--hyp`: Path to the hyperparameters file (hyp.yaml). Default: data/hyps/hyp.scratch-high.yaml.
- `--project`: Directory to save the training/evaluation outputs. Default: ROOT/runs/qat_sentive.
- `--name`: Name of the training/evaluation experiment. Default: 'exp'.
- `--exist-ok`: Flag to indicate if existing project/name should be overwritten.
- `--num-image`: Number of images to evaluate. Default: None.


## Evaluate QAT Model

```bash
python qat.py eval --weights yolov9-c.pt --data coco.yaml  
```
## Evaluation Command Arguments

### Description
This command is used to perform evaluation on QAT Models.

### Arguments

- `--weights`: Path to the model weights (.pt). Default: ROOT/runs/models_original/yolov9-c.pt.
- `--data`: Path to the dataset configuration file (data.yaml). Default: data/coco.yaml.
- `--batch-size`: Total batch size for evaluation. Default: 10.
- `--imgsz`, `--img`, `--img-size`: Validation image size (pixels). Default: 640.
- `--device`: Device to use for evaluation (e.g., "cuda:0"). Default: "cuda:0".
- `--conf-thres`: Confidence threshold for evaluation. Default: 0.001.
- `--iou-thres`: NMS threshold for evaluation. Default: 0.7.
- `--project`: Directory to save the evaluation outputs. Default: ROOT/runs/qat_eval.
- `--name`: Name of the evaluation experiment. Default: 'exp'.
- `--exist-ok`: Flag to indicate if existing project/name should be overwritten.
- `--use-pycocotools`: Generate COCO annotation json format for the custom dataset.

# Export ONNX 
The goal of exporting to ONNX is to deploy to TensorRT, not to ONNX runtime. So we only export fake quantized model into a form TensorRT will take. Fake quantization will be broken into a pair of QuantizeLinear/DequantizeLinear ONNX ops. TensorRT will take the generated ONNX graph, and execute it in int8 in the most optimized way to its capability.

## Export ONNX Model without End2End
```bash 
python3 export_qat.py --weights runs/qat/yolov9_qat/weights/qat_best_yolov9-c.pt --include onnx --dynamic --simplify --inplace
```

## Export ONNX Model End2End
```bash
python3 export_qat.py  --weights runs/qat/yolov9_qat/weights/qat_best_yolov9-c.pt --include onnx_end2end
```


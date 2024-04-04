# YOLOv9 QAT

## Introduction

This repository contains scripts for quantization-aware training (QAT) and sensitive layer analysis for YOLOv9 models.

We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov9 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

## How To QAT Training
### 1.Setup

Suggest to use docker environment.
NVIDIA PyTorch image (`nvcr.io/nvidia/pytorch:23.02-py3`)

Release 23.02 is based on CUDA 12.0.1, which requires NVIDIA Driver release 525 or later. 

```bash
$ docker pull nvcr.io/nvidia/pytorch:23.02-py3
```

1. Clone and apply patch
```bash
# use this YoloV9 as a sample base 
cd /
git clone https://github.com/WongKinYiu/yolov9.git
git clone https://github.com/levipereira/yolov9-qat.git
cd yolov9-qat
./patch_yolov9.sh /yolov9
```

2. Install dependencies
```bash
$ pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

3. Download dataset and pretrained model
```bash
$ bash scripts/get_coco.sh
$ wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt
```

## Usage

### Quantize Model

To quantize a YOLOv9 model, run:

```bash
python3 qat.py quantize --weights yolov9-c-converted.pt  --name yolov9_qat --exist-ok

python qat.py quantize --weights <weights_path> --data <data_path> --hyp <hyp_path> --device <device> ...
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


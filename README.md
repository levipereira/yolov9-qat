# YOLOv9 QAT
This repository contains an implementation of YOLOv9 with Quantization-Aware Training (QAT), specifically designed for deployment on platforms utilizing TensorRT for hardware-accelerated inference. <br>
This implementation aims to provide an efficient, low-latency version of YOLOv9 for real-time detection applications.
If you do not intend to deploy your model using TensorRT, it is recommended not to proceed with this implementation.

## Details
- The files in this repository represent a patch that adds QAT functionality to the original [YOLOv9 repository](https://github.com/WongKinYiu/yolov9/).
- This patch is intended to be applied to the main YOLOv9 repository to incorporate the ability to train with QAT.
- The implementation is optimized to work efficiently with TensorRT, an inference library that leverages hardware acceleration to enhance inference performance.
- Users interested in implementing object detection using YOLOv9 with QAT on TensorRT platforms can benefit from this repository as it provides a ready-to-use solution.

We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov9 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

## Accuracy Report
```bash
YOLOv9-C - All layers Quantized

Eval Model | AP       | AP50     | Precision  | Recall
-------------------------------------------------------
Origin     | 0.5297   | 0.699    | 0.7432     | 0.634
PTQ        | 0.5281   | 0.697    | 0.7437     | 0.63
QAT - Best | 0.5287   | 0.6976   | 0.7396     | 0.6345


YOLOv9-C - Last Layer Not Quantized
Eval Model | AP       | AP50     | Precision  | Recall
-------------------------------------------------------
Origin     | 0.5297   | 0.699    | 0.7432     | 0.634
PTQ        | 0.528    | 0.6971   | 0.7463     | 0.6296
QAT - Best | 0.5291   | 0.698    | 0.7381     | 0.6358


YOLOv9-E - All layers Quantized
Eval Model | AP       | AP50     | Precision  | Recall
-------------------------------------------------------
Origin     | 0.5576   | 0.7246   | 0.7547     | 0.6649
PTQ        | 0.5565   | 0.7237   | 0.7505     | 0.6641
QAT - Best | 0.5564   | 0.7232   | 0.7546     | 0.6627

YOLOv9-E - Last Layer Not Quantized

Eval Model | AP       | AP50     | Precision  | Recall
-------------------------------------------------------
Origin     | 0.5576   | 0.7246   | 0.7547     | 0.6649
PTQ        | 0.5568   | 0.724    | 0.7526     | 0.6623
QAT - Best | 0.5569   | 0.7235   | 0.7541     | 0.6631
```

## QAT Training (Finetune)

In this section, we'll outline the steps to perform Quantization-Aware Training (QAT) using fine-tuning. <br> **Please note that the supported quantization mode is fine-tuning only.** <br> The model should be trained using the original implementation train.py, and after training and reparameterization of the model, the user should proceed with quantization.

### Steps:

1. **Train the Model Using [Training Session](https://github.com/WongKinYiu/yolov9/tree/main?tab=readme-ov-file#training):** 
   - Utilize the original implementation train.py to train your YOLOv9 model with your dataset and desired configurations.
   - Follow the training instructions provided in the original YOLOv9 repository to ensure proper training.

2. **Reparameterize the Model [reparameterization.py](https://github.com/sunmooncode/yolov9/blob/main/tools/reparameterization.py):**
   - After completing the training, reparameterize the trained model to prepare it for quantization. This step is crucial for ensuring that the model's weights are in a suitable format for quantization.

3. **[Proceed with Quantization](#quantize-model):**
   - Once the model is reparameterized, proceed with the quantization process. This involves applying the Quantization-Aware Training technique to fine-tune the model's weights, taking into account the quantization effects.

4. **[Eval Pytorch](#evaluate-using-pytorch)  / [Eval TensorRT](#evaluate-using-tensorrt):**
   - After quantization, it's crucial to validate the performance of the quantized model to ensure that it meets your requirements in terms of accuracy and efficiency.
   - Test the quantized model thoroughly at both stages: during the quantization phase using PyTorch and after training using TensorRT.
   - Please note that different versions of TensorRT may yield varying results and perfomance

5. **Export to ONNX:**
   - [Export ONNX](#export-onnx)
   - Once you are satisfied with the quantized model's performance, you can proceed to export it to ONNX format.

6. **Deploy with TensorRT:**
   - [Deployment with TensorRT](#deployment-with-tensorrt)
   - After exporting to ONNX, you can deploy the model using TensorRT for hardware-accelerated inference on platforms supporting TensorRT.

 





By following these steps, you can successfully perform Quantization-Aware Training (QAT) using fine-tuning with your YOLOv9 model.

## How to Install and Training 
Suggest to use docker environment.
NVIDIA PyTorch image (`nvcr.io/nvidia/pytorch:23.02-py3`)

Release 23.02 is based on CUDA 12.0.1, which requires NVIDIA Driver release 525 or later. 


## Installation
```bash

docker pull nvcr.io/nvidia/pytorch:23.02-py3

## clone original yolov9
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
python qat.py sensitive --weights yolov9-c.pt --data data/coco.yaml --hyp hyp.scratch-high.yaml ...
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

### Evaluate using Pytorch 
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

### Evaluate using TensorRT
```bash
./scripts/val_trt.sh <weights> <data yaml>  <image_size>

./scripts/val_trt.sh runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.pt data/coco.yaml 640

```

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

## Deployment with Tensorrt
```bash
 /usr/src/tensorrt/bin/trtexec \
  --onnx=runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.onnx \
  --int8 --fp16  \
  --workspace=102400 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x672x672 \
  --saveEngine=runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.engine

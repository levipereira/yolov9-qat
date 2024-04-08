# YOLOv9 QAT

This repository contains an implementation of YOLOv9 with Quantization-Aware Training (QAT), specifically designed for deployment on platforms utilizing TensorRT for hardware-accelerated inference. <br>
This implementation aims to provide an efficient, low-latency version of YOLOv9 for real-time detection applications.
If you do not intend to deploy your model using TensorRT, it is recommended not to proceed with this implementation.

## Release Highlights
- This release includes an upgrade from TensorRT 8 to TensorRT 10, ensuring compatibility with the CUDA version supported - by the latest NVIDIA Ada Lovelace GPUs.
- The inference has been upgraded utilizing `enqueueV3` instead `enqueueV2`.<br>
- To maintain legacy support for TensorRT 8, a [dedicated branch](https://github.com/levipereira/yolov9-qat/tree/TensorRT-8) has been created. <br>
- We've added a new option `val_trt.sh --generate-graph` which enables [Graph Rendering](#generate-tensort-profiling-and-svg-image) functionality. This feature facilitates the creation of graphical representations of the engine plan in SVG image format. 


## Details
- The files in this repository represent a patch that adds QAT functionality to the original [YOLOv9 repository](https://github.com/WongKinYiu/yolov9/).
- This patch is intended to be applied to the main YOLOv9 repository to incorporate the ability to train with QAT.
- The implementation is optimized to work efficiently with TensorRT, an inference library that leverages hardware acceleration to enhance inference performance.
- Users interested in implementing object detection using YOLOv9 with QAT on TensorRT platforms can benefit from this repository as it provides a ready-to-use solution.

We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov9 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.


# Perfomance / Accuracy

[Full Report](#benchmark-using-tensorrt-10)

## Latency Report 

### TensorRT  Inference 

| Model Name      | Batch Size | Latency (99%) | Throughput (qps) | Total Inferences (IPS) |
|-----------------|------------|----------------|------------------|------------------------|
| YOLOv9-C (FP16) | 1          | 1.25439 ms     | 799              | 799                    |
|                 | 4          | 3.38025 ms     | 299              | 1199                   |
|                 | 8          | 6.62524 ms     | 152              | 1219                   |
|                 |            |                |                  |                        |
| YOLOv9-C (QAT)  | 1          | 1.0752 ms      | 930              | 930                    |
|                 | 4          | 2.57031 ms     | 389              | 1559                   |
|                 | 8          | 5.08618 ms     | 196              | 1575                   |


## Accuracy Report
 
 **YOLOv9-C - All layers Quantized** 

| Eval Model | AP     | AP50   | Precision | Recall |
|------------|--------|--------|-----------|--------|
| Origin (Pytorch)     | 0.5297 | 0.699  | 0.7432    | 0.634  |
| QAT (Pytorch) | 0.5287 | 0.6976 | 0.7396    | 0.6345 |
| QAT (TensorRT)   | 0.5293  | 0.698  | 0.748     | 0.632   |


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

- **This release upgrade TensorRT from 8.5 to 10.0**
- `./install_dependencies.sh --defaults  [--trex]` 
- `--defaults` Install/Upgrade required packages 
- `--trex` Install TensoRT Explorer (trex) on virtual env. Required only if you want generate Graph SVG for visualizing the profiling of a TensorRT engine. 

```bash
cd /yolov9-qat
./install_dependencies.sh --defaults
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
python3 qat.py eval --weights runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.pt  --name eval_qat_yolov9
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

## Generate TensoRT Profiling and SVG image


TensorRT Explorer can be installed by executing `./install_dependencies.sh --trex`.<br> This installation is necessary to enable the generation of Graph SV, allowing visualization of the profiling data for a TensorRT engine.

```bash
./scripts/val_trt.sh runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.pt data/coco.yaml 640 --generate-graph
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
  --maxShapes=images:8x3x640x640 \
  --saveEngine=runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.engine
```

<br><br>
# Benchmark using TensorRT 10
Set variable batch_size 
```bash
export batch_size=1
trtexec \
	--onnx=runs/qat/yolov9-c_qat/weights/qat_best_yolov9-c-converted.onnx \
	--fp16 --int8 \
	--saveEngine=yolov7_qat.engine \
	--warmUp=500 \
	--duration=10  \
	--useCudaGraph \
	--useSpinWait \
	--noDataTransfers \
	--minShapes=images:1x3x640x640 \
	--optShapes=images:${batch_size}x3x640x640 \
	--maxShapes=images:${batch_size}x3x640x640 
```

### Device 
```bash
=== Device Information ===
Available Devices:
  Device 0: "NVIDIA GeForce RTX 4090" 
Selected Device: NVIDIA GeForce RTX 4090
Selected Device ID: 0
Compute Capability: 8.9
SMs: 128
Device Global Memory: 24207 MiB
Shared Memory per SM: 100 KiB
Memory Bus Width: 384 bits (ECC disabled)
Application Compute Clock Rate: 2.58 GHz
Application Memory Clock Rate: 10.501 GHz
```

## Output Details
- `Latency`: refers to the [min, max, mean, median, 99% percentile] of the engine latency measurements, when timing the engine w/o profiling layers.
- `Throughput`: is measured in query (inference) per second (QPS).
- ` Total Inferences` : is measured in `Throughput  * Batch Size` inferences per second (IPS)

## YOLOv9-C QAT
## Batch Size 1
### Result 
- `Latency percentile(99%) = 1.0752 ms` 
- `Throughput: 930 qps` 
- `Total Inferences : 930 ips`


```bash
 === Performance summary ===
 Throughput: 930.005 qps
 Latency: min = 1.07031 ms, max = 1.07666 ms, mean = 1.07362 ms, median = 1.07324 ms, percentile(90%) = 1.07422 ms, percentile(95%) = 1.0752 ms, percentile(99%) = 1.0752 ms
 Enqueue Time: min = 0.00195312 ms, max = 0.00854492 ms, mean = 0.00245467 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
 H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 GPU Compute Time: min = 1.07031 ms, max = 1.07666 ms, mean = 1.07362 ms, median = 1.07324 ms, percentile(90%) = 1.07422 ms, percentile(95%) = 1.0752 ms, percentile(99%) = 1.0752 ms
 D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 Total Host Walltime: 10.0021 s
 Total GPU Compute Time: 9.98683 s
 ```

## BatchSize 4

### Result 
- `Latency percentile(99%) = 2.57031 ms`  
- `Throughput: 389 qps ` 
- `Total Inferences : 1559 ips`
 
```bash
  === Performance summary ===
Throughput: 389.769 qps 
Latency: min = 2.54877 ms, max = 2.57715 ms, mean = 2.56399 ms, median = 2.56396 ms, percentile(90%) = 2.56738 ms, percentile(95%) = 2.56836 ms, percentile(99%) = 2.57031 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0141602 ms, mean = 0.00250862 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 2.54877 ms, max = 2.57715 ms, mean = 2.56399 ms, median = 2.56396 ms, percentile(90%) = 2.56738 ms, percentile(95%) = 2.56836 ms, percentile(99%) = 2.57031 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0034 s
Total GPU Compute Time: 9.99698 s

```


## BatchSize 8

### Result 
- `Latency percentile(99%) = 5.08618 ms` 
- `Throughput: 196 qps ` 
- `Total Inferences : 1575 ips`

```bash
	 === Performance summary ===
 Throughput: 196.97 qps
 Latency: min = 5.0625 ms, max = 5.09375 ms, mean = 5.07529 ms, median = 5.07495 ms, percentile(90%) = 5.08008 ms, percentile(95%) = 5.08203 ms, percentile(99%) = 5.08618 ms
 Enqueue Time: min = 0.00195312 ms, max = 0.0166016 ms, mean = 0.00253098 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00537109 ms
 H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 GPU Compute Time: min = 5.0625 ms, max = 5.09375 ms, mean = 5.07529 ms, median = 5.07495 ms, percentile(90%) = 5.08008 ms, percentile(95%) = 5.08203 ms, percentile(99%) = 5.08618 ms
 D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 Total Host Walltime: 10.0117 s
 Total GPU Compute Time: 10.0085 s
```


## YOLOv9-C Origin
## Batch Size 1
### Result 
- `Latency percentile(99%) = 1.25439 ms` 
- `Throughput: 799 qps` 
- `Total Inferences : 799 ips`


```bash
=== Performance summary ===
Throughput: 799.772 qps
Latency: min = 1.24194 ms, max = 1.25684 ms, mean = 1.24871 ms, median = 1.24951 ms, percentile(90%) = 1.25098 ms, percentile(95%) = 1.25146 ms, percentile(99%) = 1.25439 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0212402 ms, mean = 0.0024104 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 1.24194 ms, max = 1.25684 ms, mean = 1.24871 ms, median = 1.24951 ms, percentile(90%) = 1.25098 ms, percentile(95%) = 1.25146 ms, percentile(99%) = 1.25439 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0016 s
Total GPU Compute Time: 9.9884 s
 ```

## BatchSize 4
### Result 
- `Latency percentile(99%) = 3.38025 ms`  
- `Throughput: 299 qps ` 
- `Total Inferences : 1199 ips`
 
```bash

=== Performance summary ===
Throughput: 299.817 qps
Latency: min = 3.30646 ms, max = 3.3905 ms, mean = 3.33371 ms, median = 3.33496 ms, percentile(90%) = 3.34644 ms, percentile(95%) = 3.35059 ms, percentile(99%) = 3.38025 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0078125 ms, mean = 0.00242277 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 3.30646 ms, max = 3.3905 ms, mean = 3.33371 ms, median = 3.33496 ms, percentile(90%) = 3.34644 ms, percentile(95%) = 3.35059 ms, percentile(99%) = 3.38025 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0061 s
Total GPU Compute Time: 10.0011 s
```


## BatchSize 8

### Result 
- `Latency percentile(99%) = 6.62524 ms` 
- `Throughput: 152 qps ` 
- `Total Inferences : 1219 ips`

```bash
=== Performance summary ===
Throughput: 152.441 qps
Latency: min = 6.51056 ms, max = 6.66724 ms, mean = 6.55825 ms, median = 6.55664 ms, percentile(90%) = 6.5791 ms, percentile(95%) = 6.58301 ms, percentile(99%) = 6.62524 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00683594 ms, mean = 0.00253658 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 6.51056 ms, max = 6.66724 ms, mean = 6.55825 ms, median = 6.55664 ms, percentile(90%) = 6.5791 ms, percentile(95%) = 6.58301 ms, percentile(99%) = 6.62524 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0104 s
Total GPU Compute Time: 10.0079 s
```
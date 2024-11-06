# YOLOv9 QAT for TensorRT  Detection / Segmentation 

This repository contains an implementation of YOLOv9 with Quantization-Aware Training (QAT), specifically designed for deployment on platforms utilizing TensorRT for hardware-accelerated inference. <br>
This implementation aims to provide an efficient, low-latency version of YOLOv9 for real-time detection applications.<br>
If you do not intend to deploy your model using TensorRT, it is recommended not to proceed with this implementation.

- The files in this repository represent a patch that adds QAT functionality to the original [YOLOv9 repository](https://github.com/WongKinYiu/yolov9/).
- This patch is intended to be applied to the main YOLOv9 repository to incorporate the ability to train with QAT.
- The implementation is optimized to work efficiently with TensorRT, an inference library that leverages hardware acceleration to enhance inference performance.
- Users interested in implementing object detection using YOLOv9 with QAT on TensorRT platforms can benefit from this repository as it provides a ready-to-use solution.


We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov9 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

For those who are not familiar with QAT, I highly recommend watching this video:<br> [Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training](https://www.youtube.com/watch?v=0VdNflU08yA)

**Important**<br>
Evaluation of the segmentation model using TensorRT is currently under development. Once I have more available time, I will complete and release this work.

🌟 We still have plenty of nodes to improve Q/DQ, and we rely on the community's contribution to enhance this project, benefiting us all. Let's collaborate and make it even better! 🚀 

## Release Highlights
- This release includes an upgrade from TensorRT 8 to TensorRT 10, ensuring compatibility with the CUDA version supported - by the latest NVIDIA Ada Lovelace GPUs.
- The inference has been upgraded utilizing `enqueueV3` instead `enqueueV2`.<br>
- To maintain legacy support for TensorRT 8, a [dedicated branch](https://github.com/levipereira/yolov9-qat/tree/TensorRT-8) has been created. **Outdated** <br>
- We've added a new option `val_trt.sh --generate-graph` which enables [Graph Rendering](#generate-tensort-profiling-and-svg-image) functionality. This feature facilitates the creation of graphical representations of the engine plan in SVG image format. 


# Perfomance / Accuracy
[Full Report](#benchmark)


## Accuracy Report
 
 **YOLOv9-C**

### Evaluation Results

## Detection
#### Activation SiLU

| Eval Model | AP     | AP50   | Precision | Recall |
|------------|--------|--------|-----------|--------|
| **Origin (Pytorch)**     | 0.529 | 0.699  | 0.743    | 0.634  |
| **INT8 (Pytorch)** | 0.529 | 0.702 | 0.742    | 0.63 |
| **INT8 (TensorRT)**   | 0.529  | 0.696  | 0.739     | 0.635   |


#### Activation ReLU  

| Eval Model | AP     | AP50   | Precision | Recall |
|------------|--------|--------|-----------|--------|
| **Origin (Pytorch)**     | 0.519 | 0.69  | 0.719    | 0.629  |
| **INT8 (Pytorch)** | 0.518 | 0.69 | 0.726    | 0.625 |
| **INT8 (TensorRT)**   | 0.517  | 0.685  | 0.723     | 0.626   |

### Evaluation Comparison 

#### Activation SiLU
| Eval Model           | AP   | AP50 | Precision | Recall |
|----------------------|------|------|-----------|--------|
| **INT8 (TensorRT)** vs **Origin (Pytorch)** |       |      |          |        |
|                      | 0.000 | -0.003 | -0.004 | +0.001 |

#### Activation ReLU
| Eval Model           | AP   | AP50 | Precision | Recall |
|----------------------|------|------|-----------|--------|
| **INT8 (TensorRT)** vs **Origin (Pytorch)** |       |      |          |        |
|                      | -0.002 | -0.005 | +0.004 | -0.003 |

## Segmentation
| Model  | Box |  |  |  | Mask |  |  |  |
|--------|-----|--|--|--|------|--|--|--|
|        | P | R | mAP50 | mAP50-95 | P | R | mAP50 | mAP50-95 |
| Origin | 0.729 | 0.632 | 0.691 | 0.521 | 0.717 | 0.611 | 0.657 | 0.423 |
| PTQ    | 0.729 | 0.626 | 0.688 | 0.520 | 0.717 | 0.604 | 0.654 | 0.421 |
| QAT    | 0.725 | 0.631 | 0.689 | 0.521 | 0.714 | 0.609 | 0.655 | 0.421 |


## Latency/Throughput Report - TensorRT

![image](https://github.com/levipereira/yolov9-qat/assets/22964932/61a46206-9784-4c75-bcd4-6534eba51223)

## Device 
| **GPU**        |                              |
|---------------------------|------------------------------|
| Device           | **NVIDIA GeForce RTX 4090**      |
| Compute Capability        | 8.9                          |
| SMs                       | 128                          |
| Device Global Memory      | 24207 MiB                    |
| Application Compute Clock Rate | 2.58 GHz               |
| Application Memory Clock Rate  | 10.501 GHz             |


### Latency/Throughput  

| Model Name      | Batch Size | Latency (99%) | Throughput (qps) | Total Inferences (IPS) |
|-----------------|------------|----------------|------------------|------------------------|
| **(FP16) SiLU** | 1          | 1.25 ms         | 803               | 803                    |
|                 | 4          | 3.37 ms         | 300              | 1200                   |
|                 | 8          | 6.6 ms         | 153              | 1224                   |
|                 | 12          | 10 ms         | 99              | 1188                   |
|                 |            |                |                  |                        |
| **INT8 (SiLU)**  | 1          | 0.97 ms         | 1030              | 1030                    |
|                 | 4          | 2,06 ms         | 486              | 1944                   |
|                 | 8          | 3.69 ms         | 271              | 2168                   |
|                 | 12          | 5.36 ms         | 189              | 2268                   |
|                 |            |                |                  |                        |
| **INT8 (ReLU)**  | 1          | 0.87 ms         | 1150              | 1150                    |
|                 | 4          | 1.78 ms         | 562              | 2248                   |
|                 | 8          | 3.06 ms         | 327              | 2616                   |
|                 | 12          | 4.63 ms         | 217              | 2604                   |

## Latency/Throughput Comparison (INT8 vs FP16)

| Model Name | Batch Size | Latency (99%) Change | Throughput (qps) Change | Total Inferences (IPS) Change |
|---|---|---|---|---|
| **INT8(SiLU)** vs **FP16** | 1 | -20.8% | +28.4% | +28.4% |
| | 4 | -37.1% | +62.0% | +62.0% |
| | 8 | -41.1% | +77.0% | +77.0% |
| | 12 | -46.9% | +90.9% | +90.9% |
 

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
- `--supervision-stride`: Supervision stride. Default: 1.
- `--no-eval-origin`: Disable eval for origin model.
- `--no-eval-ptq`: Disable eval for ptq model.


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
  --useCudaGraph \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:8x3x640x640 \
  --saveEngine=runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted.engine
```

# Benchmark
Note: To test FP16 Models (such as Origin) remove flag `--int8`
```bash
# Set variable batch_size  and model_path_no_ext
export batch_size=4
export filepath_no_ext=runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted
trtexec \
	--onnx=${filepath_no_ext}.onnx \
	--fp16 \
	--int8 \
	--saveEngine=${filepath_no_ext}.engine \
	--timingCacheFile=${filepath_no_ext}.engine.timing.cache \
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

## YOLOv9-C QAT (SiLU)
## Batch Size 1
```bash
Throughput: 1026.71 qps
Latency: min = 0.969727 ms, max = 0.975098 ms, mean = 0.972263 ms, median = 0.972656 ms, percentile(90%) = 0.973145 ms, percentile(95%) = 0.973633 ms, percentile(99%) = 0.974121 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0195312 ms, mean = 0.00228119 ms, median = 0.00219727 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 0.969727 ms, max = 0.975098 ms, mean = 0.972263 ms, median = 0.972656 ms, percentile(90%) = 0.973145 ms, percentile(95%) = 0.973633 ms, percentile(99%) = 0.974121 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0019 s
Total GPU Compute Time: 9.98417 s
```

## BatchSize 4
```bash
=== Performance summary ===
Throughput: 485.73 qps
Latency: min = 2.05176 ms, max = 2.06152 ms, mean = 2.05712 ms, median = 2.05713 ms, percentile(90%) = 2.05908 ms, percentile(95%) = 2.05957 ms, percentile(99%) = 2.06055 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00708008 ms, mean = 0.00230195 ms, median = 0.00219727 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00415039 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 2.05176 ms, max = 2.06152 ms, mean = 2.05712 ms, median = 2.05713 ms, percentile(90%) = 2.05908 ms, percentile(95%) = 2.05957 ms, percentile(99%) = 2.06055 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0035 s
Total GPU Compute Time: 9.99553 s
```


## BatchSize 8
```bash
=== Performance summary ===
Throughput: 271.107 qps
Latency: min = 3.6792 ms, max = 3.69775 ms, mean = 3.68694 ms, median = 3.68652 ms, percentile(90%) = 3.69043 ms, percentile(95%) = 3.69141 ms, percentile(99%) = 3.69336 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0090332 ms, mean = 0.0023588 ms, median = 0.00231934 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00476074 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 3.6792 ms, max = 3.69775 ms, mean = 3.68694 ms, median = 3.68652 ms, percentile(90%) = 3.69043 ms, percentile(95%) = 3.69141 ms, percentile(99%) = 3.69336 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0071 s
Total GPU Compute Time: 10.0027 s
```
## BatchSize 12
```bash
=== Performance summary ===
Throughput: 188.812 qps
Latency: min = 5.25 ms, max = 5.37097 ms, mean = 5.2946 ms, median = 5.28906 ms, percentile(90%) = 5.32129 ms, percentile(95%) = 5.32593 ms, percentile(99%) = 5.36475 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0898438 ms, mean = 0.00248513 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00463867 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 5.25 ms, max = 5.37097 ms, mean = 5.2946 ms, median = 5.28906 ms, percentile(90%) = 5.32129 ms, percentile(95%) = 5.32593 ms, percentile(99%) = 5.36475 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.01 s
Total GPU Compute Time: 10.0068 s
```



## YOLOv9-C QAT (ReLU)
## Batch Size 1
```bash
 === Performance summary ===
 Throughput: 1149.49 qps
 Latency: min = 0.866211 ms, max = 0.871094 ms, mean = 0.868257 ms, median = 0.868164 ms, percentile(90%) = 0.869385 ms, percentile(95%) = 0.869629 ms, percentile(99%) = 0.870117 ms
 Enqueue Time: min = 0.00195312 ms, max = 0.0180664 ms, mean = 0.00224214 ms, median = 0.00219727 ms, percentile(90%) = 0.00268555 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
 H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 GPU Compute Time: min = 0.866211 ms, max = 0.871094 ms, mean = 0.868257 ms, median = 0.868164 ms, percentile(90%) = 0.869385 ms, percentile(95%) = 0.869629 ms, percentile(99%) = 0.870117 ms
 D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 Total Host Walltime: 10.0018 s
 Total GPU Compute Time: 9.98235 s
```

## BatchSize 4
```bash
=== Performance summary ===
Throughput: 561.857 qps
Latency: min = 1.77344 ms, max = 1.78418 ms, mean = 1.77814 ms, median = 1.77832 ms, percentile(90%) = 1.77979 ms, percentile(95%) = 1.78076 ms, percentile(99%) = 1.78174 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0205078 ms, mean = 0.00233018 ms, median = 0.0022583 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00439453 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 1.77344 ms, max = 1.78418 ms, mean = 1.77814 ms, median = 1.77832 ms, percentile(90%) = 1.77979 ms, percentile(95%) = 1.78076 ms, percentile(99%) = 1.78174 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0043 s
Total GPU Compute Time: 9.99494 s
```


## BatchSize 8
```bash
=== Performance summary ===
Throughput: 326.86 qps
Latency: min = 3.04126 ms, max = 3.06934 ms, mean = 3.05773 ms, median = 3.05859 ms, percentile(90%) = 3.06152 ms, percentile(95%) = 3.0625 ms, percentile(99%) = 3.06396 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0209961 ms, mean = 0.00235826 ms, median = 0.00231934 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00463867 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 3.04126 ms, max = 3.06934 ms, mean = 3.05773 ms, median = 3.05859 ms, percentile(90%) = 3.06152 ms, percentile(95%) = 3.0625 ms, percentile(99%) = 3.06396 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0043 s
Total GPU Compute Time: 9.99877 s
```
## BatchSize 12
```bash
=== Performance summary ===
Throughput: 216.441 qps
Latency: min = 4.60742 ms, max = 4.63184 ms, mean = 4.61852 ms, median = 4.61816 ms, percentile(90%) = 4.62305 ms, percentile(95%) = 4.62439 ms, percentile(99%) = 4.62744 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0131836 ms, mean = 0.00250633 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00341797 ms, percentile(99%) = 0.00531006 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 4.60742 ms, max = 4.63184 ms, mean = 4.61852 ms, median = 4.61816 ms, percentile(90%) = 4.62305 ms, percentile(95%) = 4.62439 ms, percentile(99%) = 4.62744 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0074 s
Total GPU Compute Time: 10.0037 s
```

## YOLOv9-C FP16
## Batch Size 1
```bash
=== Performance summary ===
Throughput: 802.984 qps
Latency: min = 1.23901 ms, max = 1.25439 ms, mean = 1.24376 ms, median = 1.24316 ms, percentile(90%) = 1.24805 ms, percentile(95%) = 1.24902 ms, percentile(99%) = 1.24951 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00756836 ms, mean = 0.00240711 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 1.23901 ms, max = 1.25439 ms, mean = 1.24376 ms, median = 1.24316 ms, percentile(90%) = 1.24805 ms, percentile(95%) = 1.24902 ms, percentile(99%) = 1.24951 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0027 s
Total GPU Compute Time: 9.98985 s
 ```

## BatchSize 4
```bash
=== Performance summary ===
Throughput: 300.281 qps
Latency: min = 3.30341 ms, max = 3.38025 ms, mean = 3.32861 ms, median = 3.3291 ms, percentile(90%) = 3.33594 ms, percentile(95%) = 3.34229 ms, percentile(99%) = 3.37 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00830078 ms, mean = 0.00244718 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 3.30341 ms, max = 3.38025 ms, mean = 3.32861 ms, median = 3.3291 ms, percentile(90%) = 3.33594 ms, percentile(95%) = 3.34229 ms, percentile(99%) = 3.37 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0073 s
Total GPU Compute Time: 10.0025 s
```


## BatchSize 8
```bash
=== Performance summary ===
Throughput: 153.031 qps
Latency: min = 6.47882 ms, max = 6.64679 ms, mean = 6.53299 ms, median = 6.5332 ms, percentile(90%) = 6.55029 ms, percentile(95%) = 6.55762 ms, percentile(99%) = 6.59766 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0117188 ms, mean = 0.00248772 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 6.47882 ms, max = 6.64679 ms, mean = 6.53299 ms, median = 6.5332 ms, percentile(90%) = 6.55029 ms, percentile(95%) = 6.55762 ms, percentile(99%) = 6.59766 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.011 s
Total GPU Compute Time: 10.0085 s
```

## BatchSize 8
```bash
=== Performance summary ===
Throughput: 99.3162 qps
Latency: min = 10.0372 ms, max = 10.0947 ms, mean = 10.0672 ms, median = 10.0674 ms, percentile(90%) = 10.0781 ms, percentile(95%) = 10.0811 ms, percentile(99%) = 10.0859 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0078125 ms, mean = 0.00248219 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 10.0372 ms, max = 10.0947 ms, mean = 10.0672 ms, median = 10.0674 ms, percentile(90%) = 10.0781 ms, percentile(95%) = 10.0811 ms, percentile(99%) = 10.0859 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0286 s
Total GPU Compute Time: 10.0269 s
```


# Segmentation 

## FP16
### Batch Size 8

```bash
 === Performance summary ===
 Throughput: 124.055 qps
 Latency: min = 8.00354 ms, max = 8.18585 ms, mean = 8.05924 ms, median = 8.05072 ms, percentile(90%) = 8.11499 ms, percentile(95%) = 8.1438 ms, percentile(99%) = 8.17456 ms
 Enqueue Time: min = 0.00219727 ms, max = 0.0200653 ms, mean = 0.00271174 ms, median = 0.00256348 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00317383 ms, percentile(99%) = 0.00466919 ms
 H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 GPU Compute Time: min = 8.00354 ms, max = 8.18585 ms, mean = 8.05924 ms, median = 8.05072 ms, percentile(90%) = 8.11499 ms, percentile(95%) = 8.1438 ms, percentile(99%) = 8.17456 ms
 D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 Total Host Walltime: 3.01478 s
 Total GPU Compute Time: 3.01415 s
 ```

 ## INT8 / FP16 
 ### Batch Size 8 
 ```bash
  === Performance summary ===
 Throughput: 223.63 qps
 Latency: min = 4.45544 ms, max = 4.71553 ms, mean = 4.47007 ms, median = 4.46777 ms, percentile(90%) = 4.47284 ms, percentile(95%) = 4.47388 ms, percentile(99%) = 4.47693 ms
 Enqueue Time: min = 0.00219727 ms, max = 0.00854492 ms, mean = 0.00258152 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00305176 ms, percentile(99%) = 0.00439453 ms
 H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 GPU Compute Time: min = 4.45544 ms, max = 4.71553 ms, mean = 4.47007 ms, median = 4.46777 ms, percentile(90%) = 4.47284 ms, percentile(95%) = 4.47388 ms, percentile(99%) = 4.47693 ms
 D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
 Total Host Walltime: 3.00944 s
 Total GPU Compute Time: 3.00836 s
 ```
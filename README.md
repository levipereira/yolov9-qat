# YOLOv9 QAT for NVIDA TensorRT

This repository contains an implementation of YOLOv9 with Quantization-Aware Training (QAT), specifically designed for deployment on platforms utilizing TensorRT for hardware-accelerated inference. <br>
This implementation aims to provide an efficient, low-latency version of YOLOv9 for real-time detection applications.<br>
If you do not intend to deploy your model using TensorRT, it is recommended not to proceed with this implementation.

- The files in this repository represent a patch that adds QAT functionality to the original [YOLOv9 repository](https://github.com/WongKinYiu/yolov9/).
- This patch is intended to be applied to the main YOLOv9 repository to incorporate the ability to train with QAT.
- The implementation is optimized to work efficiently with TensorRT, an inference library that leverages hardware acceleration to enhance inference performance.
- Users interested in implementing object detection using YOLOv9 with QAT on TensorRT platforms can benefit from this repository as it provides a ready-to-use solution.

We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov9 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

**Important**<br>
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
| Eval Model | AP     | AP50   | Precision | Recall |
|------------|--------|--------|-----------|--------|
| **Origin (Pytorch)**     | 0.529 | 0.699  | 0.743    | 0.634  |
| **INT8 (Pytorch)** | 0.529 | 0.702 | 0.747    | 0.629 |
| **INT8 (TensorRT)**   | 0.527  | 0.695  | 0.746     | 0.627   |

### Evaluation Comparison 
| Eval Model           | AP   | AP50 | Precision | Recall |
|----------------------|------|------|-----------|--------|
| **INT8 (TensorRT)** vs **Origin (Pytorch)** |       |      |          |        |
|                      | -0.002 | -0.004 | +0.003 | -0.007 |




## Latency/Throughput Report using only TensorRT


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
| **(FP16)** | 1          | 1.25 ms         | 803               | 803                    |
|                 | 4          | 3.37 ms         | 300              | 1200                   |
|                 | 8          | 6.6 ms         | 153              | 1224                   |
|                 | 12          | 10 ms         | 99              | 1188                   |
|                 |            |                |                  |                        |
| **INT8**  | 1          | 0.99 ms         | 1006              | 1006                    |
|                 | 4          | 2.12 ms         | 473              | 1892                   |
|                 | 8          | 3.84 ms         | 261              | 2088                   |
|                 | 12          | 5.59 ms         | 178              | 2136                   |

### Latency/Throughput Comparison 
| Model Name                                       | Batch Size | Latency (99%) | Throughput (qps) | Total Inferences |
|--------------------------------------------------|------------|----------------|------------------|------------------|
| **INT8** vs **FP16**                |            |                |                  |                  |
|                                                  | 1          | -20.8%         | +25.2%           | +25.2%           |
|                                                  | 4          | -37.1%         | +57.7%           | +57.7%           |
|                                                  | 8          | -41.1%         | +70.6%           | +70.6%           |
|                                                  | 12         | -46.9%         | +79.8%           | +78.9%           |



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

```bash
# Set variable batch_size  and model_path_no_ext
export batch_size=4
export model_path_no_ext=runs/qat/yolov9_qat/weights/qat_best_yolov9-c-converted
trtexec \
	--onnx=${filepath_no_ext}.onnx \
	--fp16 --int8 \
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

## YOLOv9-C QAT
## Batch Size 1
```bash
=== Performance summary ===
Throughput: 1005.92 qps
Latency: min = 0.989258 ms, max = 0.995605 ms, mean = 0.992502 ms, median = 0.992188 ms, percentile(90%) = 0.993408 ms, percentile(95%) = 0.994141 ms, percentile(99%) = 0.994385 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00878906 ms, mean = 0.00237049 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 0.989258 ms, max = 0.995605 ms, mean = 0.992502 ms, median = 0.992188 ms, percentile(90%) = 0.993408 ms, percentile(95%) = 0.994141 ms, percentile(99%) = 0.994385 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0018 s
Total GPU Compute Time: 9.98557 s
 ```

## BatchSize 4
```bash
=== Performance summary ===
Throughput: 472.711 qps
Latency: min = 2.10327 ms, max = 2.12207 ms, mean = 2.11386 ms, median = 2.11621 ms, percentile(90%) = 2.11865 ms, percentile(95%) = 2.11914 ms, percentile(99%) = 2.12012 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00732422 ms, mean = 0.00242147 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00390625 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 2.10327 ms, max = 2.12207 ms, mean = 2.11386 ms, median = 2.11621 ms, percentile(90%) = 2.11865 ms, percentile(95%) = 2.11914 ms, percentile(99%) = 2.12012 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.004 s
Total GPU Compute Time: 9.99644 s
```


## BatchSize 8
```bash
=== Performance summary ===
Throughput: 260.891 qps
Latency: min = 3.82227 ms, max = 3.84717 ms, mean = 3.83144 ms, median = 3.83105 ms, percentile(90%) = 3.83496 ms, percentile(95%) = 3.83594 ms, percentile(99%) = 3.83984 ms
Enqueue Time: min = 0.00195312 ms, max = 0.00830078 ms, mean = 0.00251205 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00439453 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 3.82227 ms, max = 3.84717 ms, mean = 3.83144 ms, median = 3.83105 ms, percentile(90%) = 3.83496 ms, percentile(95%) = 3.83594 ms, percentile(99%) = 3.83984 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0042 s
Total GPU Compute Time: 10.0001 s
```
## BatchSize 12
```bash
=== Performance summary ===
Throughput: 179.916 qps
Latency: min = 5.53577 ms, max = 5.59412 ms, mean = 5.55652 ms, median = 5.55103 ms, percentile(90%) = 5.58081 ms, percentile(95%) = 5.58594 ms, percentile(99%) = 5.58984 ms
Enqueue Time: min = 0.00195312 ms, max = 0.0078125 ms, mean = 0.00254577 ms, median = 0.00244141 ms, percentile(90%) = 0.00292969 ms, percentile(95%) = 0.00292969 ms, percentile(99%) = 0.00457764 ms
H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
GPU Compute Time: min = 5.53577 ms, max = 5.59412 ms, mean = 5.55652 ms, median = 5.55103 ms, percentile(90%) = 5.58081 ms, percentile(95%) = 5.58594 ms, percentile(99%) = 5.58984 ms
D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
Total Host Walltime: 10.0102 s
Total GPU Compute Time: 10.0073 s
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

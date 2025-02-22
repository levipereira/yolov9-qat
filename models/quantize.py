################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy
import types

# PyTorch
import torch
import torch.optim as optim
from torch.cuda import amp
import torch.nn.functional as F

# ModelOpt Quantization (replacing pytorch_quantization)
from modelopt.torch.quantization import nn as quant_nn
from modelopt.torch.quantization.nn.modules.quant_module import QuantInputBase
from modelopt.torch.quantization import calib
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization import model_quant
from modelopt.torch.quantization.model_quant import disable_quantizer
from absl import logging as quant_logging

import onnx_graphsurgeon as gs
from utils.general import (check_requirements, LOGGER,colorstr)
from models.quantize_rules import find_quantizer_pairs


class QuantAdd(torch.nn.Module):
    def __init__(self, quantization, calib_method="max", percentile=99.999):
        super().__init__()
        self.quantization = quantization
        if quantization:
            config = {"num_bits": 8, "calibrator": calib_method}
            if calib_method == "percentile":
                config["percentile"] = percentile
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y

class QuantADownAvgChunk(torch.nn.Module):
    def __init__(self, calib_method="max", percentile=99.999):
        super().__init__()
        config = {"num_bits": 8, "calibrator": calib_method}
        if calib_method == "percentile":
            config["percentile"] = percentile
        self._chunk_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        self._chunk_quantizer._calibrator._torch_hist = True
        self.avg_pool2d = torch.nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        x = self.avg_pool2d(x)
        x = self._chunk_quantizer(x)
        return x.chunk(2, 1)

class QuantAConvAvgChunk(torch.nn.Module):
    def __init__(self, calib_method="max", percentile=99.999):
        super().__init__()
        config = {"num_bits": 8, "calibrator": calib_method}
        if calib_method == "percentile":
            config["percentile"] = percentile
        self._chunk_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        self._chunk_quantizer._calibrator._torch_hist = True
        self.avg_pool2d = torch.nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        x = self.avg_pool2d(x)
        x = self._chunk_quantizer(x)
        return x
    
class QuantRepNCSPELAN4Chunk(torch.nn.Module):
    def __init__(self, c, calib_method="max", percentile=99.999):
        super().__init__()
        config = {"num_bits": 8, "calibrator": calib_method}
        if calib_method == "percentile":
            config["percentile"] = percentile
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        self.c = c

    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims) 
       
class QuantUpsample(torch.nn.Module): 
    def __init__(self, size, scale_factor, mode, calib_method="max", percentile=99.999):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        config = {"num_bits": 8, "calibrator": calib_method}
        if calib_method == "percentile":
            config["percentile"] = percentile
        self._input_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        
    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode) 

             
class QuantConcat(torch.nn.Module): 
    def __init__(self, dim, calib_method="max", percentile=99.999):
        super().__init__()
        config = {"num_bits": 8, "calibrator": calib_method}
        if calib_method == "percentile":
            config["percentile"] = percentile
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim) 
              
   
class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


# Initialize PyTorch Quantization
def initialize():
    quant_desc_input = QuantizerAttributeConfig(calibrator="histogram")
    quant_nn.QuantConv2d.default_quant_desc_input = quant_desc_input
    quant_nn.QuantLinear.default_quant_desc_input = quant_desc_input
    quant_nn.QuantAvgPool2d.default_quant_desc_input = quant_desc_input
    quant_nn.QuantMaxPool2d.default_quant_desc_input = quant_desc_input

    quant_logging.set_verbosity(quant_logging.ERROR)
   

def remove_redundant_qdq_model(onnx_model, f):
    check_requirements('onnx')
    import onnx

    graph = gs.import_onnx(onnx_model)
    nodes = graph.nodes

    mul_nodes = [node for node in nodes if node.op == "Mul" and node.i(0).op == "Conv" and node.i(1).op == "Sigmoid"]
    many_outputs_mul_nodes = []

    for node in mul_nodes: 
        try:
            for i in range(99):
                node.o(i)
        except:
            if i > 1:
                mul_nodename_outnum = {"node": node, "out_num": i}
                many_outputs_mul_nodes.append(mul_nodename_outnum)
    
    for node_dict in many_outputs_mul_nodes:
        if node_dict["out_num"] == 2:
            if node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "QuantizeLinear":
                if node_dict["node"].o(1).o(0).o(0).op == "Concat":
                    concat_dq_out_name = node_dict["node"].o(1).o(0).outputs[0].name
                    for i, concat_input in enumerate(node_dict["node"].o(1).o(0).o(0).inputs):
                        if concat_input.name == concat_dq_out_name:
                            node_dict["node"].o(1).o(0).o(0).inputs[i] = node_dict["node"].o(0).o(0).outputs[0] 
                else:
                    node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0] 
                    

            # elif node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "Concat":
            #     concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
            #     for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
            #         if concat_input.name == concat_dq_out_name:
            #             #print("elif", concat_input.name, concat_dq_out_name )
            #             #print("will-be", node_dict["node"].outputs[0].outputs[1].inputs[i], node_dict["node"].outputs[0].outputs[0].o().outputs[0]  )
            #             node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0] 

       
    # add_nodes = [node for node in nodes if node.op == "Add"]
    # many_outputs_add_nodes = []
    # for node in add_nodes: 
    #     try:
    #         for i in range(99):
    #             node.o(i)
    #     except:
    #         if i > 1 and node.o().op == "QuantizeLinear":
    #             add_nodename_outnum = {"node": node, "out_num": i}
    #             many_outputs_add_nodes.append(add_nodename_outnum)


    # for node_dict in many_outputs_add_nodes:
    #     if node_dict["node"].outputs[0].outputs[0].op == "QuantizeLinear" and node_dict["node"].outputs[0].outputs[1].op == "Concat":
    #         concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
    #         for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
    #             if concat_input.name == concat_dq_out_name:
    #                 node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0]  

    onnx.save(gs.export_onnx(graph), f)


def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        if self.__class__.__name__ == 'QuantAvgPool2d':
            self.__init__(nninstance.kernel_size, nninstance.stride, nninstance.padding, nninstance.ceil_mode, nninstance.count_include_pad)
        elif isinstance(self, QuantInputBase):
            quant_desc_input = QuantizerAttributeConfig(calib_method="histogram")
            self.default_quant_desc_input = quant_desc_input
            super().__init__()
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            quant_desc_input = QuantizerAttributeConfig(calib_method="histogram")
            quant_desc_weight = QuantizerAttributeConfig(calib_method="histogram")
            self.default_quant_desc_input = quant_desc_input
            self.default_quant_desc_weight = quant_desc_weight
            super().__init__()
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy : Union[str, List[str], Callable], path : str) -> bool:

    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def replace_to_quantization_module(model : torch.nn.Module, ignore_policy : Union[str, List[str], Callable] = None, prefixx=colorstr('QAT:')):
    if ignore_policy:
        if isinstance(ignore_policy, str):
            disable_quantizer(model, ignore_policy)
        elif isinstance(ignore_policy, list):
            for pattern in ignore_policy:
                disable_quantizer(model, pattern)
        elif callable(ignore_policy):
            disable_quantizer(model, ignore_policy)
    return model


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)
        if len(names) == 1:
            return value
        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))

def repncspelan4_qaunt_forward(self, x):
     if hasattr(self, "repncspelan4chunkop"):
        y = list(self.repncspelan4chunkop(self.cv1(x), 2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
     else:
         y = list(self.cv1(x).split((self.c, self.c), 1))
         y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
         return self.cv4(torch.cat(y, 1))

def repbottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def upsample_quant_forward(self, x):
    if hasattr(self, "upsampleop"):
        return self.upsampleop(x)
    return F.interpolate(x)

def concat_quant_forward(self, x):
    if hasattr(self, "concatop"):
        return self.concatop(x, self.d)
    return torch.cat(x, self.d)

def adown_quant_forward(self, x):
    if hasattr(self, "adownchunkop"):
        x1, x2 = self.adownchunkop(x)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

def aconv_quant_forward(self, x):
    if hasattr(self, "aconvchunkop"):
        x = self.aconvchunkop(x)
        return self.cv1(x)
      
class QuantRepNBottleneck(torch.nn.Module):
    def __init__(self, calib_method="max", percentile=99.999):
        super().__init__()
        config = {"num_bits": 8, "calibrator": calib_method}
        if calib_method == "percentile":
            config["percentile"] = percentile
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantizerAttributeConfig(**config))

    def forward(self, x, y):
        return self._input0_quantizer(x) + self._input1_quantizer(y)

def apply_custom_rules_to_quantizer(model : torch.nn.Module, export_onnx : Callable):
    export_onnx(model,  "quantization-custom-rules-temp.onnx")
    pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
    for major, sub in pairs:
        print(f"Rules: {sub} match to {major}")
        get_attr_with_path(model, sub).input_quantizer = get_attr_with_path(model, major).input_quantizer
    os.remove("quantization-custom-rules-temp.onnx")

    for name, module in model.named_modules():
        if module.__class__.__name__ == "RepNBottleneck":
            if module.add:
                print(f"Rules: {name}.add match to {name}.cv1")
                module.addop = QuantRepNBottleneck()
                module.addop._input0_quantizer = module.cv1.conv.input_quantizer
                module.addop._input1_quantizer = module.cv1.conv.input_quantizer
                
                def new_forward(self, x):
                    return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
                module.forward = types.MethodType(new_forward, module)

        if isinstance(module, torch.nn.MaxPool2d):
                quant_conv_desc_input = QuantizerAttributeConfig(num_bits=8, calibrator="histogram")
                quant_maxpool2d = quant_nn.QuantMaxPool2d(module.kernel_size,
                                                        module.stride,
                                                        module.padding,
                                                        module.dilation,
                                                        module.ceil_mode,
                                                        quant_desc_input = quant_conv_desc_input)
                set_module(model, name, quant_maxpool2d)

        if module.__class__.__name__ == 'ADown':
            module.adownchunkop = QuantADownAvgChunk()
            module.cv1.conv.input_quantizer = module.adownchunkop._chunk_quantizer
            
            def new_forward(self, x):
                x1, x2 = self.adownchunkop(x)
                x1 = self.cv1(x1)
                x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
                x2 = self.cv2(x2)
                return torch.cat((x1, x2), 1)
            module.forward = types.MethodType(new_forward, module)

        if module.__class__.__name__ == 'AConv':
            module.aconvchunkop = QuantAConvAvgChunk()
            module.cv1.conv.input_quantizer = module.aconvchunkop._chunk_quantizer
            
            def new_forward(self, x):
                x = self.aconvchunkop(x)
                return self.cv1(x)
            module.forward = types.MethodType(new_forward, module)

def replace_custom_module_forward(model, calib_method="max", percentile=99.999):
    config = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "calibrator": calib_method},
            "*input_quantizer": {"num_bits": 8, "calibrator": calib_method},
            "default": {"num_bits": 8, "calibrator": calib_method}
        },
        "algorithm": calib_method
    }
    if calib_method == "percentile":
        config["quant_cfg"]["*weight_quantizer"]["percentile"] = percentile
        config["quant_cfg"]["*input_quantizer"]["percentile"] = percentile
        config["quant_cfg"]["default"]["percentile"] = percentile
    
    model = model_quant.quantize(model, config)
    return model

def calibrate_model(model : torch.nn.Module, dataloader, device, calib_method="max", percentile=99.999, num_batch=25):
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        if calib_method == "percentile":
                            kwargs["method"] = calib_method
                            kwargs["percentile"] = percentile
                        module.load_calib_amax(**kwargs)
                    module._amax = module._amax.to(device)

    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():

            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                imgs = datas[0].to(device, non_blocking=True).float() / 255.0
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
    
    with torch.no_grad():
        collect_stats(model, dataloader, device, num_batch=num_batch)
        compute_amax(model, method=calib_method)
   


def finetune(
    model : torch.nn.Module, train_dataloader,  per_epoch_callback : Callable = None, preprocess : Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule : Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy : Callable = None, prefix=colorstr('QAT:')
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()
                    
    model.train()
    model.requires_grad_(True)

    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device       = next(model.parameters()).device


    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            6: 1e-5,
            7: 1e-6
        }


    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])


    for iepoch in range(nepochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs  = []
        origin_outputs = []
        remove_handle  = []
        


        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):

            if ibatch >= early_exit_batchs_per_epoch:
                break
            
            if preprocess:
                imgs = preprocess(imgs)
                

            imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for mo, fo in zip(model_outputs, origin_outputs):
                    for m, f in zip(mo, fo):
                        quant_loss += quant_lossfn(m, f)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break


def export_onnx(model, input, file, *args, **kwargs):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False

import torch
import torch.nn as nn

class TRT_EfficientNMS_85(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes
    
class TRT_EfficientNMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25,
                 class_agnostic=0):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   class_agnostic_i=class_agnostic,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes

class TRT_EfficientNMSX_85(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0,num_boxes,(batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMSX_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=5)
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices
    
class TRT_EfficientNMSX(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0,num_boxes,(batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25,
                 class_agnostic=0):
        out = g.op("TRT::EfficientNMSX_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   class_agnostic_i=class_agnostic,
                   score_threshold_f=score_threshold,
                   outputs=5)
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices

class TRT_ROIAlign(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode= 1,
        mode=1,  # 1- avg pooling  / 0 - max pooling
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=1,
        mode=1,
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        return g.op(
            "TRT::ROIAlign_TRT",
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_i=coordinate_transformation_mode,
            mode_i=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )
    
class ONNX_EfficientNMS_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes
        

    def forward(self, x):
        if isinstance(x, list):  
            x = x[1]
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold, self.class_agnostic)
        else:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS_85.apply(bboxes, scores, self.background_class, self.box_coding,
                                                            self.iou_threshold, self.max_obj,
                                                            self.plugin_version, self.score_activation,
                                                            self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes

class ONNX_EfficientNMSX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes
        

    def forward(self, x):
        if isinstance(x, list):  
            x = x[1]
        x = x.permute(0, 2, 1)
        bboxes_x = x[..., 0:1]
        bboxes_y = x[..., 1:2]
        bboxes_w = x[..., 2:3]
        bboxes_h = x[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = x[..., 4:]
        scores = obj_conf
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold, self.class_agnostic)
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes, det_indices

 

class End2End_TRT(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, mask_resolution=56, pooler_scale=0.25, sampling_ratio=0, max_wh=None, device=None, n_classes=80, is_det_model=True):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        if is_det_model:
            self.patch_model = ONNX_EfficientNMS_TRT 
            self.end2end = self.patch_model(class_agnostic, max_obj, iou_thres, score_thres, max_wh, device, n_classes)
        else:
            self.patch_model = ONNX_End2End_MASK_TRT 
            self.end2end = self.patch_model(class_agnostic, max_obj, iou_thres, score_thres, mask_resolution, pooler_scale, sampling_ratio, max_wh, device, n_classes) 
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x
    

class ONNX_End2End_MASK_TRT(nn.Module):
    """onnx module with ONNX-TensorRT NMS/ROIAlign operation."""
    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        mask_resolution=160,
        pooler_scale=0.25,
        sampling_ratio=0,
        max_wh=None,
        device=None,
        n_classes=80
    ):
        super().__init__()
        assert isinstance(max_wh,(int)) or max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.max_obj = max_obj
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio
       
    def forward(self, x):
        if isinstance(x, list):   ## remove auxiliary branch
            x = x[1]
        det=x[0]
        proto=x[1]
        det = det.permute(0, 2, 1)

        bboxes_x = det[..., 0:1]
        bboxes_y = det[..., 1:2]
        bboxes_w = det[..., 2:3]
        bboxes_h = det[..., 3:4]
        bboxes = torch.cat([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        scores = det[..., 4: 4 + self.n_classes]
       
        batch_size, nm, proto_h, proto_w = proto.shape
        total_object = batch_size * self.max_obj
        masks = det[..., 4 + self.n_classes : 4 + self.n_classes + nm]
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold,self.class_agnostic)
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        
        batch_indices = torch.ones_like(det_indices) * torch.arange(batch_size, device=self.device, dtype=torch.int32).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)
        det_masks = masks[batch_indices, det_indices]


        pooled_proto = TRT_ROIAlign.apply(  proto,
                                            det_boxes.view(total_object, 4),
                                            batch_indices,
                                            1,
                                            1,
                                            self.mask_resolution,
                                            self.mask_resolution,
                                            self.sampling_ratio,
                                            self.pooler_scale
                                        )
        pooled_proto = pooled_proto.view(
            total_object, nm, self.mask_resolution * self.mask_resolution,
        )

        det_masks = (
            torch.matmul(det_masks.unsqueeze(dim=1), pooled_proto)
            .sigmoid()
            .view(batch_size, self.max_obj, self.mask_resolution * self.mask_resolution)
        )

        return num_det, det_boxes, det_scores, det_classes, det_masks
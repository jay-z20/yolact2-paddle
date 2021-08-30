import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle import ParamAttr
#from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict

from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from backbone import construct_backbone

from utils import timer
from utils.functions import MovingAverage, make_net

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
# torch.cuda.current_device()   ###########################

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = 0 # torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

use_jit = False
print("use_jit",use_jit)
ScriptModuleWrapper = nn.Layer #torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = lambda fn, _rcn=None: fn #torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn



class Concat(nn.Layer):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.LayerList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return paddle.concat([net(x) for net in self.nets], axis=1, **self.extra_params)

prior_cache = defaultdict(lambda: None)

class PredictionModule(nn.Layer):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf
    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.
    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim # Defined by Yolact
        self.num_priors  = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index
        self.num_heads   = cfg.num_heads # Defined by Yolact

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            # if cfg.use_prediction_module: # False
            #     self.block = Bottleneck(out_channels, out_channels // 4)
            #     self.conv = nn.Conv2D(out_channels, out_channels, kernel_size=1, bias=True)  ####
            #     self.bn = nn.BatchNorm2D(out_channels)

            self.bbox_layer = nn.Conv2D(out_channels, self.num_priors * 4, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)),                **cfg.head_layer_params)
            self.conf_layer = nn.Conv2D(out_channels, self.num_priors * self.num_classes, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)), **cfg.head_layer_params)
            self.mask_layer = nn.Conv2D(out_channels, self.num_priors * self.mask_dim, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)),    **cfg.head_layer_params)
            
            if cfg.use_mask_scoring: # False
                self.score_layer = nn.Conv2D(out_channels, self.num_priors, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)), **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2D(out_channels, self.num_priors * cfg.num_instance_coeffs, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)), **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.))),
                        nn.ReLU()
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2D(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)))

        self.aspect_ratios = aspect_ratios # [[1,0.5,2]]
        self.scales = scales  # 24

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])
        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.shape[2]
        conv_w = x.shape[3]
        
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        
        if cfg.use_prediction_module: # False
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).transpose((0, 2, 3, 1)).reshape([x.shape[0], -1, 4]) # 1x?x4
        conf = src.conf_layer(conf_x).transpose((0, 2, 3, 1)).reshape([x.shape[0], -1, self.num_classes]) # 1x?x81
        
        if cfg.eval_mask_branch: # True
            mask = src.mask_layer(mask_x).transpose((0, 2, 3, 1)).reshape([x.shape[0], -1, self.mask_dim]) # 1x?x32
        else:
            mask = paddle.zeros([x.shape[0], bbox.size(1), self.mask_dim])

        if cfg.use_mask_scoring:
            score = src.score_layer(x).transpose((0, 2, 3, 1)).reshape([x.shape[0], -1, 1])

        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).transpose((0, 2, 3, 1)).reshape([x.shape[0], -1, cfg.num_instance_coeffs])    

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = F.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = F.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask) # activation_func.tanh

                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).transpose(0, 2, 3, 1).reshape(x.shape[0], -1, self.mask_dim)
                    mask = mask * F.sigmoid(gate)

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode='constant', value=0)
        
        priors = self.make_priors(conv_h, conv_w)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        if cfg.use_mask_scoring:
            preds['score'] = score

        if cfg.use_instance_coeff:
            preds['inst'] = inst
        
        return preds

    def make_priors(self, conv_h, conv_w, device=None):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = (conv_h, conv_w)

        with timer.env('makepriors'):
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for ars in self.aspect_ratios:
                        for scale in self.scales:
                            for ar in ars:
                                if not cfg.backbone.preapply_sqrt:
                                    ar = sqrt(ar)

                                if cfg.backbone.use_pixel_scales:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                                else:
                                    w = scale * ar / conv_w
                                    h = scale / ar / conv_h
                                
                                # This is for backward compatability with a bug where I made everything square by accident
                                if cfg.backbone.use_square_anchors:
                                    h = w

                                prior_data += [x, y, w, h]

                self.priors = paddle.to_tensor(prior_data).astype('float32').reshape([-1, 4])#.detach() # .cuda()
                self.priors.stop_gradient = True
                self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
                self.last_conv_size = (conv_w, conv_h)
                prior_cache[size] = None
            # elif self.priors.device != device:  ##################
            #     # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
            #     if prior_cache[size] is None:
            #         prior_cache[size] = {}
                
            #     if device not in prior_cache[size]:
            #         prior_cache[size][device] = self.priors.to(device)  ###

            #     self.priors = prior_cache[size][device]
        
        return self.priors

class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf
    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.
    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.LayerList([
            nn.Conv2D(x, cfg.fpn.num_features, kernel_size=1, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)))
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.LayerList([
            nn.Conv2D(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)))
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.LayerList([
                nn.Conv2D(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)))
                for _ in range(cfg.fpn.num_downsample)
            ])
        
        self.interpolation_mode     = cfg.fpn.interpolation_mode
        self.num_downsample         = cfg.fpn.num_downsample
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers

    #@script_method_wrapper
    def forward(self, convouts:List[paddle.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = paddle.zeros([1]).cuda()
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].shape
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            
            x = x + lat_layer(convouts[j])
            out[j] = x
        
        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                out[j] = F.relu(out[j])

        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx])

        return out

class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.shape[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p



class Yolact(nn.Layer):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
    You can set the arguments by changing them in the backbone config object in config.py.
    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size**2
        elif cfg.mask_type == mask_type.lincomb:  ## 
            if cfg.mask_proto_use_grid:
                self.grid = paddle.to_tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:   ##
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src
            
            if self.proto_src is None: in_channels = 3
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
            else: in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers  # [1,2,3]
        src_channels = self.backbone.channels # [256,512,1024,2048]

        if cfg.use_maskiou: # False
            self.maskiou_net = FastMaskIoUNet()

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)


        self.prediction_layers = nn.LayerList()
        cfg.num_heads = len(self.selected_layers)

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:  # False
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg.use_semantic_segmentation_loss:  # True
            self.semantic_seg_conv = nn.Conv2D(src_channels[0], cfg.num_classes-1, kernel_size=1, weight_attr=ParamAttr(initializer=XavierUniform()),bias_attr=ParamAttr(initializer=Constant(0.)))

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
            conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        paddle.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        # self.state_dict : 556 
        state_dict = paddle.load(path) # 660
        print("loading model path:",path)
        # for k in state_dict.keys():
        #     if k not in self.state_dict().keys():
        #         print(k)
        
        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.set_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_weights(backbone_path)

    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.children():
            if isinstance(module, nn.BatchNorm2D):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
    
    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.shape
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        
        with timer.env('backbone'): # x: 1x3x550x550
            outs = self.backbone(x) # len:4  4x:1x256x138x138  1x512x69x69 1x1024x35x35  1x2048x18x18

        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers] # selected_layers: 1 2 3
                outs = self.fpn(outs) # len: 5 1x512x69x69  1x256x35x35 1x256x18x18 1x256x9x9 1x256x5x5

        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src] # 1x512x69x69
                
                if self.num_grids > 0: # 0
                    grids = self.grid.repeat((proto_x.shape[0], 1, 1, 1))
                    proto_x = paddle.concat([proto_x, grids], axis=1)

                proto_out = self.proto_net(proto_x) # 1x32x138x138
                proto_out = cfg.mask_proto_prototype_activation(proto_out) # torch.nn.functional.relu

                if cfg.mask_proto_prototypes_as_features: # False
                    # Clone here because we don't want to transpose this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.transpose((0, 2, 3, 1)) # 1x138x138x32

                if cfg.mask_proto_bias: # False
                    bias_shape = [x for x in proto_out.shape]
                    bias_shape[-1] = 1
                    proto_out = paddle.concat([proto_out, paddle.ones(*bias_shape)], -1)


        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

            if cfg.use_mask_scoring:
                pred_outs['score'] = []

            if cfg.use_instance_coeff:
                pred_outs['inst'] = []
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].shape[2:], mode='bilinear', align_corners=False)
                    pred_x = paddle.concat([pred_x, proto_downsampled], axis=1)

                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x)
                
                for k, v in p.items():
                    pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = paddle.concat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        if self.training:
            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            return pred_outs
        else:
            if cfg.use_mask_scoring:
                pred_outs['score'] = F.sigmoid(pred_outs['score'])

            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = F.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = F.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0 ] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            else:

                if cfg.use_objectness_score:
                    objectness = F.sigmoid(pred_outs['conf'][:, :, 0])
                    
                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
                        * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
                    
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            return self.detect(pred_outs, self)




# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    # Use the first argument to set the config if you want
    import sys
    if len(sys.argv) > 1:
        from data.config import set_cfg
        set_cfg(sys.argv[1])

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # GPU
    net = net.cuda()
    paddle.set_default_tensor_type('torch.cuda.FloatTensor')

    x = paddle.zeros((1, 3, cfg.max_size, cfg.max_size))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

    print()
    for k, a in y.items():
        print(k + ': ', a.size(), paddle.sum(a))
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
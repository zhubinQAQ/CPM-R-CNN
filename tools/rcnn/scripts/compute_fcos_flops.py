"""
Only for computing the flops and parameters of RetinaNet and FCOS.
"""

import os
import shutil
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), '../../'))

from pet.utils.net import make_conv
from pet.utils.misc import logging_rank
from pet.utils.measure import measure_model

import pet.rcnn.modeling.backbone
import pet.rcnn.modeling.fpn
from pet.rcnn.modeling.rpn.retinanet.retinanet import RetinaNetModule
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list


# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument("--size", type=int, nargs=2)
parser.add_argument('opts', help='See pet/rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)

logging_rank('Called with args: {}'.format(args))


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone for feature extraction
        conv_body = registry.BACKBONES[cfg.BACKBONE.CONV_BODY]
        self.Conv_Body = conv_body()
        self.dim_in = self.Conv_Body.dim_out
        self.spatial_scale = self.Conv_Body.spatial_scale

        # Feature Pyramid Networks
        if cfg.MODEL.FPN_ON:
            fpn_body = registry.FPN_BODY[cfg.FPN.BODY]
            self.Conv_Body_FPN = fpn_body(self.dim_in, self.spatial_scale)
            self.dim_in = self.Conv_Body_FPN.dim_out
            self.spatial_scale = self.Conv_Body_FPN.spatial_scale
        else:
            self.dim_in = self.dim_in[-1:]
            self.spatial_scale = self.spatial_scale[-1:]

        # Region Proposal Network
        if cfg.MODEL.RETINANET_ON:
            self.RPN = RetinaNetModule(self.dim_in)
        if cfg.MODEL.FCOS_ON:
            self.RPN = FCOSModule(self.dim_in)

    def forward(self, x, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        conv_features = self.Conv_Body(x)

        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        results = self.RPN(conv_features, targets)


@registry.FCOS_HEADS.register("fcos_head")
class FCOSHead(torch.nn.Module):
    def __init__(self, dim_in):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        self.dim_in = dim_in[-1]

        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.NUM_CLASSES - 1
        use_lite = cfg.FCOS.USE_LITE
        use_bn = cfg.FCOS.USE_BN
        use_gn = cfg.FCOS.USE_GN
        use_dcn = cfg.FCOS.USE_DCN
        dense_points = cfg.FCOS.DENSE_POINTS

        self.fpn_strides = cfg.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.FCOS.CENTERNESS_ON_REG

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.FCOS.NUM_CONVS):
            conv_type = 'deform' if use_dcn and i == cfg.FCOS.NUM_CONVS - 1 else 'normal'
            cls_tower.append(
                make_conv(self.dim_in, self.dim_in, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )
            bbox_tower.append(
                make_conv(self.dim_in, self.dim_in, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            self.dim_in, num_classes * dense_points, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            self.dim_in, 4 * dense_points, kernel_size=3, stride=1, padding=1
        )
        self.centerness = nn.Conv2d(
            self.dim_in, 1 * dense_points, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.bbox_pred(box_tower)
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

        return logits, bbox_reg, centerness


@registry.FCOS_HEADS.register("fcoslite_head")
class FCOSHeadLite(torch.nn.Module):
    def __init__(self, dim_in):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHeadLite, self).__init__()
        self.dim_in = dim_in[-1]

        num_classes = cfg.MODEL.NUM_CLASSES - 1
        use_lite = cfg.FCOS.FCOSLITE_HEAD.USE_LITE
        use_bn = cfg.FCOS.FCOSLITE_HEAD.USE_BN
        use_gn = cfg.FCOS.FCOSLITE_HEAD.USE_GN
        tower_conv_kernel = cfg.FCOS.FCOSLITE_HEAD.TOWER_CONV_KERNEL
        last_conv_kernel = cfg.FCOS.FCOSLITE_HEAD.LAST_CONV_KERNEL
        dense_points = cfg.FCOS.DENSE_POINTS

        self.fpn_strides = cfg.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.FCOS.CENTERNESS_ON_REG

        base_tower = []
        for i in range(1):
            base_tower.append(
                make_conv(self.dim_in, self.dim_in, kernel=1, stride=1, dilation=1, use_dwconv=False,
                          conv_type='normal', use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=False)
            )

        cls_tower = []
        for i in range(cfg.FCOS.FCOSLITE_HEAD.CLS_NUM_CONVS):
            cls_tower.append(
                make_conv(self.dim_in, self.dim_in, kernel=tower_conv_kernel, stride=1, dilation=1,
                          use_dwconv=use_lite, conv_type='normal', use_bn=use_bn, use_gn=use_gn, use_relu=True,
                          kaiming_init=False, suffix_1x1=use_lite)
            )

        bbox_tower = []
        for i in range(cfg.FCOS.FCOSLITE_HEAD.BBOX_NUM_CONVS):
            bbox_tower.append(
                make_conv(self.dim_in, self.dim_in, kernel=tower_conv_kernel, stride=1, dilation=1,
                          use_dwconv=use_lite, conv_type='normal', use_bn=use_bn, use_gn=use_gn, use_relu=True,
                          kaiming_init=False, suffix_1x1=use_lite)
            )

        self.add_module('base_tower', nn.Sequential(*base_tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            self.dim_in, num_classes * dense_points, kernel_size=1, stride=1, padding=0
        )
        self.bbox_pred = nn.Conv2d(
            self.dim_in, 4 * dense_points, kernel_size=last_conv_kernel, stride=1, padding=last_conv_kernel // 2
        )
        self.centerness = nn.Conv2d(
            self.dim_in, 1 * dense_points, kernel_size=last_conv_kernel, stride=1, padding=last_conv_kernel // 2
        )

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            feature = self.base_tower(feature)
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.bbox_pred(box_tower)
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    def __init__(self, dim_in):
        super(FCOSModule, self).__init__()

        head = registry.FCOS_HEADS[cfg.FCOS.FCOS_HEAD]
        self.head = head(dim_in)

    def forward(self, features, targets=None):
        return self.head(features)


if __name__ == '__main__':
    model = Generalized_RCNN()
    logging_rank(model)

    model.eval()
    n_flops, n_convops, n_params = measure_model(model, args.size[0], args.size[1])
    logging_rank('FLOPs: {:.4f}M, Conv_FLOPs: {:.4f}M, Params: {:.4f}M'.
                 format(n_flops / 1e6, n_convops / 1e6, n_params / 1e6))


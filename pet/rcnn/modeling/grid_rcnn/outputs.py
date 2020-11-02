import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F

from pet.lib.ops import SeConv2d
from pet.utils.net import make_fc
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


@registry.ROI_GRID_OUTPUTS.register("Grid_output")
class Grid_output(nn.Module):
    def __init__(self, dim_in, stage):
        super(Grid_output, self).__init__()
        self.stage = stage
        self.dim_in = dim_in[-1]
        self.grid_points = cfg.GRID_RCNN.GRID_POINTS if not cfg.GRID_RCNN.CASCADE_MAPPING_ON else \
        cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.GRID_NUM[stage]
        self.point_feat_channels = cfg.GRID_RCNN.GRID_HEAD.POINT_FEAT_CHANNELS
        self.conv_out_channels = self.point_feat_channels * self.grid_points
        deconv_kernel_size = 4
        self.norm1 = nn.GroupNorm(self.grid_points, self.conv_out_channels)
        self.deconv_1 = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=self.grid_points)
        self.deconv_2 = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.grid_points,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=self.grid_points)
        if cfg.GRID_RCNN.IOU_HELPER and self.stage == cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_NUM - 1:
            resolution = cfg.GRID_RCNN.ROI_XFORM_RESOLUTION_CLS
            input_size = self.conv_out_channels * resolution[0] * resolution[1]
            self.iou_fc1 = make_fc(input_size, 1024)
            self.iou_fc2 = make_fc(1024, 1024)
            self.iou_pred = nn.Linear(1024, 2)
            init.normal_(self.iou_pred.weight, std=0.01)
            init.constant_(self.iou_pred.bias, 0)
        if cfg.GRID_RCNN.SE_ON:
            self.se_helper = SeConv2d(self.conv_out_channels, int(self.conv_out_channels * 0.0625))

    def forward(self, x, x_so):
        if cfg.GRID_RCNN.FUSED_ON:
            # predicted heatmap with fused features
            x2 = torch.cat(x_so, dim=1)
            x2 = self.deconv_1(x2)
            x2 = F.relu(self.norm1(x2), inplace=True)
            heatmap = self.deconv_2(x2)
        else:
            if cfg.GRID_RCNN.OFFSET_ON:
                x2 = x + x_so
                x2 = self.deconv_1(x2)
                x2 = F.relu(self.norm1(x2), inplace=True)
                heatmap = self.deconv_2(x2)
            else:
                heatmap = None
        # predicted heatmap with original features (applicable during training)
        if self.training or not cfg.GRID_RCNN.FUSED_ON:
            x1 = x
            x1 = self.deconv_1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            if cfg.GRID_RCNN.SE_ON:
                x1 = self.se_helper(x1)
            heatmap_unfused = self.deconv_2(x1)
        else:
            heatmap_unfused = heatmap

        if cfg.GRID_RCNN.IOU_HELPER and self.stage == cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_NUM - 1:
            x = x.view(x.size(0), -1)
            x = F.relu(self.iou_fc1(x), inplace=True)
            x = F.relu(self.iou_fc2(x), inplace=True)
            iou_logits = self.iou_pred(x)
        else:
            iou_logits = None

        return dict(fused=heatmap, unfused=heatmap_unfused), iou_logits


@registry.ROI_CLS_OUTPUTS.register("Cls_output")
class Cls_output(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.cls_score = nn.Linear(self.dim_in, cfg.MODEL.NUM_CLASSES)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        cls_score = self.cls_score(x)

        return cls_score
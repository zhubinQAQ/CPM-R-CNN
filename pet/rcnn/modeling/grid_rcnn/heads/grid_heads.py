import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from pet.models.imagenet.utils import convert_conv2convws_model
from pet.utils.net import make_conv, make_fc
from pet.rcnn.utils.poolers import Pooler
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


@registry.ROI_GRID_HEADS.register("roi_grid_head")
class roi_grid_head(nn.Module):
    def __init__(self, dim_in, spatial_scale, stage):
        super(roi_grid_head, self).__init__()
        self.grid_points = cfg.GRID_RCNN.GRID_POINTS if not cfg.GRID_RCNN.CASCADE_MAPPING_ON else \
            cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.GRID_NUM[stage]
        self.roi_feat_size = cfg.GRID_RCNN.ROI_FEAT_SIZE

        self.num_convs = cfg.GRID_RCNN.GRID_HEAD.NUM_CONVS
        self.point_feat_channels = cfg.GRID_RCNN.GRID_HEAD.POINT_FEAT_CHANNELS

        self.conv_out_channels = self.point_feat_channels * self.grid_points
        self.class_agnostic = False
        self.dim_in = dim_in[-1]

        assert self.grid_points >= 4
        self.grid_size = int(np.sqrt(self.grid_points))
        if self.grid_size * self.grid_size != self.grid_points:
            raise ValueError('grid_points must be a square number')

        # the predicted heatmap is half of whole_map_size
        if not isinstance(self.roi_feat_size, int):
            raise ValueError('Only square RoIs are supporeted in Grid R-CNN')
        self.whole_map_size = self.roi_feat_size * 4

        self.convs = []
        conv_kernel_size = 3
        for i in range(self.num_convs):
            in_channels = (
                self.dim_in if i == 0 else self.conv_out_channels)
            stride = 2 if i == 0 else 1
            padding = (conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              self.conv_out_channels,
                              kernel_size=conv_kernel_size,
                              stride=stride,
                              padding=padding),
                    nn.GroupNorm(4 * self.grid_points, self.conv_out_channels, eps=1e-5),
                    nn.ReLU(inplace=True)
                )
            )
        self.convs = nn.Sequential(*self.convs)

        # find the 4-neighbor of each grid point
        self.neighbor_points = self._get_neighbors()
        # total edges in the grid
        self.num_edges = sum([len(p) for p in self.neighbor_points])

        if cfg.GRID_RCNN.FUSED_ON:
            self.forder_trans = self._build_trans(nn.ModuleList())  # first-order feature transition
            self.sorder_trans = self._build_trans(nn.ModuleList())  # second-order feature transition

        method = cfg.GRID_RCNN.ROI_XFORM_METHOD
        resolution = cfg.GRID_RCNN.ROI_XFORM_RESOLUTION_GRID
        sampling_ratio = cfg.GRID_RCNN.ROI_XFORM_SAMPLING_RATIO
        spatial_scale = [spatial_scale[0]] if cfg.GRID_RCNN.FINEST_LEVEL_ROI else spatial_scale
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler
        self.dim_out = dim_in

        if cfg.GRID_RCNN.OFFSET_ON:
            self.offset_conv = make_conv(self.dim_in, 64, kernel=3, stride=2)
            self.offset_fc = make_fc(64*7*7, 4 * self.grid_points)

    def _get_neighbors(self):
        neighbor_points = []
        for i in range(self.grid_size):  # i-th column
            for j in range(self.grid_size):  # j-th row
                neighbors = []
                if i > 0:  # left: (i - 1, j)
                    neighbors.append((i - 1) * self.grid_size + j)
                if j > 0:  # up: (i, j - 1)
                    neighbors.append(i * self.grid_size + j - 1)
                if j < self.grid_size - 1:  # down: (i, j + 1)
                    neighbors.append(i * self.grid_size + j + 1)
                if i < self.grid_size - 1:  # right: (i + 1, j)
                    neighbors.append((i + 1) * self.grid_size + j)
                neighbor_points.append(tuple(neighbors))
        return neighbor_points

    def _build_trans(self, trans):
        for neighbors in self.neighbor_points:
            _trans = nn.ModuleList()
            for _ in range(len(neighbors)):
                # each transition module consists of a 5x5 depth-wise conv and
                # 1x1 conv.
                _trans.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.point_feat_channels,
                            self.point_feat_channels,
                            5,
                            stride=1,
                            padding=2,
                            groups=self.point_feat_channels),
                        nn.Conv2d(self.point_feat_channels,
                                  self.point_feat_channels, 1)))
            trans.append(_trans)
        return trans

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # TODO: compare mode = "fan_in" or "fan_out"
                kaiming_init(m)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        nn.init.constant_(self.deconv_2.bias, -np.log(0.99 / 0.01))

    def forward(self, features, proposals):
        x = self.pooler(features, proposals)
        roi_feature = x
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        # RoI feature transformation, downsample 2x
        x = self.convs(x)

        if cfg.GRID_RCNN.FUSED_ON:
            c = self.point_feat_channels
            # first-order fusion
            x_fo = [None for _ in range(self.grid_points)]
            for i, points in enumerate(self.neighbor_points):
                x_fo[i] = x[:, i * c:(i + 1) * c]
                for j, point_idx in enumerate(points):
                    x_fo[i] = x_fo[i] + self.forder_trans[i][j](
                        x[:, point_idx * c:(point_idx + 1) * c])

            # second-order fusion
            x_so = [None for _ in range(self.grid_points)]
            for i, points in enumerate(self.neighbor_points):
                x_so[i] = x[:, i * c:(i + 1) * c]
                for j, point_idx in enumerate(points):
                    x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])
            return x, x_so
        else:
            if cfg.GRID_RCNN.OFFSET_ON:
                x_offset = self.offset_conv(roi_feature)
                x_offset = self.offset_fc(x_offset)
                return x, x_offset
            else:
                return x, None
import torch
import torch.nn as nn
import torch.nn.functional as F

from pet.models.imagenet.utils import convert_conv2convws_model
from pet.utils.net import make_fc
from pet.rcnn.utils.poolers import Pooler
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


@registry.ROI_CLS_HEADS.register("roi_cls_head")
class roi_cls_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""

    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]

        method = cfg.GRID_RCNN.ROI_XFORM_METHOD
        resolution = cfg.GRID_RCNN.ROI_XFORM_RESOLUTION_CLS
        sampling_ratio = cfg.GRID_RCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        input_size = self.dim_in * resolution[0] * resolution[1]
        mlp_dim = cfg.GRID_RCNN.MLP_HEAD.MLP_DIM
        use_bn = cfg.GRID_RCNN.MLP_HEAD.USE_BN
        use_gn = cfg.GRID_RCNN.MLP_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, mlp_dim, use_bn, use_gn)
        self.fc7 = make_fc(mlp_dim, mlp_dim, use_bn, use_gn)
        self.dim_out = mlp_dim

        if cfg.GRID_RCNN.MLP_HEAD.USE_WS:
            self = convert_conv2convws_model(self)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

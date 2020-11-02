import torch
import torch.nn as nn
import torch.nn.functional as F

from pet.utils.net import make_fc
from pet.rcnn.utils.poolers import Pooler
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


@registry.ROI_CASCADE_HEADS.register("semseg_fused_box_head")
class semseg_fused_box_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""

    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]
        
        semseg_method = cfg.SRCNN.ROI_XFORM_METHOD
        semseg_resolution = cfg.SRCNN.ROI_XFORM_RESOLUTION
        semseg_sampling_ratio = cfg.SRCNN.ROI_XFORM_SAMPLING_RATIO
        semseg_pooler = Pooler(
            method=semseg_method,
            output_size=semseg_resolution,
            scales=[spatial_scale[1]],
            sampling_ratio=semseg_sampling_ratio,
        )

        method = cfg.FAST_RCNN.ROI_XFORM_METHOD
        resolution = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        
        input_size = self.dim_in * resolution[0] * resolution[1]
        mlp_dim = cfg.FAST_RCNN.MLP_HEAD.MLP_DIM
        use_bn = cfg.FAST_RCNN.MLP_HEAD.USE_BN
        use_gn = cfg.FAST_RCNN.MLP_HEAD.USE_GN
        self.pooler = pooler
        self.semseg_pooler = semseg_pooler
        self.fc6 = make_fc(input_size, mlp_dim, use_bn, use_gn)
        self.fc7 = make_fc(mlp_dim, mlp_dim, use_bn, use_gn)
        self.dim_out = mlp_dim

    def forward(self, feats, proposals):
        assert isinstance(feats, tuple) or 'semantic segmentation fused box head input feature type error!'
        
        x, semseg_feats = feats
        x = self.pooler(x, proposals)
        bbox_semseg_feat = self.semseg_pooler([semseg_feats], proposals)
        if bbox_semseg_feat.shape[-2:] != x.shape[-2:]:
            bbox_semseg_feat = F.adaptive_avg_pool2d(bbox_semseg_feat, x.shape[-2:])
        x = x + bbox_semseg_feat
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)

        return x

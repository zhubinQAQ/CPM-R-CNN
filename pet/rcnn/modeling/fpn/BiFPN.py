import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pet.utils.net import make_conv
from pet.rcnn.core.config import cfg
from pet.rcnn.modeling.fpn import get_min_max_levels
from pet.rcnn.modeling import registry


def resize(x, size):
    if x.shape[-2:] == size:
        return x
    elif x.shape[-2:] < size:
        return F.interpolate(x, size=size, mode='nearest')
    else:
        assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
        kernel_size = (math.ceil(x.shape[-2] / size[-2]), math.ceil(x.shape[-1] / size[-1]))
        x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size, ceil_mode=True)
        return x


class Fusion2D(nn.Module):
    def __init__(self, init_value=0.5, eps=1e-4):
        super(Fusion2D, self).__init__()
        self.eps = eps
        self.w1 = nn.Parameter(torch.FloatTensor([init_value]))
        self.w2 = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x1, x2):
        return (x1 * self.w1 + x2 * self.w2) / (self.w1 + self.w2 + self.eps)


class Fusion3D(nn.Module):
    def __init__(self, init_value=0.333, eps=1e-4):
        super(Fusion3D, self).__init__()
        self.eps = eps
        self.w1 = nn.Parameter(torch.FloatTensor([init_value]))
        self.w2 = nn.Parameter(torch.FloatTensor([init_value]))
        self.w3 = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x1, x2, x3):
        return (x1 * self.w1 + x2 * self.w2 + x3 + self.w3) / (self.w1 + self.w2 + self.w3 + self.eps)


# ---------------------------------------------------------------------------- #
# Functions for bolting BiFPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
@registry.FPN_BODY.register("bifpn")
class bifpn(nn.Module):
    # dim_in = [256, 512, 1024, 2048]
    # spatial_scale = [1/4, 1/8, 1/16, 1/32]
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in[-1]  # 2048
        self.spatial_scale = spatial_scale

        self.num_stack = cfg.FPN.BIFPN.NUM_STACK
        bifpn_dim = cfg.FPN.BIFPN.DIM
        self.eps = cfg.FPN.BIFPN.EPS
        use_lite = cfg.FPN.BIFPN.USE_LITE
        use_bn = cfg.FPN.BIFPN.USE_BN
        use_gn = cfg.FPN.BIFPN.USE_GN
        min_level, max_level = get_min_max_levels()  # 3, 7
        self.num_backbone_stages = len(dim_in) - (
                min_level - cfg.FPN.LOWEST_BACKBONE_LVL)  # 3 (cfg.FPN.LOWEST_BACKBONE_LVL=2)

        # bifpn module
        self.bifpn_in = nn.ModuleList()
        for i in range(self.num_backbone_stages):
            px_in = make_conv(dim_in[-1 - i], bifpn_dim, kernel=1, use_bn=use_bn, use_gn=use_gn)
            self.bifpn_in.append(px_in)
        self.dim_in = bifpn_dim

        # add bifpn connections
        self.bifpn_stages = nn.ModuleList()
        for _ in range(self.num_stack):
            stage = nn.ModuleDict()

            # fusion weights
            stage['p6_td_fusion'] = Fusion2D()
            stage['p5_td_fusion'] = Fusion2D()
            stage['p4_td_fusion'] = Fusion2D()
            stage['p3_out_fusion'] = Fusion2D()
            stage['p4_out_fusion'] = Fusion3D()
            stage['p5_out_fusion'] = Fusion3D()
            stage['p6_out_fusion'] = Fusion3D()
            stage['p7_out_fusion'] = Fusion2D()

            # top-down connect
            stage['p6_td_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                            use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            stage['p5_td_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                            use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            stage['p4_td_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                            use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)

            # output
            stage['p3_out_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                             use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            stage['p4_out_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                             use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            stage['p5_out_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                             use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            stage['p6_out_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                             use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            stage['p7_out_conv'] = make_conv(bifpn_dim, bifpn_dim, kernel=3, use_dwconv=use_lite, use_bn=use_bn,
                                             use_gn=use_gn, use_relu=use_bn or use_gn, suffix_1x1=use_lite)
            self.bifpn_stages.append(stage)

        self.extra_levels = max_level - cfg.FPN.HIGHEST_BACKBONE_LVL  # 2
        for _ in range(self.extra_levels):
            self.spatial_scale.append(self.spatial_scale[-1] * 0.5)

        self.spatial_scale = self.spatial_scale[min_level - 2:]
        self.dim_out = [self.dim_in for _ in range(max_level - min_level + 1)]

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        px_outs = []
        for i in range(self.num_backbone_stages):  # [P5 - P3]
            px = self.bifpn_in[i](x[-i - 1])
            px_outs.append(px)

        for _ in range(self.extra_levels):  # P6, P7
            px_outs.insert(0, F.max_pool2d(px_outs[0], 2, stride=2))

        p7, p6, p5, p4, p3 = px_outs  # or: [P6 - P2]
        p7_shape, p6_shape, p5_shape, p4_shape, p3_shape = \
            p7.shape[2:], p6.shape[2:], p5.shape[2:], p4.shape[2:], p3.shape[2:]
        for stage in self.bifpn_stages:
            p6_td = stage['p6_td_conv'](stage['p6_td_fusion'](p6, resize(p7, p6_shape)))
            p5_td = stage['p5_td_conv'](stage['p5_td_fusion'](p5, resize(p6_td, p5_shape)))
            p4_td = stage['p4_td_conv'](stage['p4_td_fusion'](p4, resize(p5_td, p4_shape)))

            p3 = stage['p3_out_conv'](stage['p3_out_fusion'](p3, resize(p4_td, p3_shape)))
            p4 = stage['p4_out_conv'](stage['p4_out_fusion'](p4, p4_td, resize(p3, p4_shape)))
            p5 = stage['p5_out_conv'](stage['p5_out_fusion'](p5, p5_td, resize(p4, p5_shape)))
            p6 = stage['p6_out_conv'](stage['p6_out_fusion'](p6, p6_td, resize(p5, p6_shape)))
            p7 = stage['p7_out_conv'](stage['p7_out_fusion'](p7, resize(p6, p7_shape)))

        return [p3, p4, p5, p6, p7]  # [P3 - P7]

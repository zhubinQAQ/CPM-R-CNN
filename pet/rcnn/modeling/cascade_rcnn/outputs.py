import torch.nn as nn
import torch.nn.init as init

from torch.nn import functional as F

from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


# ---------------------------------------------------------------------------- #
# R-CNN bbox branch outputs
# ---------------------------------------------------------------------------- #
@registry.ROI_CASCADE_OUTPUTS.register("Box_output")
class Box_output(nn.Module):
    def __init__(self, dim_in, stage):
        super().__init__()
        self.stage = stage
        self.dim_in = dim_in

        self.cls_score = nn.Linear(self.dim_in, cfg.MODEL.NUM_CLASSES)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(self.dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(self.dim_in, 4 * cfg.MODEL.NUM_CLASSES)
        
        if cfg.CASCADE_RCNN.IOU_HELPER and self.stage == cfg.CASCADE_RCNN.NUM_STAGE - 1:
            self.iou_fc1 = nn.Linear(self.dim_in, 1024)
            self.iou_fc2 = nn.Linear(1024, 1024)
            self.iou_pred = nn.Linear(1024, 2)
            init.normal_(self.iou_pred.weight, std=0.01)
            init.constant_(self.iou_pred.bias, 0)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        if cfg.CASCADE_RCNN.IOU_HELPER and self.stage == cfg.CASCADE_RCNN.NUM_STAGE - 1:
            x = F.relu(self.iou_fc1(x), inplace=True)
            x = F.relu(self.iou_fc2(x), inplace=True)
            iou_logits = self.iou_pred(x)
        else:
            iou_logits = None

        return cls_score, bbox_pred, iou_logits

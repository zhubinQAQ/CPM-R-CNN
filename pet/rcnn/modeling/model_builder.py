import numpy as np

import torch
import torch.nn as nn

from pet.utils.data.structures.image_list import to_image_list
import pet.lib.ops as ops
import pet.rcnn.modeling.backbone
import pet.rcnn.modeling.fpn
from pet.rcnn.modeling.rpn.rpn import build_rpn
from pet.rcnn.modeling.fast_rcnn.fast_rcnn import FastRCNN
from pet.rcnn.modeling.grid_rcnn.grid_rcnn import GridRCNN
from pet.rcnn.modeling.grid_cascade_rcnn.grid_cascade_rcnn import GridCascadeRCNN
from pet.rcnn.modeling.cascade_rcnn.cascade_rcnn import CascadeRCNN
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


class Generalized_RCNN(nn.Module):
    def __init__(self, is_train=True):
        super().__init__()

        # Normalization
        if not is_train:
            self.Norm = ops.AffineChannel2d(3)
            self.Norm.weight.data = torch.from_numpy(1. / np.array(cfg.PIXEL_STDS)).float()
            self.Norm.bias.data = torch.from_numpy(-1. * np.array(cfg.PIXEL_MEANS) /
                                                   np.array(cfg.PIXEL_STDS)).float()

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
        self.RPN = build_rpn(self.dim_in)

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.FASTER_RCNN:
                if cfg.MODEL.CASCADE_ON:
                    self.Cascade_RCNN = CascadeRCNN(self.dim_in, self.spatial_scale)
                else:
                    self.Fast_RCNN = FastRCNN(self.dim_in, self.spatial_scale)
            elif cfg.MODEL.GRID_ON:
                if cfg.GRID_RCNN.CASCADE_MAPPING_ON:
                    self.Grid_Cascade_RCNN = GridCascadeRCNN(self.dim_in, self.spatial_scale)
                else:
                    self.Grid_RCNN = GridRCNN(self.dim_in, self.spatial_scale)

        self._init_modules()

    def _init_modules(self):
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False
            if cfg.MODEL.FPN_ON:
                for p in self.Conv_Body_FPN.parameters():
                    p.requires_grad = False

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        conv_features = self.Conv_Body(images.tensors)

        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        proposals, proposal_losses = self.RPN(images, conv_features, targets)

        if cfg.MODEL.SEMSEG_ON:
            semseg_losses = {}
            x, semseg_features, loss_semseg = self.SemSeg_RCNN(conv_features, targets)
            conv_features = (conv_features, semseg_features)
            semseg_losses.update(loss_semseg)
        else:
            semseg_losses = {}

        if not cfg.MODEL.RPN_ONLY:
            roi_losses = {}

            if cfg.MODEL.FASTER_RCNN:
                if cfg.MODEL.CASCADE_ON:
                    box_features, result, loss_box = self.Cascade_RCNN(conv_features, proposals, targets)
                else:
                    box_features, result, loss_box = self.Fast_RCNN(conv_features, proposals, targets)
                roi_losses.update(loss_box)
            elif cfg.MODEL.GRID_ON:
                if cfg.GRID_RCNN.CASCADE_MAPPING_ON:
                    grid_features, result, loss_grid = self.Grid_Cascade_RCNN(conv_features, proposals, targets)
                else:
                    grid_features, result, loss_grid = self.Grid_RCNN(conv_features, proposals, targets)
                roi_losses.update(loss_grid)

            if cfg.MODEL.MASK_ON:
                if not cfg.MRCNN.MASKIOU_ON:
                    x, result, loss_mask = self.Mask_RCNN(conv_features, result, targets)
                    roi_losses.update(loss_mask)
                else:
                    x, result, loss_mask, roi_feature, selected_mask, labels, maskiou_targets = self.Mask_RCNN(
                        conv_features, result, targets)
                    roi_losses.update(loss_mask)

                    loss_maskiou, result = self.MaskIoU_RCNN(roi_feature, result, selected_mask, labels,
                                                             maskiou_targets)
                    roi_losses.update(loss_maskiou)

            if cfg.MODEL.KEYPOINT_ON:
                x, result, loss_keypoint = self.Keypoint_RCNN(conv_features, result, targets)
                roi_losses.update(loss_keypoint)

            if cfg.MODEL.PARSING_ON:
                x, result, loss_parsing = self.Parsing_RCNN(conv_features, result, targets)
                roi_losses.update(loss_parsing)

            if cfg.MODEL.UV_ON:
                x, result, loss_uv = self.UV_RCNN(conv_features, result, targets)
                roi_losses.update(loss_uv)
        else:
            # RPN-only models don't have roi_heads
            x = conv_features
            result = proposals
            roi_losses = {}

        if self.training:
            outputs = {}
            outputs['metrics'] = {}
            outputs['losses'] = {}
            outputs['losses'].update(proposal_losses)
            outputs['losses'].update(semseg_losses)
            outputs['losses'].update(roi_losses)
            return outputs

        return result

    def box_net(self, images, targets=None):
        # _images = images
        images = to_image_list(images, cfg.TEST.SIZE_DIVISIBILITY)
        images_norm = self.Norm(images.tensors)
        conv_features = self.Conv_Body(images_norm)

        if cfg.MODEL.FPN_ON:
            conv_features = self.Conv_Body_FPN(conv_features)
        else:
            conv_features = [conv_features[-1]]

        proposals, proposal_losses = self.RPN(images, conv_features, targets)

        if cfg.MODEL.SEMSEG_ON:
            x, semseg_features, loss_semseg = self.SemSeg_RCNN(conv_features, targets)
            conv_features = (conv_features, semseg_features)

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.FASTER_RCNN:
                if cfg.MODEL.CASCADE_ON:
                    box_features, result, loss_box = self.Cascade_RCNN(conv_features, proposals, targets)
                else:
                    box_features, result, loss_box = self.Fast_RCNN(conv_features, proposals, targets)
            elif cfg.MODEL.GRID_ON:
                if cfg.GRID_RCNN.CASCADE_MAPPING_ON:
                    grid_features, result, loss_grid = self.Grid_Cascade_RCNN(conv_features, proposals, targets)
                    # print(grid_features.shape, [image.shape for image in _images])
                    # print(zhubin)
                else:
                    grid_features, result, loss_grid = self.Grid_RCNN(conv_features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals

        return conv_features, result

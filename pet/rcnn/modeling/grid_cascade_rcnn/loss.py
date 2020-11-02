import torch
from torch.nn import functional as F

from pet.lib.ops import smooth_l1_loss, l2_loss
from pet.utils.data.structures.boxlist_ops import boxlist_iou
from pet.rcnn.utils.misc import cat
from pet.rcnn.utils.matcher import Matcher
from pet.rcnn.utils.box_coder import BoxCoder
from pet.rcnn.utils.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from pet.rcnn.utils.misc import cat
from pet.rcnn.core.config import cfg

import numpy as np


class CLSLossComputation(object):

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            cls_agnostic_bbox_reg=False
    ):
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            labels.append(labels_per_image)

        return labels

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, proposals_per_image in zip(
                labels, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits):
        class_logits = cat(class_logits, dim=0)
        # device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)

        return classification_loss


class GridLossComputation(object):
    def __init__(
            self,
            stage,
            loss_weight,
            proposal_matcher,
            pos_radius,
            grid_points,
            roi_feat_size,
    ):
        self.stage = stage
        self.loss_weight = loss_weight
        self.proposal_matcher = proposal_matcher
        self.pos_radius = pos_radius
        self.grid_points = grid_points
        self.roi_feat_size = roi_feat_size
        self.whole_map_size = self.roi_feat_size * 4
        self.grid_size = int(np.sqrt(self.grid_points))
        self.sub_regions = calc_sub_regions(grid_points, self.grid_size, self.whole_map_size)

    def subsample(self, proposals, targets):
        bboxes = []
        gt_bboxes = []
        new_proposals = []
        match_quality_matrixs = []
        for proposal, target in zip(proposals, targets):
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            pos_idxs = matched_idxs >= 0
            if cfg.GRID_RCNN.IOU_HELPER:
                match_quality_matrixs.append(match_quality_matrix[:, pos_idxs])
            # Mask RCNN needs "labels" and "masks "fields for creating the targets
            target = target.copy_with_fields(["labels"])
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = target[matched_idxs.clamp(min=0)]

            if self.stage != 0:
                proposal = proposal[pos_idxs]
                matched_targets = matched_targets[pos_idxs]
            new_proposals.append(proposal)
            bboxes.append(proposal.bbox)
            gt_bboxes.append(matched_targets.bbox)
        if cfg.GRID_RCNN.BETTER_ROI:
            bboxes, gt_bboxes, new_proposals = select_better_roi(bboxes, gt_bboxes, new_proposals)
        pos_bboxes = torch.cat([bbox for bbox in bboxes], dim=0).cpu()
        pos_gt_bboxes = torch.cat([bbox for bbox in gt_bboxes], dim=0).cpu()
        self.pos_result = (pos_bboxes, pos_gt_bboxes)
        self.match_quality_matrixs = match_quality_matrixs
        return new_proposals

    def prepare_iou_target(self):
        iou_targets = []
        for match_quality_matrix in self.match_quality_matrixs:
            fg_iou, ind = match_quality_matrix.max(dim=0)
            fg_iou = fg_iou.unsqueeze(1)
            bg_iou = 1 - fg_iou
            iou = torch.cat([bg_iou, fg_iou], dim=1)
            iou_targets.append(iou)
        iou_targets = torch.cat(iou_targets)
        iou_targets = iou_targets.cuda()
        return iou_targets

    def prepare_target(self, proposals, targets):
        # mix all samples (across images) together.
        # new target box & new proposals should contain all roi in a batch
        # pos_bboxes, pos_gt_bboxes = self.match_targets_to_proposals(proposals, targets)
        pos_bboxes, pos_gt_bboxes = self.pos_result
        assert pos_bboxes.shape == pos_gt_bboxes.shape

        # expand pos_bboxes to 2x of original size
        mapping_ratio = cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_MAPPING_RATIO[self.stage]
        x1 = pos_bboxes[:, 0] - mapping_ratio * ((pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2)
        y1 = pos_bboxes[:, 1] - mapping_ratio * ((pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2)
        x2 = pos_bboxes[:, 2] + mapping_ratio * ((pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2)
        y2 = pos_bboxes[:, 3] + mapping_ratio * ((pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2)
        pos_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        pos_bbox_ws = (pos_bboxes[:, 2] - pos_bboxes[:, 0]).unsqueeze(-1)
        pos_bbox_hs = (pos_bboxes[:, 3] - pos_bboxes[:, 1]).unsqueeze(-1)

        num_rois = pos_bboxes.shape[0]
        map_size = self.whole_map_size
        # this is not the final target shape
        targets = torch.zeros((num_rois, self.grid_points, map_size, map_size),
                              dtype=torch.float)

        # pre-compute interpolation factors for all grid points.
        # the first item is the factor of x-dim, and the second is y-dim.
        # for a 9-point grid, factors are like (1, 0), (0.5, 0.5), (0, 1)
        factors = []
        for j in range(self.grid_points):
            x_idx = j // self.grid_size
            y_idx = j % self.grid_size
            factors.append((1 - x_idx / (self.grid_size - 1),
                            1 - y_idx / (self.grid_size - 1)))

        radius = self.pos_radius
        radius2 = radius ** 2
        for i in range(num_rois):
            # ignore small bboxes
            if (pos_bbox_ws[i] <= self.grid_size
                    or pos_bbox_hs[i] <= self.grid_size):
                continue
            # for each grid point, mark a small circle as positive
            for j in range(self.grid_points):
                factor_x, factor_y = factors[j]
                gridpoint_x = factor_x * pos_gt_bboxes[i, 0] + (
                        1 - factor_x) * pos_gt_bboxes[i, 2]
                gridpoint_y = factor_y * pos_gt_bboxes[i, 1] + (
                        1 - factor_y) * pos_gt_bboxes[i, 3]

                cx = int((gridpoint_x - pos_bboxes[i, 0]) / pos_bbox_ws[i] *
                         map_size)
                cy = int((gridpoint_y - pos_bboxes[i, 1]) / pos_bbox_hs[i] *
                         map_size)

                for x in range(cx - radius, cx + radius + 1):
                    for y in range(cy - radius, cy + radius + 1):
                        if x >= 0 and x < map_size and y >= 0 and y < map_size:
                            if (x - cx) ** 2 + (y - cy) ** 2 <= radius2:
                                targets[i, j, y, x] = 1
                if cfg.GRID_RCNN.TARGET_REFINE:
                    if cx < 0 or cx >= map_size or cy < 0 or cy >= map_size:
                        print(cx, cy)
                        x = cx
                        y = cy
                        if cx < 0:
                            x = 0
                        if cx >= map_size:
                            x = 55
                        if cy < 0:
                            y = 0
                        if cy >= map_size:
                            y = 55
                        targets[i, j, y, x] = 1
        # reduce the target heatmap size by a half
        # proposed in Grid R-CNN Plus (https://arxiv.org/abs/1906.05688).
        sub_targets = []
        for i in range(self.grid_points):
            sub_x1, sub_y1, sub_x2, sub_y2 = self.sub_regions[i]
            sub_targets.append(targets[:, [i], sub_y1:sub_y2, sub_x1:sub_x2])
        sub_targets = torch.cat(sub_targets, dim=1)
        sub_targets = sub_targets.cuda()
        return sub_targets

    def loss_grid(self, grid_logit, grid_target):
        return self.loss_weight * F.binary_cross_entropy_with_logits(grid_logit, grid_target.float())

    def __call__(self, proposals, grid_logits, iou_logits, targets):
        grid_targets = self.prepare_target(proposals, targets)
        if cfg.GRID_RCNN.FUSED_ON:
            loss_fused = self.loss_grid(grid_logits['fused'], grid_targets)
        else:
            loss_fused = 0
        loss_unfused = self.loss_grid(grid_logits['unfused'], grid_targets)
        loss_grid = loss_fused + loss_unfused
        if cfg.GRID_RCNN.IOU_HELPER and self.stage == cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_NUM - 1:
            iou_targets = self.prepare_iou_target()
            loss_iou = l2_loss(iou_logits, iou_targets)
        else:
            loss_iou = 0
        return loss_grid, loss_iou


def calc_sub_regions(grid_points, grid_size, whole_map_size):
    """Compute point specific representation regions.

    See Grid R-CNN Plus (https://arxiv.org/abs/1906.05688) for details.
    """
    # to make it consistent with the original implementation, half_size
    # is computed as 2 * quarter_size, which is smaller
    half_size = whole_map_size // 4 * 2
    sub_regions = []
    for i in range(grid_points):
        x_idx = i // grid_size
        y_idx = i % grid_size
        if x_idx == 0:
            sub_x1 = 0
        elif x_idx == grid_size - 1:
            sub_x1 = half_size
        else:
            ratio = x_idx / (grid_size - 1) - 0.25
            sub_x1 = max(int(ratio * whole_map_size), 0)

        if y_idx == 0:
            sub_y1 = 0
        elif y_idx == grid_size - 1:
            sub_y1 = half_size
        else:
            ratio = y_idx / (grid_size - 1) - 0.25
            sub_y1 = max(int(ratio * whole_map_size), 0)
        sub_regions.append(
            (sub_x1, sub_y1, sub_x1 + half_size, sub_y1 + half_size))
    return sub_regions


def select_better_roi(bboxes, gt_bboxes, new_proposals):
    assert isinstance(bboxes, list)
    assert isinstance(gt_bboxes, list)
    _bboxes = []
    _gt_bboxes = []
    _new_proposals = []
    ratio = cfg.GRID_RCNN.BETTER_ROI_RATIO
    for box_per_img, gt_per_img, new_proposals_per_img in zip(bboxes, gt_bboxes, new_proposals):
        c = lambda x: [x[:, 0] + 0.5*(x[:, 2] - x[:, 0]), x[:, 1] + 0.5 * (x[:, 3] - x[:, 1])]
        box_per_img_center = c(box_per_img)
        gt_per_img_center = c(gt_per_img)
        dist = (gt_per_img_center[0] - box_per_img_center[0])**2 + (gt_per_img_center[1] - box_per_img_center[1])**2
        max_dist = (ratio*(gt_per_img[:, 2] - gt_per_img[:, 0]))**2 + (ratio*(gt_per_img[:, 3] - gt_per_img[:, 1]))**2
        ind = (max_dist-dist) >= 0
        _bboxes.append(box_per_img[ind, :])
        _gt_bboxes.append(gt_per_img[ind, :])
        _new_proposals.append(new_proposals_per_img[ind])
    return _bboxes, _gt_bboxes, _new_proposals


def loss_evaluator(stage=0, type=None):
    if type == 'cls':
        matcher = Matcher(
            cfg.GRID_RCNN.FG_IOU_THRESHOLD,
            cfg.GRID_RCNN.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )
        fg_bg_sampler = BalancedPositiveNegativeSampler(
            cfg.GRID_RCNN.BATCH_SIZE_PER_IMAGE, cfg.GRID_RCNN.POSITIVE_FRACTION
        )
        cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        evaluator = CLSLossComputation(
            matcher,
            fg_bg_sampler,
            cls_agnostic_bbox_reg
        )
    elif type == 'grid':
        grid_matcher = Matcher(
            cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.FG_IOU_THRESHOLD[stage],
            cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.BG_IOU_THRESHOLD[stage],
            allow_low_quality_matches=False,
        )
        loss_weight = cfg.GRID_RCNN.LOSS_WEIGHT
        pos_radius = cfg.GRID_RCNN.POS_RADIUS
        grid_points = cfg.GRID_RCNN.GRID_POINTS if not cfg.GRID_RCNN.CASCADE_MAPPING_ON else \
        cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.GRID_NUM[stage]
        roi_feat_size = cfg.GRID_RCNN.ROI_FEAT_SIZE

        evaluator = GridLossComputation(
            stage,
            loss_weight,
            grid_matcher,
            pos_radius,
            grid_points,
            roi_feat_size,
        )
    else:
        raise Exception('Type error!')
    return evaluator

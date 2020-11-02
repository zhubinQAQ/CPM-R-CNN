import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from pet.utils.data.structures.bounding_box import BoxList
from pet.lib.ops.boxlist_ops import boxlist_nms, boxlist_soft_nms, boxlist_box_voting, boxlist_ml_nms
from pet.utils.data.structures.boxlist_ops import cat_boxlist
from pet.rcnn.utils.box_coder import BoxCoder
from pet.rcnn.core.config import cfg
from pet.rcnn.modeling.grid_rcnn.loss import calc_sub_regions


def resize_boxes(bbox):
    assert isinstance(bbox, torch.Tensor)
    thresh = 96 ** 2
    delta_ratio = 0.7
    s = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    l_ind = s > thresh
    _bbox = bbox
    delta_x = delta_ratio * 0.5 * (_bbox[:, 2] - _bbox[:, 0])
    delta_y = delta_ratio * 0.5 * (_bbox[:, 3] - _bbox[:, 1])
    _bbox[:, 0] = _bbox[:, 0] + delta_x
    _bbox[:, 1] = _bbox[:, 1] + delta_y
    _bbox[:, 2] = _bbox[:, 2] - delta_x
    _bbox[:, 3] = _bbox[:, 3] - delta_y
    bbox[l_ind, :] = _bbox[l_ind, :]
    return bbox


class CLSPostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            score_thresh,
            nms,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(CLSPostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms

        self.bf_npy = []
        self.mid_npy = []
        self.af_npy = []

    def forward(self, x, boxes, rescore=False):
        class_logits = x
        class_prob = F.softmax(class_logits, -1)
        if rescore:
            weighted_score = True
            for boxes_per_image in boxes:
                rescores = class_prob[torch.arange(class_prob.shape[0]), boxes_per_image.get_field("labels")]
                if weighted_score:
                    weight = (0.8, 0.2)
                    weight_type = 'pow'
                    if weight_type == 'pow':
                        rescores = (boxes_per_image.get_field("scores") ** weight[0]) * (rescores ** weight[1])
                    else:
                        # print(boxes_per_image.get_field("scores"), rescores)
                        rescores = boxes_per_image.get_field("scores") * rescores
                        # rescores = boxes_per_image.get_field("scores")
                boxes_per_image.add_field("scores", rescores)
            return boxes

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = concat_boxes.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
                class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes, rescore)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes, rescore):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # multiclass nms
        scores = boxlist.get_field("scores")
        device = scores.device
        num_repeat = int(boxlist.bbox.shape[0] / num_classes)
        labels = np.tile(np.arange(num_classes), num_repeat)
        boxlist.add_field("labels", torch.from_numpy(labels).to(dtype=torch.int64, device=device))
        fg_labels = torch.from_numpy(
            (np.arange(boxlist.bbox.shape[0]) % num_classes != 0).astype(int)
        ).to(dtype=torch.bool, device=device)
        _scores = scores > self.score_thresh
        inds_all = _scores & fg_labels
        result = boxlist_ml_nms(boxlist[inds_all], self.nms)

        return result


class GridPostProcessor(nn.Module):
    def __init__(
            self,
            stage,
            grid_points,
            roi_feat_size,
            nms_on=True,
    ):
        super(GridPostProcessor, self).__init__()
        self.stage = stage
        self.grid_points = grid_points
        self.roi_feat_size = roi_feat_size
        self.whole_map_size = self.roi_feat_size * 4
        self.grid_size = int(np.sqrt(self.grid_points))
        self.sub_regions = calc_sub_regions(grid_points, self.grid_size, self.whole_map_size)

        self.nms_on = nms_on

    def forward(self, grid_logits, proposals, iou_logits=None, is_train=False, targets=None):
        grid_pred = grid_logits['fused'] if cfg.GRID_RCNN.FUSED_ON else grid_logits['unfused']
        refine_proposals = []
        if is_train:
            for per_proposals, target in zip(proposals, targets):
                box_num = per_proposals.bbox.shape[0]
                # get the gt location
                keep = self._filter_boxes(per_proposals, target)
                per_proposals = per_proposals[keep]

                # get the predict box
                if per_proposals.bbox.shape[0] != 0:
                    per_pred = grid_pred[:box_num][keep]
                    result_box = self.get_boxes(per_proposals, per_pred, is_train)
                    if cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.RESIZE_ROI:
                        result_box = resize_boxes(result_box)
                    per_proposals.bbox = result_box

                # add gt box to result box
                per_proposals = self.add_gt_proposals(per_proposals, target)
                refine_proposals.append(per_proposals)
                grid_pred = grid_pred[box_num:]
        else:
            for per_proposals in proposals:
                box_num = per_proposals.bbox.shape[0]
                per_pred = grid_pred[:box_num]
                result_box = self.get_boxes(per_proposals, per_pred, is_train)
                if cfg.GRID_RCNN.IOU_HELPER and self.stage == cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_NUM - 1:
                    score = per_proposals.get_field("scores")
                    iou_score = iou_logits[:, 1]
                    # ind = iou_score > cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.FG_IOU_THRESHOLD[self.stage]
                    # result_box = result_box[ind]
                    assert score.shape == iou_score.shape
                    if cfg.GRID_RCNN.IOU_HELPER_MERGE:
                        score = score * iou_score
                    else:
                        score = iou_score
                    per_proposals.add_field("scores", score)
                per_proposals.bbox = result_box
                refine_proposals.append(per_proposals)
                grid_pred = grid_pred[box_num:]

        return refine_proposals

    def get_boxes(self, proposals, grid_pred, is_train):
        det_bboxes = proposals.bbox
        # TODO: refactoring
        assert det_bboxes.shape[0] > 0
        assert det_bboxes.shape[0] == grid_pred.shape[0]
        device = det_bboxes.get_device()
        det_bboxes = det_bboxes.cpu()
        grid_pred = grid_pred.sigmoid().cpu()

        R, c, h, w = grid_pred.shape
        half_size = self.whole_map_size // 4 * 2
        assert h == w == half_size
        assert c == self.grid_points

        # find the point with max scores in the half-sized heatmap
        grid_pred = grid_pred.view(R * c, h * w)
        pred_scores, pred_position = grid_pred.max(dim=1)
        xs = pred_position % w
        ys = pred_position // w

        # get the position in the whole heatmap instead of half-sized heatmap
        for i in range(self.grid_points):
            xs[i::self.grid_points] += self.sub_regions[i][0]
            ys[i::self.grid_points] += self.sub_regions[i][1]

        pred_scores, xs, ys = tuple(
            map(lambda x: x.view(R, c), [pred_scores, xs, ys]))

        # import os
        # import cv2
        # img_name = 'stage_{}_distribute.png'.format(self.stage)
        # if os.path.isfile(img_name):
        #     img = cv2.imread(img_name, cv2.COLOR_BGR2GRAY)
        # else:
        #     img = np.zeros((56, 56))
        # for x, y in zip(xs, ys):
        #     print('start')
        #     for _x, _y in zip(x, y):
        #         print(_x, _y)
        #         _x = _x.cpu().numpy()
        #         _y = _y.cpu().numpy()
        #         if (0< _x and _x < 56) and (0< _y and _y < 56):
        #             if img[_y, _x] < 255:
        #                 img[_y, _x] += 1
        #     print('stop')
        #
        # cv2.imwrite(img_name, img)
        # assert False
        # get expanded pos_bboxes

        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        mapping_ratio = cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_MAPPING_RATIO[self.stage]
        if cfg.GRID_RCNN.EXTEND_ROI:
            mapping_ratio = 1
        x1 = (det_bboxes[:, 0, None] - mapping_ratio * (widths / 2))
        y1 = (det_bboxes[:, 1, None] - mapping_ratio * (heights / 2))
        # map the grid point to the absolute coordinates
        abs_xs = (xs.float() + 0.5) / (2 * w) * (1 + mapping_ratio) * widths + x1
        abs_ys = (ys.float() + 0.5) / (2 * h) * (1 + mapping_ratio) * heights + y1

        # get the grid points indices that fall on the bbox boundaries
        x1_inds = [i for i in range(self.grid_size)]
        y1_inds = [i * self.grid_size for i in range(self.grid_size)]
        x2_inds = [
            self.grid_points - self.grid_size + i
            for i in range(self.grid_size)
        ]
        y2_inds = [(i + 1) * self.grid_size - 1 for i in range(self.grid_size)]

        # voting of all grid points on some boundary
        bboxes_x1 = (abs_xs[:, x1_inds] * pred_scores[:, x1_inds]).sum(
            dim=1, keepdim=True) / (
                        pred_scores[:, x1_inds].sum(dim=1, keepdim=True))
        bboxes_y1 = (abs_ys[:, y1_inds] * pred_scores[:, y1_inds]).sum(
            dim=1, keepdim=True) / (
                        pred_scores[:, y1_inds].sum(dim=1, keepdim=True))
        bboxes_x2 = (abs_xs[:, x2_inds] * pred_scores[:, x2_inds]).sum(
            dim=1, keepdim=True) / (
                        pred_scores[:, x2_inds].sum(dim=1, keepdim=True))
        bboxes_y2 = (abs_ys[:, y2_inds] * pred_scores[:, y2_inds]).sum(
            dim=1, keepdim=True) / (
                        pred_scores[:, y2_inds].sum(dim=1, keepdim=True))
        bbox_res = torch.cat(
            [bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2], dim=1)
        size = proposals.size
        bbox_res[:, [0, 2]].clamp_(min=0, max=size[1] - 1)
        bbox_res[:, [1, 3]].clamp_(min=0, max=size[0] - 1)

        bbox_res = bbox_res.cuda(device=device)
        return bbox_res

    def _filter_boxes(self, proposal, target):
        """Only keep boxes with positive height and width, and not-gt.
        """
        last_bbox = proposal.bbox
        gt_bbox = target.bbox
        for i in range(gt_bbox.shape[0]):
            last_bbox = torch.where(last_bbox == gt_bbox[i], torch.full_like(last_bbox, -1), last_bbox)
        s = sum([last_bbox[:, 0], last_bbox[:, 1], last_bbox[:, 2], last_bbox[:, 3]])
        keep = np.where(s.cpu() > 0)[0]
        return keep

    def add_gt_proposals(self, proposal, target):
        device = proposal.bbox.device
        gt_box = target.copy_with_fields(['labels'])
        gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
        proposal = cat_boxlist((proposal, gt_box))

        return proposal


def post_processor(stage=0, type=None):
    if type == 'cls':
        score_thresh = cfg.GRID_RCNN.SCORE_THRESH
        nms = cfg.GRID_RCNN.NMS
        postprocessor = CLSPostProcessor(
            score_thresh,
            nms,
        )
    elif type == 'grid':
        grid_points = cfg.GRID_RCNN.GRID_POINTS if not cfg.GRID_RCNN.CASCADE_MAPPING_ON else \
            cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.GRID_NUM[stage]
        roi_feat_size = cfg.GRID_RCNN.ROI_FEAT_SIZE
        postprocessor = GridPostProcessor(
            stage,
            grid_points,
            roi_feat_size,
        )
    else:
        raise Exception('Type error!')
    return postprocessor

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from pet.utils.data.structures.bounding_box import BoxList
from pet.utils.data.structures.boxlist_ops import boxlist_nms, boxlist_soft_nms, boxlist_box_voting, boxlist_ml_nms
from pet.utils.data.structures.boxlist_ops import cat_boxlist
from pet.rcnn.utils.box_coder import BoxCoder
from pet.rcnn.core.config import cfg
from pet.rcnn.modeling.grid_rcnn.loss import calc_sub_regions


class CLSPostProcessor(nn.Module):
    """
    From a set of classification scores and proposals,
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
        """
        super(CLSPostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms

    def forward(self, x, boxes):
        class_logits = x
        class_prob = F.softmax(class_logits, -1)

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
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
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
    """
    From a set of grid scores and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            grid_points,
            roi_feat_size,
    ):
        super(GridPostProcessor, self).__init__()
        self.grid_points = grid_points
        self.roi_feat_size = roi_feat_size
        self.whole_map_size = self.roi_feat_size * 4
        self.grid_size = int(np.sqrt(self.grid_points))
        self.sub_regions = calc_sub_regions(grid_points, self.grid_size, self.whole_map_size)

    def forward(self, grid_logits, proposals):
        grid_pred = grid_logits['fused']
        result_box = self.get_boxes(proposals[0], grid_pred)
        proposals[0].bbox = result_box

        return proposals

    def get_boxes(self, proposals, grid_pred):
        det_bboxes = proposals.bbox
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

        # reshape to (num_rois, grid_points)
        pred_scores, xs, ys = tuple(
            map(lambda x: x.view(R, c), [pred_scores, xs, ys]))

        # get expanded pos_bboxes
        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        x1 = (det_bboxes[:, 0, None] - widths / 2)
        y1 = (det_bboxes[:, 1, None] - heights / 2)
        # map the grid point to the absolute coordinates
        abs_xs = (xs.float() + 0.5) / w * widths + x1
        abs_ys = (ys.float() + 0.5) / h * heights + y1

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


def post_processor(type=None):
    if type == 'cls':
        score_thresh = cfg.GRID_RCNN.SCORE_THRESH
        nms = cfg.GRID_RCNN.NMS
        postprocessor = CLSPostProcessor(
            score_thresh,
            nms,
        )
    elif type == 'grid':
        grid_points = cfg.GRID_RCNN.GRID_POINTS
        roi_feat_size = cfg.GRID_RCNN.ROI_FEAT_SIZE
        postprocessor = GridPostProcessor(
            grid_points,
            roi_feat_size,
        )
    else:
        raise Exception('Type error!')
    return postprocessor

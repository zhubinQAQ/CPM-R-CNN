import torch
from torch import nn

from pet.rcnn.modeling.grid_rcnn import heads
from pet.rcnn.modeling.grid_rcnn import outputs
from pet.rcnn.modeling.grid_rcnn.inference import post_processor
from pet.rcnn.modeling.grid_rcnn.loss import loss_evaluator
from pet.rcnn.utils.misc import keep_only_positive_boxes, random_jitter
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg

from pet.utils.data.structures.boxlist_ops import cat_boxlist


class GridRCNN(torch.nn.Module):

    def __init__(self, dim_in, spatial_scale):
        super(GridRCNN, self).__init__()
        # cls
        head_cls = registry.ROI_CLS_HEADS[cfg.GRID_RCNN.ROI_CLS_HEAD]
        self.Head_cls = head_cls(dim_in, spatial_scale)
        output_cls = registry.ROI_CLS_OUTPUTS[cfg.GRID_RCNN.ROI_CLS_OUTPUT]
        self.Output_cls = output_cls(self.Head_cls.dim_out)

        # grid
        head_grid = registry.ROI_GRID_HEADS[cfg.GRID_RCNN.ROI_GRID_HEAD]
        self.Head_grid = head_grid(dim_in, spatial_scale)
        output_grid = registry.ROI_GRID_OUTPUTS[cfg.GRID_RCNN.ROI_GRID_OUTPUT]
        self.Output_grid = output_grid(self.Head_grid.dim_out)

        self.max_sample_num_grid = cfg.GRID_RCNN.MAX_SAMPLE_NUM_GRID

        self.cls_post_processor = post_processor(type='cls')
        self.cls_loss_evaluator = loss_evaluator(type='cls')
        self.grid_post_processor = post_processor(type='grid')
        self.grid_loss_evaluator = loss_evaluator(type='grid')

    def forward(self, features, proposals, targets=None):
        if self.training:
            loss = {}
            features, proposals, loss_cls = self._forward_train_cls(features, proposals, targets)
            x, result, loss_grid = self._forward_train_grid(features, proposals, targets)
            loss.update(loss_cls)
            loss.update(loss_grid)
            return x, result, loss
        else:
            features, result, _ = self._forward_test_cls(features, proposals)
            if len(result[0]) == 0:
                return features, result, {}
            x, result, _ = self._forward_test_grid(features, result)
            return x, result, {}

    def _forward_train_cls(self, features, proposals, targets=None):
        with torch.no_grad():
            proposals = self.cls_loss_evaluator.subsample(proposals, targets)

        x = self.Head_cls(features, proposals)
        class_logits = self.Output_cls(x)

        loss_classifier = self.cls_loss_evaluator([class_logits])
        return (
            features,
            proposals,
            dict(loss_classifier=loss_classifier),
        )

    def _forward_test_cls(self, features, proposals):
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.Head_cls(features, proposals)
        # final classifier that converts the features into predictions
        class_logits = self.Output_cls(x)

        result = self.cls_post_processor(class_logits, proposals)
        return features, result, {}

    def _forward_train_grid(self, features, proposals, targets=None):
        all_proposals = proposals
        if cfg.GRID_RCNN.RANDOM_JITTER:
            proposals = random_jitter(proposals)
        proposals = keep_only_positive_boxes(proposals, roi_batch_size=self.max_sample_num_grid,
                                             across_sample=cfg.GRID_RCNN.ACROSS_SAMPLE)

        x, x_so = self.Head_grid(features, proposals)
        grid_logits = self.Output_grid(x, x_so)

        loss_grid = self.grid_loss_evaluator(proposals, grid_logits, targets)

        return (
            x,
            all_proposals,
            dict(loss_grid=loss_grid),
        )

    def _forward_test_grid(self, features, proposals):
        # large_ind = select_boxes(proposals, type='split')
        # old_proposals = [proposals[0].copy_with_fields(['scores', 'labels'])]

        x, x_so = self.Head_grid(features, proposals)
        grid_logits = self.Output_grid(x, x_so)
        result = self.grid_post_processor(grid_logits, proposals)
        # result = select_boxes((result, old_proposals), type='cat', ind=large_ind)
        return x, result, {}


def select_boxes(proposals, type=None, ind=None):
    if type == 'split':
        assert isinstance(proposals, list)
        thresh = 0 ** 2
        bbox = proposals[0].bbox
        s = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        l_ind = s > thresh
        s_ind = s <= thresh
        return [s_ind, l_ind]
    elif type == 'cat':
        assert isinstance(proposals, tuple)
        assert ind is not None
        proposals, old_proposals = proposals
        proposals[0] = proposals[0][ind[0]]
        old_proposals[0] = old_proposals[0][ind[1]]
        return [cat_boxlist((proposals[0], old_proposals[0]))]
    else:
        raise Exception('error')
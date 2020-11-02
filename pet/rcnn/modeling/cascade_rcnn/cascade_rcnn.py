import copy

import torch
from torch import nn

from pet.rcnn.modeling.cascade_rcnn import heads
from pet.rcnn.modeling.cascade_rcnn import outputs
from pet.rcnn.modeling.cascade_rcnn.inference import box_post_processor
from pet.rcnn.modeling.cascade_rcnn.loss import box_loss_evaluator
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg

from pet.utils.data.structures.boxlist_ops import cat_boxlist


class CascadeRCNN(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, dim_in, spatial_scale):
        super(CascadeRCNN, self).__init__()
        self.num_stage = cfg.CASCADE_RCNN.NUM_STAGE
        self.test_stage = cfg.CASCADE_RCNN.TEST_STAGE
        self.stage_loss_weights = cfg.CASCADE_RCNN.STAGE_WEIGHTS
        self.test_ensemble = cfg.CASCADE_RCNN.TEST_ENSEMBLE

        head = registry.ROI_CASCADE_HEADS[cfg.CASCADE_RCNN.ROI_BOX_HEAD]
        output = registry.ROI_CASCADE_OUTPUTS[cfg.CASCADE_RCNN.ROI_BOX_OUTPUT]

        for stage in range(1, self.num_stage + 1):
            stage_name = '_{}'.format(stage)
            setattr(self, 'Box_Head' + stage_name, head(dim_in, spatial_scale))
            setattr(self, 'Output' + stage_name, output(getattr(self, 'Box_Head' + stage_name).dim_out, stage-1))
        if cfg.CASCADE_RCNN.RESCORE_ON:
            from pet.rcnn.modeling.grid_cascade_rcnn.loss import loss_evaluator
            from pet.rcnn.modeling.grid_cascade_rcnn.inference import post_processor
            head_rescore = registry.ROI_CLS_HEADS[cfg.GRID_RCNN.ROI_CLS_HEAD]
            output_rescore = registry.ROI_CLS_OUTPUTS[cfg.GRID_RCNN.ROI_CLS_OUTPUT]
            self.Head_rescore = head_rescore(dim_in, spatial_scale)
            self.Output_rescore = output_rescore(self.Head_rescore.dim_out)
            self.rescore_loss_evaluator = loss_evaluator(type='cls')
            self.cls_post_processor = post_processor(type='cls')
            self.cls_init_proposals = None

    def forward(self, conv_features, proposals, targets=None):
        all_loss = dict()
        ms_scores = []
        for i in range(self.num_stage):
            head = getattr(self, 'Box_Head_{}'.format(i + 1))
            output = getattr(self, 'Output_{}'.format(i + 1))

            loss_scalar = self.stage_loss_weights[i]
            loss_evaluator = box_loss_evaluator(i)

            if self.training:
                # Cascade R-CNN subsamples during training the proposals with a fixed
                # positive / negative ratio
                with torch.no_grad():
                    proposals = loss_evaluator.subsample(proposals, targets)
                    if i == 0:
                        self.cls_init_proposals = copy.deepcopy(proposals)

            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = head(conv_features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression, iou_logits = output(x)
            ms_scores.append(class_logits)

            if not self.training:
                post_processor_test = box_post_processor(i, is_train=False)
                if i < self.test_stage - 1:
                    proposals = post_processor_test((class_logits, box_regression), proposals, iou_logits=iou_logits)
                else:
                    if self.test_ensemble:
                        assert len(ms_scores) == self.test_stage
                        class_logits = sum(ms_scores) / self.test_stage
                    result = post_processor_test((class_logits, box_regression), proposals, iou_logits=iou_logits)
                    return x, result, {}
            else:
                post_processor_train = box_post_processor(i, is_train=True)
                loss_classifier, loss_box_reg, per_loss_iou = loss_evaluator(
                    [class_logits], [box_regression], iou_logits
                )
                all_loss['s{}_cls_loss'.format(i + 1)] = loss_classifier * loss_scalar
                all_loss['s{}_bbox_loss'.format(i + 1)] = loss_box_reg * loss_scalar
                if i < self.num_stage - 1:
                    with torch.no_grad():
                        proposals = post_processor_train((class_logits, box_regression), proposals, targets, iou_logits=iou_logits)
        if cfg.CASCADE_RCNN.IOU_HELPER:
            per_loss_iou *= cfg.CASCADE_RCNN.IOU_LOSS_WEIGHT
            all_loss.update({'loss_iou_{}'.format(self.num_stage): per_loss_iou})
        if cfg.CASCADE_RCNN.RESCORE_ON:
            if self.training:
                proposals, loss_rescore = self._forward_train_rescore(conv_features, self.cls_init_proposals, proposals, targets)
                all_loss.update(loss_rescore)
            else:
                proposals, _ = self._forward_test_rescore(conv_features, proposals)
        return (
            x,
            proposals,
            all_loss,
        )

    def _forward_train_rescore(self, features, cls_proposals, grid_proposals, targets):
        assert cls_proposals is not None
        with torch.no_grad():
            proposals = get_full_sample_boxes(cls_proposals, grid_proposals)
            proposals = self.rescore_loss_evaluator.subsample(proposals, targets)

        x = self.Head_rescore(features, proposals)
        class_logits = self.Output_rescore(x)

        loss_rescore = self.rescore_loss_evaluator([class_logits]) * cfg.CASCADE_RCNN.RESCORE_LOSS_WEIGHT
        return (
            proposals,
            dict(loss_rescore=loss_rescore),
        )

    def _forward_test_rescore(self, features, proposals):
        x = self.Head_rescore(features, proposals)
        # final classifier that converts the features into predictions
        class_logits = self.Output_rescore(x)

        result = self.cls_post_processor(class_logits, proposals, rescore=True)
        return result, {}


def get_full_sample_boxes(cls_proposals, grid_proposals):
    full_boxes = []
    for cls_proposal, grid_proposal in zip(cls_proposals, grid_proposals):
        labels = cls_proposal.get_field("labels")
        neg_inds = labels <= 0
        inds = neg_inds.nonzero().squeeze(1)
        if cfg.GRID_RCNN.RESCORE_OPTION.KEEP_RATIO:
            pos_num = grid_proposal.bbox.shape[0]
            neg_num = pos_num * 3
            if neg_num <= inds.shape[0]:
                _ind = torch.randperm(inds.shape[0])[:neg_num]
                inds = inds[_ind]
        box = cat_boxlist((cls_proposal[inds], grid_proposal))
        full_boxes.append(box)
    return full_boxes
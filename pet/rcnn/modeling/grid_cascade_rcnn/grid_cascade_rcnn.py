import torch
from torch import nn

from pet.rcnn.modeling.grid_rcnn import heads
from pet.rcnn.modeling.grid_rcnn import outputs
from pet.rcnn.modeling.grid_cascade_rcnn.inference import post_processor
from pet.rcnn.modeling.grid_cascade_rcnn.loss import loss_evaluator
from pet.rcnn.utils.misc import keep_only_positive_boxes, random_jitter
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg

from pet.utils.data.structures.boxlist_ops import cat_boxlist


class GridCascadeRCNN(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, dim_in, spatial_scale):
        super(GridCascadeRCNN, self).__init__()
        # cls
        head_cls = registry.ROI_CLS_HEADS[cfg.GRID_RCNN.ROI_CLS_HEAD]
        self.Head_cls = head_cls(dim_in, spatial_scale)
        output_cls = registry.ROI_CLS_OUTPUTS[cfg.GRID_RCNN.ROI_CLS_OUTPUT]
        self.Output_cls = output_cls(self.Head_cls.dim_out)

        self.cls_post_processor = post_processor(type='cls')
        self.cls_loss_evaluator = loss_evaluator(type='cls')

        # grid
        self.max_sample_num_grid = cfg.GRID_RCNN.MAX_SAMPLE_NUM_GRID

        # cascade mapping options
        self.test_ensemble = cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.TEST_ENSEMBLE
        self.stage_num = cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_NUM
        self.test_stage = cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.TEST_STAGE
        self.stage_loss_weight = cfg.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_WEIGHTS

        head_grid = registry.ROI_GRID_HEADS[cfg.GRID_RCNN.ROI_GRID_HEAD]
        output_grid = registry.ROI_GRID_OUTPUTS[cfg.GRID_RCNN.ROI_GRID_OUTPUT]

        for stage in range(self.stage_num):
            setattr(self, 'Head_grid_' + str(stage), head_grid(dim_in, spatial_scale, stage))
            setattr(self, 'Output_grid_' + str(stage),
                    output_grid(getattr(self, 'Head_grid_' + str(stage)).dim_out, stage))
            setattr(self, 'grid_loss_evaluator_' + str(stage), loss_evaluator(stage=stage, type='grid'))
            setattr(self, 'grid_post_processor_' + str(stage), post_processor(stage=stage, type='grid'))

        if cfg.GRID_RCNN.RESCORE_ON:
            head_rescore = registry.ROI_CLS_HEADS[cfg.GRID_RCNN.ROI_CLS_HEAD]
            output_rescore = registry.ROI_CLS_OUTPUTS[cfg.GRID_RCNN.ROI_CLS_OUTPUT]
            self.Head_rescore = head_rescore(dim_in, spatial_scale)
            self.Output_rescore = output_rescore(self.Head_rescore.dim_out)
            self.rescore_loss_evaluator = loss_evaluator(type='cls')

    def forward(self, features, proposals, targets=None):
        if self.training:
            loss = {}
            features, proposals, loss_cls = self._forward_train_cls(features, proposals, targets)
            x, result, loss_grid = self._forward_train_cascade(features, proposals, targets)
            if cfg.GRID_RCNN.RESCORE_ON:
                result, loss_rescore = self._forward_train_rescore(features, proposals, result, targets)
                loss.update(loss_rescore)
            loss.update(loss_cls)
            loss.update(loss_grid)
            return x, result, loss
        else:
            features, proposals, _ = self._forward_test_cls(features, proposals)
            if len(proposals[0]) == 0:
                return features, proposals, {}
            x, result, _ = self._forward_test_cascade(features, proposals)
            if cfg.GRID_RCNN.RESCORE_ON:
                result, _ = self._forward_test_rescore(features, result)
            # return features, proposals, {}
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

    def _forward_train_cascade(self, features, proposals, targets=None):
        loss_grid = {}
        if cfg.GRID_RCNN.ENHANCE_FEATURES:
            features = enhance_features(features)
        for stage in range(self.stage_num):
            grid_post_processor = getattr(self, 'grid_post_processor_' + str(stage))

            x, grid_logits, proposals, per_loss_grid, per_loss_iou = self._forward_train_grid(stage, features, proposals,
                                                                                targets=targets)
            per_loss_grid *= self.stage_loss_weight[stage]
            if cfg.GRID_RCNN.IOU_HELPER and stage == self.stage_num - 1:
                per_loss_iou *= cfg.GRID_RCNN.IOU_LOSS_WEIGHT
                loss_grid.update({'loss_iou_{}'.format(stage + 1): per_loss_iou})
            loss_grid.update({'loss_grid_{}'.format(stage + 1): per_loss_grid})

            if stage < self.stage_num - 1:
                with torch.no_grad():
                    proposals = grid_post_processor(grid_logits, proposals, targets=targets, is_train=True)

        return (
            x,
            proposals,
            loss_grid,
        )

    def _forward_train_grid(self, stage, features, proposals, targets=None):
        head = getattr(self, 'Head_grid_' + str(stage))
        output = getattr(self, 'Output_grid_' + str(stage))
        stage_loss_evaluator = getattr(self, 'grid_loss_evaluator_' + str(stage))
        if stage == 0:
            if cfg.GRID_RCNN.RANDOM_JITTER:
                proposals = random_jitter(proposals)
            proposals = keep_only_positive_boxes(proposals, roi_batch_size=self.max_sample_num_grid,
                                                 across_sample=cfg.GRID_RCNN.ACROSS_SAMPLE)

        with torch.no_grad():
            proposals = stage_loss_evaluator.subsample(proposals, targets)

        if cfg.GRID_RCNN.EXTEND_ROI:
            proposals = extend(proposals)
        if cfg.GRID_RCNN.OFFSET_ON:
            x, x_offset = head(features, proposals)
            new_proposals = apply_offset(proposals, x_offset)
            x_so = output_offset(new_proposals)
        else:
            x, x_so = head(features, proposals)
        grid_logits, iou_logits = output(x, x_so)

        # new proposals means they have new refine labels
        loss_grid, loss_iou = stage_loss_evaluator(proposals, grid_logits, iou_logits, targets)

        return (
            x,
            grid_logits,
            proposals,
            loss_grid,
            loss_iou,
        )

    def _forward_test_cascade(self, features, proposals):
        logits = []
        all_iou_logits = []
        if cfg.GRID_RCNN.ENHANCE_FEATURES:
            features = enhance_features(features)
        # large_ind = select_boxes(proposals, type='split')
        # old_proposals = [proposals[0].copy_with_fields(['scores', 'labels'])]
        for stage in range(self.stage_num):
            grid_post_processor = getattr(self, 'grid_post_processor_' + str(stage))
            x, grid_logits, iou_logits, proposals, _ = self._forward_test_grid(stage, features, proposals)

            logits.append(grid_logits)
            all_iou_logits.append(iou_logits)

            if stage < self.stage_num - 1:
                proposals = grid_post_processor(grid_logits, proposals, iou_logits)
                # break
                # if stage==0:
                #     old_proposals = [proposals[0].copy_with_fields(['scores', 'labels'])]
                if stage == self.test_stage - 1:
                    break
            else:
                if self.test_ensemble:
                    raise Exception('unsupported operation!')
                else:
                    logits = logits[-1]
                proposals = grid_post_processor(logits, proposals, iou_logits)
        # proposals = select_boxes((proposals, old_proposals), type='cat', ind=large_ind)
        return (
            x,
            proposals,
            {},
        )

    def _forward_test_grid(self, stage, features, proposals):
        head = getattr(self, 'Head_grid_' + str(stage))
        output = getattr(self, 'Output_grid_' + str(stage))
        if cfg.GRID_RCNN.EXTEND_ROI:
            proposals = extend(proposals)
        x, x_so = head(features, proposals)
        grid_logits, iou_logits = output(x, x_so)
        return x, grid_logits, iou_logits, proposals, {}

    def _forward_train_rescore(self, features, cls_proposals, grid_proposals, targets):
        with torch.no_grad():
            proposals = get_full_sample_boxes(cls_proposals, grid_proposals)
            proposals = self.rescore_loss_evaluator.subsample(proposals, targets)

        x = self.Head_rescore(features, proposals)
        class_logits = self.Output_rescore(x)

        loss_rescore = self.rescore_loss_evaluator([class_logits]) * cfg.GRID_RCNN.RESCORE_LOSS_WEIGHT
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

    def apply_offset(self, proposals, x_offset):
        
        return new
    

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


def enhance_features(features):
    import torch.nn.functional as F
    out = []
    # p2, p3, p4, p5, p6 (0.25, 0.125, 0.0625, 0.03125, 0.015625)
    for i in range(len(features)):
        feature = features[i]
        if i > 0:
            _, _, h, w = feature.shape
            down_features = F.interpolate(out[i - 1], (h, w), mode='nearest')
            feature = feature + down_features

        if i < len(features) - 1:
            _, _, h, w = feature.shape
            up_features = F.interpolate(features[i + 1], (h, w), mode='nearest')
            feature = feature + up_features
            out.append(feature)
        else:
            break
    assert len(out) == 4
    assert features[0].shape == out[0].shape
    # print([o.shape for o in out])
    # assert False
    return out


def extend(boxlists):
    rt_boxlists = []
    for boxlist in boxlists:
        boxes = boxlist.bbox
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0).unsqueeze(1)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0).unsqueeze(1)
        delta = torch.cat([-0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h], dim=1)
        assert delta.shape == boxes.shape
        new = boxes + delta
        i_w, i_h = boxlist.size
        new[:, 0] = new[:, 0].clamp(min=0, max=i_w)
        new[:, 1] = new[:, 1].clamp(min=0, max=i_h)
        new[:, 2] = new[:, 2].clamp(min=0, max=i_w)
        new[:, 3] = new[:, 3].clamp(min=0, max=i_h)
        boxlist.bbox = new
        rt_boxlists.append(boxlist)
    return rt_boxlists
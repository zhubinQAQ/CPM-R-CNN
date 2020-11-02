import tempfile
import os
import json
import pickle
import numpy as np
from collections import OrderedDict

from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from pet.rcnn.datasets.mycocoeval import COCOeval

import torch

from pet.utils.data.structures.bounding_box import BoxList
from pet.utils.data.structures.boxlist_ops import boxlist_iou
from pet.utils.misc import logging_rank
from pet.rcnn.datasets import dataset_catalog
from pet.rcnn.core.config import cfg


def post_processing(results, image_ids, dataset):
    cpu_device = torch.device("cpu")
    results = [o.to(cpu_device) for o in results]
    num_im = len(image_ids)

    box_results, ims_dets, ims_labels = prepare_box_results(results, image_ids, dataset)
    if cfg.MODEL.MASK_ON:
        seg_results, ims_segs = prepare_segmentation_results(results, image_ids, dataset)
    else:
        seg_results = []
        ims_segs = [None for _ in range(num_im)]
    if cfg.MODEL.KEYPOINT_ON:
        kpt_results, ims_kpts = prepare_keypoint_results(results, image_ids, dataset)
    else:
        kpt_results = []
        ims_kpts = [None for _ in range(num_im)]
    if cfg.MODEL.PARSING_ON:
        par_results, par_score = prepare_parsing_results(results, image_ids, dataset)
        ims_pars = par_results
    else:
        par_results = []
        par_score = []
        ims_pars = [None for _ in range(num_im)]
    if cfg.MODEL.UV_ON:
        uvs_results, ims_uvs = prepare_uv_results(results, image_ids, dataset)
    else:
        uvs_results = []
        ims_uvs = [None for _ in range(num_im)]

    eval_results = [box_results, seg_results, kpt_results, par_results, par_score, uvs_results]
    ims_results = [ims_dets, ims_labels, ims_segs, ims_kpts, ims_pars, ims_uvs]
    return eval_results, ims_results


def evaluation(dataset, all_boxes, all_segms, all_keyps, all_parss, all_pscores, all_uvs):
    output_folder = os.path.join(cfg.CKPT, 'test')
    expected_results = ()
    expected_results_sigma_tol = 4

    coco_results = {}
    iou_types = ("bbox",)
    coco_results["bbox"] = all_boxes

    box_only = False if cfg.MODEL.RETINANET_ON or cfg.MODEL.FCOS_ON else cfg.MODEL.RPN_ONLY
    if box_only:
        logging_rank("Evaluating bbox proposals", local_rank=0)
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    coco_results["bbox"], dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logging_rank(res, local_rank=0)

        check_expected_results(res, expected_results, expected_results_sigma_tol)
        torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return

    results = COCOResults(*iou_types)
    logging_rank("Evaluating predictions", local_rank=0)
    for iou_type in iou_types:
        if iou_type == "parsing":
            eval_ap = cfg.PRCNN.EVAL_AP
            num_parsing = cfg.PRCNN.NUM_PARSING
            assert len(cfg.TEST.DATASETS) == 1, \
                'Parsing only support one dataset now'
            im_dir = dataset_catalog.get_im_dir(cfg.TEST.DATASETS[0])
            ann_fn = dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0])
            res = evaluate_parsing(coco_results[iou_type], eval_ap, num_parsing, im_dir, ann_fn, output_folder)
            results.update_parsing(res)
        else:
            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if output_folder:
                    file_path = os.path.join(output_folder, iou_type + ".json")
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type
                )
                results.update(res)
    logging_rank(results, local_rank=0)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results


def prepare_box_results(results, image_ids, dataset):
    box_results = []
    ims_dets = []
    ims_labels = []
    box_only = False if cfg.MODEL.RETINANET_ON or cfg.MODEL.FCOS_ON else cfg.MODEL.RPN_ONLY
    if box_only:
        return results, None, None
    else:
        for i, result in enumerate(results):
            image_id = image_ids[i]
            original_id = dataset.id_to_img_map[image_id]
            if len(result) == 0:
                ims_dets.append(None)
                ims_labels.append(None)
                continue
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            result = result.resize((image_width, image_height))
            boxes = result.bbox
            scores = result.get_field("scores")
            labels = result.get_field("labels")
            ims_dets.append(np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False))
            result = result.convert("xywh")
            boxes = result.bbox.tolist()
            scores = scores.tolist()
            labels = labels.tolist()
            ims_labels.append(labels)
            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
            box_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
    return box_results, ims_dets, ims_labels


# inspired from Detectron
def evaluate_box_proposals(
        predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
        coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    if iou_type != "uv":
        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)
        coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
        # coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
    else:
        calc_mode = 'GPSm' if cfg.UVRCNN.GPSM_ON else 'GPS'
        pkl_result_file = json_result_file.replace('.json', '.pkl')
        with open(pkl_result_file, 'wb') as f:
            pickle.dump(coco_results, f, 2)
        if cfg.TEST.DATASETS[0].find('test') > -1:
            return
        evalDataDir = os.path.dirname(__file__) + '/../../../data/DensePoseData/eval_data/'
        coco_dt = coco_gt.loadRes(coco_results)
        test_sigma = 0.255
        coco_eval = denseposeCOCOeval(evalDataDir, coco_gt, coco_dt, iou_type, test_sigma)
        coco_eval.evaluate(calc_mode=calc_mode)
    coco_eval.accumulate()
    if iou_type == "bbox":
        _print_detection_eval_metrics(coco_gt, coco_eval)
    coco_eval.summarize()
    return coco_eval


def _print_detection_eval_metrics(coco_gt, coco_eval):
    # mAP = 0.0
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    category_ids = coco_gt.getCatIds()
    categories = [c['name'] for c in coco_gt.loadCats(category_ids)]
    classes = tuple(['__background__'] + categories)
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{} = {:.1f}'.format(cls, 100 * ap))


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "1", "2", "3", "4", "5", "6", "AP60", "AP70", "AP80", "AP90"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        "parsing": ["mIoU", "mIoUs", "mIoUm", "mIoUl", "APp50", "APpvol", "PCP"],
        "uv": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints", "parsing", 'uv')
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, (COCOeval, ))
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        if iou_type == 'uv':
            idx_map = [0, 1, 6, 11, 12]
            for idx, metric in enumerate(metrics):
                res[metric] = s[idx_map[idx]]
        else:
            for idx, metric in enumerate(metrics):
                res[metric] = s[idx]

    def update_parsing(self, eval):
        if eval is None:
            return

        res = self.results['parsing']
        for k, v in eval.items():
            res[k] = v

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("pet.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)

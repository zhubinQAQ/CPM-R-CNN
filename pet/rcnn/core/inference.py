import numpy as np

import torch

import pet.rcnn.core.test as rcnn_test
from pet.rcnn.modeling.mask_rcnn.inference import mask_results
from pet.rcnn.modeling.keypoint_rcnn.inference import keypoint_results
from pet.rcnn.modeling.parsing_rcnn.inference import parsing_results
from pet.rcnn.modeling.uv_rcnn.inference import uv_results
from pet.rcnn.core.config import cfg


def inference(model, img):
    results, features = rcnn_test.im_detect_bbox(model, [img])
    image_width = img.shape[1]
    image_height = img.shape[0]
    if cfg.MODEL.MASK_ON and not cfg.MODEL.RPN_ONLY:
        results = rcnn_test.im_detect_mask(model, results, features)
    if cfg.MODEL.KEYPOINT_ON:
        results = rcnn_test.im_detect_keypoint(model, results, features)
    if cfg.MODEL.PARSING_ON:
        results = rcnn_test.im_detect_parsing(model, results, features)
    if cfg.MODEL.UV_ON:
        results = rcnn_test.im_detect_uv(model, results, features)

    cpu_device = torch.device("cpu")
    results = [o.to(cpu_device) for o in results]
    result = results[0].resize((image_width, image_height))

    boxes = result.bbox
    scores = result.get_field("scores")
    im_labels = result.get_field("labels").tolist()
    im_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    if cfg.MODEL.MASK_ON:
        masks = result.get_field("mask")
        im_segs = mask_results(masks, result)
    else:
        im_segs = None
    if cfg.MODEL.KEYPOINT_ON:
        keypoints = result.get_field("keypoints")
        _, im_kpts = keypoint_results(keypoints, result)
    else:
        im_kpts = None
    if cfg.MODEL.PARSING_ON:
        parsing = result.get_field("parsing")
        im_pars = parsing_results(parsing, result)
    else:
        im_pars = None
    if cfg.MODEL.UV_ON:
        uv_logits = result.get_field("uv")
        im_uvs = uv_results(uv_logits, result)
    else:
        im_uvs = None

    return im_dets, im_labels, im_segs, im_kpts, im_pars, im_uvs

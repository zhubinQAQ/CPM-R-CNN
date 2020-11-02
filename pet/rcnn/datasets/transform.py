from pet.utils.data import transforms as T

from pet.rcnn.core.config import cfg


def build_transforms(is_train=True):
    if is_train:
        min_size = cfg.TRAIN.SCALES
        max_size = cfg.TRAIN.MAX_SIZE
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.TRAIN.BRIGHTNESS
        contrast = cfg.TRAIN.CONTRAST
        saturation = cfg.TRAIN.SATURATION
        hue = cfg.TRAIN.HUE
        left_right = cfg.TRAIN.LEFT_RIGHT
    else:
        min_size = cfg.TEST.SCALE
        max_size = cfg.TEST.MAX_SIZE
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        left_right = ()

    to_bgr255 = cfg.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob, left_right),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

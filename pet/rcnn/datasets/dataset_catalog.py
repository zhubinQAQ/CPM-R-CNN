import os.path as osp

from pet.utils.data.dataset_catalog import COMMON_DATASETS

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Available datasets
_DATASETS = {
    'cocohumanparts_2017_headface_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_headface_train2017.json',
    },
    'cocohumanparts_2017_headface_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_headface_val2017.json',
    },
    'cocohumanparts_2017_personheadface_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_personheadface_train2017.json',
    },
    'cocohumanparts_2017_personheadface_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_personheadface_val2017.json',
    },
    'cocohumanparts_2017_all_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_humanparts_train2017.json',
    },
    'cocohumanparts_2017_all_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_humanparts_val2017.json',
    },
    'cocohumanparts_2017_onlyparts_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_onlyparts_train2017.json',
    },
    'cocohumanparts_2017_onlyparts_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/COCOHumanParts/instance_onlyparts_val2017.json',
    },
    'OID_2019_train': {
        _IM_DIR:
            _DATA_DIR + '/OID/challenge2019/images/train',
        _ANN_FN:
            _DATA_DIR + '/OID/challenge2019/Json_Annos/instances_train2019.json',
    },
    'OID_2019_val': {
        _IM_DIR:
            _DATA_DIR + '/OID/challenge2019/images/validation',
        _ANN_FN:
            _DATA_DIR + '/OID/challenge2019/Json_Annos/instances_val2019.json',
    },
    'OID_2019_test': {
        _IM_DIR:
            _DATA_DIR + '/OID/challenge2019/images/test',
        _ANN_FN:
            _DATA_DIR + '/OID/challenge2019/Json_Annos/instances_test2019.json',
    }
}
_DATASETS.update(COMMON_DATASETS)


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]

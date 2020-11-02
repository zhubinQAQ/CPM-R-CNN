import os
import os.path as osp
import copy
import yaml
import numpy as np
from ast import literal_eval

from pet.utils.collections import AttrDict

__C = AttrDict()
cfg = __C


# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #
# Device for training or testing
# E.g., 'cuda' for using GPU, 'cpu' for using CPU
__C.DEVICE = 'cuda'

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# Pixel mean values (BGR order) as a list
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Pixel std values (BGR order) as a list
__C.PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])

# Directory for saving checkpoints and loggers
__C.CKPT = 'ckpts/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x'

# Display the log per iteration
__C.DISPLAY_ITER = 20

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# A very small number that's used many times
__C.EPS = 1e-14

# Convert image to BGR format (for Caffe2 models), in range 0-255
__C.TO_BGR255 = True


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The type of model to use
# The string must match a function in the modeling.model_builder module
# (e.g., 'generalized_rcnn', 'retinanet', ...)
__C.MODEL.TYPE = 'generalized_rcnn'

# FPN is enabled if True
__C.MODEL.FPN_ON = False

# The meaning of FASTER_RCNN depends on the context (training vs. inference):
# 1) During training, FASTER_RCNN = True means that end-to-end training will be
#    used to jointly train the RPN subnetwork and the Fast R-CNN subnetwork
#    (Faster R-CNN = RPN + Fast R-CNN).
# 2) During inference, FASTER_RCNN = True means that the model's RPN subnetwork
#    will be used to generate proposals rather than relying on precomputed
#    proposals. Note that FASTER_RCNN = True can be used at inference time even
#    if the Faster R-CNN model was trained with stagewise training (which
#    consists of alternating between RPN and Fast R-CNN training in a way that
#    finally leads to a single network).
__C.MODEL.FASTER_RCNN = True

# Indicates the model uses Grid R-CNN
__C.MODEL.GRID_ON = False  # TODO

# Indicates the model uses Cascade R-CNN
__C.MODEL.CASCADE_ON = False

# Indicates the model makes instance mask predictions (as in Mask R-CNN)
__C.MODEL.MASK_ON = False

# Indicates the model makes semantic segmentation predictions (as in Semantic R-CNN)
__C.MODEL.SEMSEG_ON = False

# Indicates the model makes keypoint predictions (as in Mask R-CNN for keypoints)
__C.MODEL.KEYPOINT_ON = False

# Indicates the model makes parsing predictions (as in Parsing R-CNN)
__C.MODEL.PARSING_ON = False

# Indicates the model makes UV predictions (as in DensePose/Parsing R-CNN)
__C.MODEL.UV_ON = False

# Indicates the model makes part bbox predictions (as in Hier R-CNN)
__C.MODEL.HIER_ON = False  # TODO

# Indicates the model's computation terminates with the production of RPN
# proposals (i.e., it outputs proposals ONLY, no actual object detections)
__C.MODEL.RPN_ONLY = False 

# RetinaNet is used (instead of Fast(er)/Mask R-CNN/RPN) if True
__C.MODEL.RETINANET_ON = False

# FCOS is used
__C.MODEL.FCOS_ON = False

# Indicates the model trained by mix-supervised (as in Mix-Supervised Learning)
__C.MODEL.MSL_ON = False  # TODO

# Type of batch normalizaiton, default: 'freeze'
# E.g., 'normal', 'freeze', 'sync', ...
__C.MODEL.BATCH_NORM = 'freeze'

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = -1

# Use a class agnostic bounding box regressor instead of the default per-class
# regressor
__C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# Swap model conv1 weight, for pet/rcnn we use BGR input channel (cv2), for pet/cls we use RGB channel, 
# for caffe/caffe2 model using BGR channel. Thus if we use pet pretrain weights set 'True', else if use 
# caffe or caffe2 weights set 'False'.
__C.MODEL.CONV1_RGB2BGR = True


# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Type of the optimizer
# E.g., 'SGD', 'RMSPROP', 'ADAM' ...
__C.SOLVER.OPTIMIZER = 'SGD'

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# Maximum number of max iterations
__C.SOLVER.MAX_ITER = 90000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

# L2 regularization hyperparameter for GroupNorm's parameters
__C.SOLVER.WEIGHT_DECAY_GN = 0.0

# Whether to double the learning rate for bias
__C.SOLVER.BIAS_DOUBLE_LR = True

# Whether to have weight decay on bias as well
__C.SOLVER.BIAS_WEIGHT_DECAY = False

# Multiple learning rate for fine-tuning
# Random initial layer learning rate is LR_MULTIPLE * BASE_LR
__C.SOLVER.LR_MULTIPLE = 1.0  # TODO

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 10.0

# WARM_UP_METHOD can be either 'CONSTANT' or 'LINEAR' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'LINEAR'

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'POLY', 'STEP', 'COSINE', ...
__C.SOLVER.LR_POLICY = 'STEP'

# For 'POLY', the power in poly to drop LR
__C.SOLVER.LR_POW = 0.9

# For 'STEP', Non-uniform step iterations
__C.SOLVER.STEPS = [60000, 80000]

# For 'STEP', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1

# Snapshot (model checkpoint) period
__C.SOLVER.SNAPSHOT_ITERS = 10000


# -----------------------------------------------------------------------------
# DataLoader options
# -----------------------------------------------------------------------------
__C.DATALOADER = AttrDict()

# Type of training sampler, default: 'DistributedSampler'
# E.g., 'DistributedSampler', 'RepeatFactorTrainingSampler', ...
__C.DATALOADER.SAMPLER_TRAIN = "DistributedSampler"

# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
__C.DATALOADER.ASPECT_RATIO_GROUPING = True

# if True, the dataloader will filter out images that have no associated
# annotations at train time.
__C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True  # TODO

# ---------------------------------------------------------------------------- #
# RepeatFactorTrainingSampler options
# ---------------------------------------------------------------------------- #
__C.DATALOADER.RFTSAMPLER = AttrDict()

# parameters for RepeatFactorTrainingSampler
# rep_times = max(MIN_REPEAT_TIMES, min(MAX_REPEAT_TIMES, math.pow((REPEAT_THRESHOLD / cat_freq),POW)))
__C.DATALOADER.RFTSAMPLER.REPEAT_THRESHOLD = 0.001
__C.DATALOADER.RFTSAMPLER.POW = 0.5
__C.DATALOADER.RFTSAMPLER.MAX_REPEAT_TIMES = 10000.0
__C.DATALOADER.RFTSAMPLER.MIN_REPEAT_TIMES = 1.0


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Initialize network with weights from this .pkl file
__C.TRAIN.WEIGHTS = ''

# Datasets to train on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

# Scales to use during training
# Each scale is the pixel size of an image's shortest side
# If multiple scales are listed, then one is selected uniformly at random for
# each training image (i.e., scale jitter data augmentation)
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Number of Python threads to use for the data loader during training
__C.TRAIN.LOADER_THREADS = 4

# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
__C.TRAIN.SIZE_DIVISIBILITY = 32

# Mini-batch size for training
# This is global, so if we have 8 GPUs and BATCH_SIZE = 16, each GPU will
# see 2 images per batch
__C.TRAIN.BATCH_SIZE = 16

# Freeze the backbone architecture during training if set to True
__C.TRAIN.FREEZE_CONV_BODY = False

# Training will resume from the latest snapshot (model checkpoint) found in the
# output directory
__C.TRAIN.AUTO_RESUME = True

# Image ColorJitter Augmentation
__C.TRAIN.BRIGHTNESS = 0.0
__C.TRAIN.CONTRAST = 0.0
__C.TRAIN.SATURATION = 0.0
__C.TRAIN.HUE = 0.0

# Left right mapping for flipping training
__C.TRAIN.LEFT_RIGHT = ()


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Initialize network with weights from this .pkl file
__C.TEST.WEIGHTS = ''

# Number of Python threads to use for the data loader during testing
__C.TEST.LOADER_THREADS = 4

# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
__C.TEST.SIZE_DIVISIBILITY = 32

# Datasets to test on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASETS = ()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALE = 600

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Number of images in each GPU for testing
__C.TEST.IMS_PER_GPU = 1

# ---------------------------------------------------------------------------- #
# Soft NMS
# ---------------------------------------------------------------------------- #
__C.TEST.SOFT_NMS = AttrDict()

# Use soft NMS instead of standard NMS if set to True
__C.TEST.SOFT_NMS.ENABLED = False

# See soft NMS paper for definition of these options
__C.TEST.SOFT_NMS.METHOD = 'linear'

__C.TEST.SOFT_NMS.SIGMA = 0.5
# For the soft NMS overlap threshold, we simply use TEST.NMS

# ---------------------------------------------------------------------------- #
# Bounding box voting (from the Multi-Region CNN paper)
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_VOTE = AttrDict()

# Use box voting if set to True
__C.TEST.BBOX_VOTE.ENABLED = False

# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# The method used to combine scores when doing bounding box voting
# Valid options include ('ID', 'AVG', 'IOU_AVG', 'GENERALIZED_AVG', 'QUASI_SUM')
__C.TEST.BBOX_VOTE.SCORING_METHOD = 'ID'

# Hyperparameter used by the scoring method (it has different meanings for
# different methods)
__C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# ---------------------------------------------------------------------------- #
__C.TEST.BBOX_AUG = AttrDict()

# Enable test-time augmentation for bounding box detection if True
__C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
__C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
__C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
__C.TEST.BBOX_AUG.MAX_SIZE = 4000

# ---------------------------------------------------------------------------- #
# Test-time augmentations for mask detection
# ---------------------------------------------------------------------------- #
__C.TEST.MASK_AUG = AttrDict()

# Enable test-time augmentation for instance mask detection if True
__C.TEST.MASK_AUG.ENABLED = False

# Heuristic used to combine mask predictions
# SOFT prefix indicates that the computation is performed on soft masks
#   Valid options: ('SOFT_AVG', 'SOFT_MAX', 'LOGIT_AVG')
__C.TEST.MASK_AUG.HEUR = 'SOFT_AVG'

# ---------------------------------------------------------------------------- #
# Test-augmentations for keypoints detection
# ---------------------------------------------------------------------------- #
__C.TEST.KPS_AUG = AttrDict()

# Enable test-time augmentation for keypoint detection if True
__C.TEST.KPS_AUG.ENABLED = False

# Heuristic used to combine keypoint predictions
#   Valid options: ('HM_AVG', 'HM_MAX')
__C.TEST.KPS_AUG.HEUR = 'HM_AVG'

# ---------------------------------------------------------------------------- #
# Test-time augmentations for parsing detection
# ---------------------------------------------------------------------------- #
__C.TEST.PARSING_AUG = AttrDict()

# Enable test-time augmentation for instance parsing detection if True
__C.TEST.PARSING_AUG.ENABLED = False

# Heuristic used to combine parsing predictions
# SOFT prefix indicates that the computation is performed on soft parsings
#   Valid options: ('SOFT_AVG', 'SOFT_MAX', 'LOGIT_AVG')
__C.TEST.PARSING_AUG.HEUR = 'SOFT_AVG'

# ---------------------------------------------------------------------------- #
# Test-time augmentations for UV detection
# See configs/test_time_aug/e2e_UV_rcnn_R-50-FPN_2x.yaml for an example
# ---------------------------------------------------------------------------- #
__C.TEST.UV_AUG = AttrDict()

# Enable test-time augmentation for instance UV detection if True
__C.TEST.UV_AUG.ENABLED = False

# Heuristic used to combine UV predictions
# SOFT prefix indicates that the computation is performed on soft UVs
#   Valid options: ('SOFT_AVG', 'SOFT_MAX')
__C.TEST.UV_AUG.HEUR = 'SOFT_AVG'


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
__C.BACKBONE = AttrDict()

# The backbone conv body to use
__C.BACKBONE.CONV_BODY = 'resnet'

# The eps of batch_norm layer
__C.BACKBONE.BN_EPS = 1e-5


# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.RESNET = AttrDict()

# The number of layers in each block
# (2, 2, 2, 2) for resnet18 with basicblock
# (3, 4, 6, 3) for resnet34 with basicblock
# (3, 4, 6, 3) for resnet50
# (3, 4, 23, 3) for resnet101
# (3, 8, 36, 3) for resnet152
__C.BACKBONE.RESNET.LAYERS = (3, 4, 6, 3)

# Network initial width
__C.BACKBONE.RESNET.WIDTH = 64

# Use bottleneck block, False for basicblock
__C.BACKBONE.RESNET.BOTTLENECK = True

# Place the stride 2 conv on the 3x3 filter.
# True for resnet-b
__C.BACKBONE.RESNET.STRIDE_3X3 = False

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnet-c
__C.BACKBONE.RESNET.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnet-d
__C.BACKBONE.RESNET.AVG_DOWN = False

# Use group normalization
__C.BACKBONE.RESNET.USE_GN = False

# Use attentive normalization
# when it is True means use an_bn (an with bn)
# when it is True and USE_GN is True means use an_gn (an with gn)
__C.BACKBONE.RESNET.USE_AN = False

# Use weight standardization
__C.BACKBONE.RESNET.USE_WS = False

# Use a aligned module in each block
__C.BACKBONE.RESNET.USE_ALIGN = False

# Type of context module in each block
# 'se' for se, 'gcb' for gcb
__C.BACKBONE.RESNET.STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')

# Context module innerplanes ratio
__C.BACKBONE.RESNET.CTX_RATIO = 0.0625

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.RESNET.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# Apply dilation in stage "c5"
__C.BACKBONE.RESNET.C5_DILATION = 1

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
__C.BACKBONE.RESNET.FREEZE_AT = 2

# ---------------------------------------------------------------------------- #
# ResNeXt options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.RESNEXT = AttrDict()

# The number of layers in each block
# (3, 4, 6, 3) for resnext50
# (3, 4, 23, 3) for resnext101
# (3, 8, 36, 3) for resnext152
__C.BACKBONE.RESNEXT.LAYERS = (3, 4, 6, 3)

# Cardinality (groups) of convolution layers
__C.BACKBONE.RESNEXT.C = 32

# Network initial width of each (conv) group
__C.BACKBONE.RESNEXT.WIDTH = 4

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnext-c
__C.BACKBONE.RESNEXT.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnext-d
__C.BACKBONE.RESNEXT.AVG_DOWN = False

# Use group normalization
__C.BACKBONE.RESNEXT.USE_GN = False

# Use weight standardization
__C.BACKBONE.RESNEXT.USE_WS = False

# Use a aligned module in each block
__C.BACKBONE.RESNEXT.USE_ALIGN = False

# Type of context module in each block
# 'se' for se, 'gcb' for gcb
__C.BACKBONE.RESNEXT.STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')

# Context module innerplanes ratio
__C.BACKBONE.RESNEXT.CTX_RATIO = 0.0625

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.RESNEXT.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# Apply dilation in stage "c5"
__C.BACKBONE.RESNEXT.C5_DILATION = 1

# Freeze model weights before and including which block.
# Choices: [0, 2, 3, 4, 5]. O means not fixed. First conv and bn are defaults to
# be fixed.
__C.BACKBONE.RESNEXT.FREEZE_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
__C.FPN = AttrDict()

# The Body of FPN to use
# (e.g., "fpn", "hrfpn", "nasfpn", "deconvx")
__C.FPN.BODY = "fpn"

# Use C5 or P5 to generate P6
__C.FPN.USE_C5 = True

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# FPN may be used for just RPN, just object detection, or both
# E.g., "conv2"-like level
__C.FPN.LOWEST_BACKBONE_LVL = 2

# E.g., "conv5"-like level
__C.FPN.HIGHEST_BACKBONE_LVL = 5

# Use FPN for RoI transform for object detection if True
__C.FPN.MULTILEVEL_ROIS = True

# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0  # TODO

__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to  # TODO

# Coarsest level of the FPN pyramid
__C.FPN.ROI_MAX_LEVEL = 5

# Finest level of the FPN pyramid
__C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
__C.FPN.MULTILEVEL_RPN = True

# Coarsest level of the FPN pyramid
__C.FPN.RPN_MAX_LEVEL = 6

# Finest level of the FPN pyramid
__C.FPN.RPN_MIN_LEVEL = 2

# Use extra FPN levels, as done in the RetinaNet paper
__C.FPN.EXTRA_CONV_LEVELS = False

# Use FPN Lite (dwconv) to replace standard FPN
__C.FPN.USE_LITE = False

# Use BatchNorm in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_BN = False

# Use GroupNorm in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_GN = False

# Use Weight Standardization in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_WS = False

# ---------------------------------------------------------------------------- #
# FPN bifpn body options
# ---------------------------------------------------------------------------- #
__C.FPN.BIFPN = AttrDict()

# Number of stacking BIFPN module
__C.FPN.BIFPN.NUM_STACK = 7

# Channel dimension of the BIFPN feature levels
__C.FPN.BIFPN.DIM = 256

# A small value to avoid numerical instability
__C.FPN.BIFPN.EPS = 1e-4

# Use BIFPN Lite (dwconv) to replace standard BIFPN
__C.FPN.BIFPN.USE_LITE = False

# Use BatchNorm in the BIFPN-specific layers
__C.FPN.BIFPN.USE_BN = False

# Use GroupNorm in the BIFPN-specific layers
__C.FPN.BIFPN.USE_GN = False

# ---------------------------------------------------------------------------- #
# FPN hrfpn body options
# ---------------------------------------------------------------------------- #
__C.FPN.HRFPN = AttrDict()

# Channel dimension of the HRFPN feature levels
__C.FPN.HRFPN.DIM = 256

# Pooling type in HRFPN for down-sampling
__C.FPN.HRFPN.POOLING_TYPE = 'AVG'

# Number of extra pooling layer in HRFPN for down-sampling
__C.FPN.HRFPN.NUM_EXTRA_POOLING = 1

# Use HRFPN Lite (dwconv) to replace standard HRFPN
__C.FPN.HRFPN.USE_LITE = False

# Use BatchNorm in the HRFPN-specific layers
__C.FPN.HRFPN.USE_BN = False

# Use GroupNorm in the HRFPN-specific layers
__C.FPN.HRFPN.USE_GN = False

# ---------------------------------------------------------------------------- #
# FPN nasfpn body options
# ---------------------------------------------------------------------------- #
__C.FPN.NASFPN = AttrDict()

# Number of stacking NASFPN module
__C.FPN.NASFPN.NUM_STACK = 7

# Channel dimension of the NASFPN feature levels
__C.FPN.NASFPN.DIM = 256

# Use NASFPN Lite (dwconv) to replace standard NASFPN
__C.FPN.NASFPN.USE_LITE = False

# Use BatchNorm in the NASFPN-specific layers
__C.FPN.NASFPN.USE_BN = False

# Use GroupNorm in the NASFPN-specific layers
__C.FPN.NASFPN.USE_GN = False


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
__C.RPN = AttrDict()
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
__C.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)

# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
__C.RPN.ANCHOR_STRIDE = (16,)

# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)

# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
__C.RPN.STRADDLE_THRESH = 0

# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
__C.RPN.FG_IOU_THRESHOLD = 0.7

# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
__C.RPN.BG_IOU_THRESHOLD = 0.3

# Total number of RPN examples per image
__C.RPN.BATCH_SIZE_PER_IMAGE = 256

# Target fraction of foreground (positive) examples per RPN minibatch
__C.RPN.POSITIVE_FRACTION = 0.5

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
__C.RPN.PRE_NMS_TOP_N_TRAIN = 12000

__C.RPN.PRE_NMS_TOP_N_TEST = 6000

# Number of top scoring RPN proposals to keep after applying NMS
__C.RPN.POST_NMS_TOP_N_TRAIN = 2000

__C.RPN.POST_NMS_TOP_N_TEST = 1000

# NMS threshold used on RPN proposals
__C.RPN.NMS_THRESH = 0.7

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
__C.RPN.MIN_SIZE = 0

# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
__C.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000

__C.RPN.FPN_POST_NMS_TOP_N_TEST = 2000

# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
__C.RPN.FPN_POST_NMS_PER_BATCH = True

# Custom rpn head, empty to use default conv or separable conv
__C.RPN.RPN_HEAD = "SingleConvRPNHead" # TODO

# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
__C.RPN.SMOOTH_L1_BETA = 1.0 / 9


# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()

# The head of Fast R-CNN to use
# (e.g., "roi_2mlp_head", "roi_convx_head")
__C.FAST_RCNN.ROI_BOX_HEAD = "roi_2mlp_head"

# Output module of Fast R-CNN head
__C.FAST_RCNN.ROI_BOX_OUTPUT = "Box_output"

# RoI transformation function (e.g., ROIPool or ROIAlign or ROIAlignV2)
__C.FAST_RCNN.ROI_XFORM_METHOD = 'ROIAlign'

# Number of grid sampling points in ROIAlign (usually use 2)
# Only applies to ROIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0

# RoI transform output resolution
# Note: some models may have constraints on what they can use, e.g. they use
# pretrained FC layers like in VGG16, and will ignore this option
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = (14, 14)

# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
__C.FAST_RCNN.FG_IOU_THRESHOLD = 0.5

# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
__C.FAST_RCNN.BG_IOU_THRESHOLD = 0.5

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.FAST_RCNN.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
__C.FAST_RCNN.BATCH_SIZE_PER_IMAGE = 512

# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
__C.FAST_RCNN.POSITIVE_FRACTION = 0.25

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.FAST_RCNN.SCORE_THRESH = 0.05

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.FAST_RCNN.NMS = 0.5

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
__C.FAST_RCNN.DETECTIONS_PER_IMG = 100

# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
__C.FAST_RCNN.SMOOTH_L1_BETA = 1

# ---------------------------------------------------------------------------- #
# Fast R-CNN mlp head options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN.MLP_HEAD = AttrDict()

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.MLP_HEAD.MLP_DIM = 1024

# Use BatchNorm in the Fast R-CNN mlp head
__C.FAST_RCNN.MLP_HEAD.USE_BN = False

# Use GroupNorm in the Fast R-CNN mlp head
__C.FAST_RCNN.MLP_HEAD.USE_GN = False

# Use Weight Standardization in the Fast R-CNN mlp head
__C.FAST_RCNN.MLP_HEAD.USE_WS = False

# ---------------------------------------------------------------------------- #
# Fast R-CNN convfc head options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN.CONVFC_HEAD = AttrDict()

# Dilation
__C.FAST_RCNN.CONVFC_HEAD.DILATION = 1

# Hidden Conv layer dimension when using Convs for the RoI box head
__C.FAST_RCNN.CONVFC_HEAD.CONV_DIM = 256

# Number of stacked Conv layers in the RoI box head
__C.FAST_RCNN.CONVFC_HEAD.NUM_STACKED_CONVS = 4

# Hidden layer dimension when using an MLP for the RoI box head
__C.FAST_RCNN.CONVFC_HEAD.MLP_DIM = 1024

# Use Fast R-CNN Lite (dwconv) to replace standard Fast R-CNN
__C.FAST_RCNN.CONVFC_HEAD.USE_LITE = False

# Use BatchNorm in the Fast R-CNN convfc head
__C.FAST_RCNN.CONVFC_HEAD.USE_BN = False

# Use GroupNorm in the Fast R-CNN convfc head
__C.FAST_RCNN.CONVFC_HEAD.USE_GN = False

# Use Weight Standardization in the Fast R-CNN convfc head
__C.FAST_RCNN.CONVFC_HEAD.USE_WS = False

# ---------------------------------------------------------------------------- #
# Grid R-CNN options
# ---------------------------------------------------------------------------- #
__C.GRID_RCNN = AttrDict()

# Use cascade mapping in Grid R-CNN
__C.GRID_RCNN.CASCADE_MAPPING_ON = False

# TODO: use re-score in Grid R-CNN
__C.GRID_RCNN.RESCORE_ON = False

# The head of Grid R-CNN to use
# (e.g., "roi_2mlp_head", "roi_convx_head")
__C.GRID_RCNN.ROI_GRID_HEAD = "roi_grid_head"

# Output module of Grid R-CNN head
__C.GRID_RCNN.ROI_GRID_OUTPUT = "Grid_output"

# (e.g., "roi_2mlp_head", "roi_convx_head")
__C.GRID_RCNN.ROI_CLS_HEAD = "roi_cls_head"

# Output module of Grid R-CNN head
__C.GRID_RCNN.ROI_CLS_OUTPUT = "Cls_output"

# The number is 192 when Image-across sampling strategy is open
__C.GRID_RCNN.MAX_SAMPLE_NUM_GRID = 96

__C.GRID_RCNN.ACROSS_SAMPLE = False

# RoI transformation function and associated options
__C.GRID_RCNN.ROI_XFORM_METHOD = 'ROIAlign'

# Number of grid sampling points in ROIAlign (usually use 2)
# Only applies to ROIAlign
__C.GRID_RCNN.ROI_XFORM_SAMPLING_RATIO = 2

# RoI transformation function (e.g., ROIPool or ROIAlign)
__C.GRID_RCNN.ROI_XFORM_RESOLUTION_CLS = (7, 7)

__C.GRID_RCNN.ROI_XFORM_RESOLUTION_GRID = (14, 14)

# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
__C.GRID_RCNN.FG_IOU_THRESHOLD = 0.5

# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
__C.GRID_RCNN.BG_IOU_THRESHOLD = 0.5

__C.GRID_RCNN.BATCH_SIZE_PER_IMAGE = 512

# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
__C.GRID_RCNN.POSITIVE_FRACTION = 0.25

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.GRID_RCNN.SCORE_THRESH = 0.03

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.GRID_RCNN.NMS = 0.3

# Loss weight in Grid R-CNN
__C.GRID_RCNN.LOSS_WEIGHT = 15

# Make gt points as cruciform targets
__C.GRID_RCNN.POS_RADIUS = 1

# Number of grid in Grid R-CNN
__C.GRID_RCNN.GRID_POINTS = 9

# The size of feature after pooling in Grid R-CNN
__C.GRID_RCNN.ROI_FEAT_SIZE = 14

# Use random jitter before grid branch
__C.GRID_RCNN.RANDOM_JITTER = False

# Use Finest level ROI for Grid head
__C.GRID_RCNN.FINEST_LEVEL_ROI = False

__C.GRID_RCNN.TARGET_REFINE = False

__C.GRID_RCNN.BETTER_ROI = False

__C.GRID_RCNN.BETTER_ROI_RATIO = 0.25

__C.GRID_RCNN.ENHANCE_FEATURES = False

__C.GRID_RCNN.FUSED_ON = True

__C.GRID_RCNN.EXTEND_ROI = False

__C.GRID_RCNN.OFFSET_ON = False

__C.GRID_RCNN.IOU_HELPER = False

__C.GRID_RCNN.IOU_HELPER_MERGE = False

__C.GRID_RCNN.IOU_LOSS_WEIGHT = 1.0

__C.GRID_RCNN.RESCORE_LOSS_WEIGHT = 1.0

__C.GRID_RCNN.SE_ON = False

# ---------------------------------------------------------------------------- #
# Grid R-CNN cls head options
# ---------------------------------------------------------------------------- #
__C.GRID_RCNN.MLP_HEAD = AttrDict()

# Hidden layer dimension when using an MLP for the RoI box head
__C.GRID_RCNN.MLP_HEAD.MLP_DIM = 1024

# Use BatchNorm in the Grid R-CNN mlp head
__C.GRID_RCNN.MLP_HEAD.USE_BN = False

# Use GroupNorm in the Grid R-CNN mlp head
__C.GRID_RCNN.MLP_HEAD.USE_GN = False

# Use Weight Standardization in the Grid R-CNN mlp head
__C.GRID_RCNN.MLP_HEAD.USE_WS = False

# ---------------------------------------------------------------------------- #
# Grid R-CNN grid head options
# ---------------------------------------------------------------------------- #
__C.GRID_RCNN.GRID_HEAD = AttrDict()

# Number of convolution in Grid R-CNN head
__C.GRID_RCNN.GRID_HEAD.NUM_CONVS = 8

# Number of channels to represent one point in Grid R-CNN head
__C.GRID_RCNN.GRID_HEAD.POINT_FEAT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# Grid R-CNN cascade mapping options
# ---------------------------------------------------------------------------- #
__C.GRID_RCNN.CASCADE_MAPPING_OPTION = AttrDict()

# Number of stages in grid R-CNN
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_NUM = 3

# Stage id for testing
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.TEST_STAGE = 3

# TODO: use ensemble strategy in inference
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.TEST_ENSEMBLE = True

# Loss weight in each stage
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_WEIGHTS = (1.0, 0.5, 0.25)

# Mapping ratio in each stage
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.STAGE_MAPPING_RATIO = (1.0, 0.5, 0.25)

# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.FG_IOU_THRESHOLD = [0.5, 0.6, 0.7]

# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
__C.GRID_RCNN.CASCADE_MAPPING_OPTION.BG_IOU_THRESHOLD = [0.5, 0.6, 0.7]

__C.GRID_RCNN.CASCADE_MAPPING_OPTION.GRID_NUM = (9, 9, 9)

__C.GRID_RCNN.CASCADE_MAPPING_OPTION.RESIZE_ROI = False

# ---------------------------------------------------------------------------- #
# Grid R-CNN re-score options
# ---------------------------------------------------------------------------- #
# TODO
__C.GRID_RCNN.RESCORE_OPTION = AttrDict()

__C.GRID_RCNN.RESCORE_OPTION.KEEP_RATIO = False

# ---------------------------------------------------------------------------- #
# Cascade R-CNN options
# ---------------------------------------------------------------------------- #
__C.CASCADE_RCNN = AttrDict()

# The head of Cascade R-CNN to use
# (e.g., "roi_2mlp_head", "roi_convx_head")
__C.CASCADE_RCNN.ROI_BOX_HEAD = "roi_2mlp_head"

# Output module of Cascade R-CNN head
__C.CASCADE_RCNN.ROI_BOX_OUTPUT = "Box_output"

# Number stages of Cascade R-CNN to use
__C.CASCADE_RCNN.NUM_STAGE = 3

# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
__C.CASCADE_RCNN.FG_IOU_THRESHOLD = [0.5, 0.6, 0.7]

# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
__C.CASCADE_RCNN.BG_IOU_THRESHOLD = [0.5, 0.6, 0.7]

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.CASCADE_RCNN.BBOX_REG_WEIGHTS = ((10., 10., 5., 5.), (20., 20., 10., 10.),
                                     (30., 30., 15., 15.))

# Weights for cascade stages
__C.CASCADE_RCNN.STAGE_WEIGHTS = (1.0, 0.5, 0.25)

# Stage id for testing
__C.CASCADE_RCNN.TEST_STAGE = 3

# Use ensemble results for testing
__C.CASCADE_RCNN.TEST_ENSEMBLE = True

__C.CASCADE_RCNN.RESCORE_ON = False

__C.CASCADE_RCNN.IOU_HELPER = False
__C.CASCADE_RCNN.IOU_HELPER_MERGE = False

__C.CASCADE_RCNN.IOU_LOSS_WEIGHT = 1.0

__C.CASCADE_RCNN.RESCORE_LOSS_WEIGHT = 1.0



# ---------------------------------------------------------------------------- #
# Mask R-CNN options ("MRCNN" means Mask R-CNN)
# ---------------------------------------------------------------------------- #
__C.MRCNN = AttrDict()

# The head of Mask R-CNN to use
# (e.g., "roi_convx_head")
__C.MRCNN.ROI_MASK_HEAD = "roi_convx_head"

# Output module of Mask R-CNN head
__C.MRCNN.ROI_MASK_OUTPUT = "Mask_deconv_output"

# RoI transformation function and associated options
__C.MRCNN.ROI_XFORM_METHOD = 'ROIAlign'

# Number of grid sampling points in ROIAlign (usually use 2)
# Only applies to ROIAlign
__C.MRCNN.ROI_XFORM_SAMPLING_RATIO = 0

# RoI transformation function (e.g., ROIPool or ROIAlign)
__C.MRCNN.ROI_XFORM_RESOLUTION = (14, 14)

# Resolution of mask predictions
__C.MRCNN.RESOLUTION = (28, 28)

# Multi-task loss weight for masks
__C.MRCNN.WEIGHT_LOSS_MASK = 1.0  # TODO

# Whether or not resize and translate masks to the input image.
__C.MRCNN.POSTPROCESS_MASKS = False  # TODO

__C.MRCNN.POSTPROCESS_MASKS_THRESHOLD = 0.5  # TODO

# Mask roi max batch size
__C.MRCNN.ROI_BATCH_SIZE = -1

# Use Finest level ROI for mask head
__C.MRCNN.FINEST_LEVEL_ROI = False

# Multi-task loss weight to use for mask head
__C.MRCNN.LOSS_WEIGHT = 1.0

# Use mask IoU head for mask head
__C.MRCNN.MASKIOU_ON = False

# Loss weight for mask IoU head
__C.MRCNN.MASKIOU_LOSS_WEIGHT = 1.0

# ---------------------------------------------------------------------------- #
# Mask R-CNN convx head options
# ---------------------------------------------------------------------------- #
__C.MRCNN.CONVX_HEAD = AttrDict()

# Hidden Conv layer dimension
__C.MRCNN.CONVX_HEAD.CONV_DIM = 256

# Number of stacked Conv layers in the RoI box head
__C.MRCNN.CONVX_HEAD.NUM_STACKED_CONVS = 4

# Use dilated convolution in the mask head
__C.MRCNN.CONVX_HEAD.DILATION = 1

# Use Mask R-CNN Lite (dwconv) to replace standard Mask R-CNN
__C.MRCNN.CONVX_HEAD.USE_LITE = False

# Use BatchNorm in the Mask R-CNN convx head
__C.MRCNN.CONVX_HEAD.USE_BN = False

# Use GroupNorm in the Mask R-CNN convx head
__C.MRCNN.CONVX_HEAD.USE_GN = False

# Use Weight Standardization in the Mask R-CNN convx head
__C.MRCNN.CONVX_HEAD.USE_WS = False


# ---------------------------------------------------------------------------- #
# Visualization options
# ---------------------------------------------------------------------------- #
__C.VIS = AttrDict()

# Dump detection visualizations
__C.VIS.ENABLED = False

# Score threshold for visualization
__C.VIS.VIS_TH = 0.9

# ---------------------------------------------------------------------------- #
# Show box options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_BOX = AttrDict()

# Visualizing detection bboxes
__C.VIS.SHOW_BOX.ENABLED = True

# Visualization color scheme
# 'green', 'category' or 'instance'
__C.VIS.SHOW_BOX.COLOR_SCHEME = 'green'

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
__C.VIS.SHOW_BOX.COLORMAP = 'COCO81'

# Border thick
__C.VIS.SHOW_BOX.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show class options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_CLASS = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_CLASS.ENABLED = True

# Default: gray
__C.VIS.SHOW_CLASS.COLOR = (218, 227, 218)

# Font scale of class string
__C.VIS.SHOW_CLASS.FONT_SCALE = 0.45

# ---------------------------------------------------------------------------- #
# Show segmentation options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_SEGMS = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_SEGMS.ENABLED = True

# Whether show mask
__C.VIS.SHOW_SEGMS.SHOW_MASK = True

# False = (255, 255, 255) = white
__C.VIS.SHOW_SEGMS.MASK_COLOR_FOLLOW_BOX = True

# Mask ahpha
__C.VIS.SHOW_SEGMS.MASK_ALPHA = 0.4

# Whether show border
__C.VIS.SHOW_SEGMS.SHOW_BORDER = True

# Border color, (255, 255, 255) for white, (0, 0, 0) for black
__C.VIS.SHOW_SEGMS.BORDER_COLOR = (255, 255, 255)

# Border thick
__C.VIS.SHOW_SEGMS.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show keypoints options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_KPS = AttrDict()

# Visualizing detection keypoints
__C.VIS.SHOW_KPS.ENABLED = True

# Keypoints threshold
__C.VIS.SHOW_KPS.KPS_TH = 2

# Default: white
__C.VIS.SHOW_KPS.KPS_COLOR_WITH_PARSING = (255, 255, 255)

# Keypoints alpha
__C.VIS.SHOW_KPS.KPS_ALPHA = 0.7

# Link thick
__C.VIS.SHOW_KPS.LINK_THICK = 2

# Circle radius
__C.VIS.SHOW_KPS.CIRCLE_RADIUS = 3

# Circle thick
__C.VIS.SHOW_KPS.CIRCLE_THICK = -1

# ---------------------------------------------------------------------------- #
# Show parsing options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_PARSS = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_PARSS.ENABLED = True

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
__C.VIS.SHOW_PARSS.COLORMAP = 'CIHP20'

# Parsing alpha
__C.VIS.SHOW_PARSS.PARSING_ALPHA = 0.4

# Whether show border
__C.VIS.SHOW_PARSS.SHOW_BORDER = True

# Border color
__C.VIS.SHOW_PARSS.BORDER_COLOR = (255, 255, 255)

# Border thick
__C.VIS.SHOW_PARSS.BORDER_THICK = 1

# ---------------------------------------------------------------------------- #
# Show uv options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_UV = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_UV.ENABLED = True

# Whether show border
__C.VIS.SHOW_UV.SHOW_BORDER = True

# Border thick
__C.VIS.SHOW_UV.BORDER_THICK = 6

# Grid thick
__C.VIS.SHOW_UV.GRID_THICK = 2

# Grid lines num
__C.VIS.SHOW_UV.LINES_NUM = 15


# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set()

# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'PIXEL_MEAN': 'PIXEL_MEANS',
    'PIXEL_STD': 'PIXEL_STDS',
}


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    # if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
    #     __C.RPN.RPN_ON = True
    # if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
    #     __C.TEST.PRECOMPUTED_PROPOSALS = False
    # if not __C.FPN.FPN_ON:
    #     __C.MRCNN.FINEST_LEVEL_ROI = False
    #     __C.KRCNN.FINEST_LEVEL_ROI = False
    #     __C.PRCNN.FINEST_LEVEL_ROI = False
    #     __C.UVRCNN.FINEST_LEVEL_ROI = False
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
            format(full_key, new_key, msg)
    )

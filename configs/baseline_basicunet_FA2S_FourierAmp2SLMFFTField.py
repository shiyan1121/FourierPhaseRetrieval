import os
import time
from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.projects.segmentation.data import (BaseDataset,
                                                   build_train_loader,
                                                   build_test_loader)
from detectron2.projects.segmentation.data import parse_json_annotation_file
from detectron2.projects.segmentation.transforms import RandomCrop, CenterCrop

from ..common.optim import AdamW as optimizer
from ..common.optim import grad_clippers
from ..common.schedule import multi_step_scheduler as lr_scheduler

from ..data.data_mapper_FourierAmp2SLMFFTField import FieldGenerationDataMapper
from ..evaluation.evaluator import FieldGnrtEvaluator
from ..evaluation.split_combine import SplitCombiner
from ..modeling.field_gnrt import FieldGenerator
from ..modeling.head_unet import FieldGnrtUNetHead
from ..modeling.backbone.BasicUNet import BasicUNetBackbone
from ..common.losses import (LossList, MSEloss, MS_SSIMloss, WrappedMSEloss, Circularloss, L1loss, FMSEloss)

# ================================================================
# output_dir
# ================================================================
OUTPUT_DIR = 'E:\AIPoweredOptics_Fourier2SLM\code\Fourier2SLM\\results'

file_name, _ = os.path.splitext(os.path.basename(__file__))
creat_time = time.strftime('%y%m%d', time.localtime(time.time()))

output_dir = os.path.join(OUTPUT_DIR, f'{file_name}_{creat_time}')
os.makedirs(output_dir, exist_ok=True)

# ================================================================
# 设置 global variable
# ================================================================
INIT_CHECKPOINT = ''
model = 'baseline_basicunet_FA2S_FourierAmp2SLMFFTField_250225_zero'
INIT_CHECKPOINT = os.path.join(OUTPUT_DIR, model, 'model_final.pth')

ANNO_FILE_TRAIN = 'E:\AIPoweredOptics_Fourier2SLM\data\jsons\Fourier2SLM_EMNIST_zero_PR_train.json'
ANNO_FILE_VALID = 'E:\AIPoweredOptics_Fourier2SLM\data\jsons\Fourier2SLM_EMNIST_zero_rand_PR_valid.json'
META, DATA_LIST = parse_json_annotation_file(ANNO_FILE_TRAIN)
TRANSFORM_FIELD = {'image': 'image', 'label': 'segmentation'}

# training params
BATCH_SIZE_PER_GPU=8
BATCH_SIZE_PER_GPU_VALID=1
NUM_WORKERS=2
GPU_NUM = 1
DATA_NUM = len(DATA_LIST)

TRAIN_EPOCHS = 50
TRAIN_REPEAT = 1
EPOCH_ITERS = (DATA_NUM * TRAIN_REPEAT) // (GPU_NUM * BATCH_SIZE_PER_GPU)
MAX_ITERS = TRAIN_EPOCHS * EPOCH_ITERS
AMP_ENABLED = True
EVAL_EPOCH = 20
SAVE_EPOCH = TRAIN_EPOCHS // 5
LOG_ITER = 5
GRAD_CLIPPER = grad_clippers.grad_value_clipper

# dataloader parameters
# RandomCrop
CROP_SIZE = [400, 400]
RESIZE_SIZE = None
CENTER = [200, 200]


# split combine
SPLIT_COMBINE_ENABLE = False
EVAL_CROP_SIZE = CROP_SIZE
STRIDE = [x//2 for x in CROP_SIZE]
SIGMA = 0.25
COMBINE_METHOD = 'gw'

# optimizer parameters
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
LR_VALUES = [0.1, 0.01, 0.001, 0.0001]
LR_MILESTONES = [EPOCH_ITERS*5, EPOCH_ITERS*25, EPOCH_ITERS*50, EPOCH_ITERS*75]
WARMUP_ITER = 0.05*EPOCH_ITERS

# model parameters
INPUT_CHANNEL = 1
OUTPUT_CHANNEL = 2
FEATS = [64, 128, 256, 512]
PRETRAINED = False
NORM_PRAMS = [[0]*INPUT_CHANNEL, [1]*INPUT_CHANNEL]
LOSS_FUNCTION = L(LossList)(
    losses=[MSEloss()],
    weights=[1]
)

# ================================================================
# 设置 dataloader
# ================================================================
dataloader = OmegaConf.create()

dataloader.train = L(build_train_loader)(
    dataset=L(BaseDataset)(anno_file=ANNO_FILE_TRAIN),
    mapper=L(FieldGenerationDataMapper)(
        transforms=[
            # L(RandomCrop)(
            #     crop_size = CROP_SIZE,
            #     fields = TRANSFORM_FIELD
            # )
        ]
    ),
    batch_size=BATCH_SIZE_PER_GPU,
    num_workers=NUM_WORKERS,
)

dataloader.test = L(build_test_loader)(
    dataset=L(BaseDataset)(anno_file=ANNO_FILE_VALID),
    mapper=L(FieldGenerationDataMapper)(
        transforms=[
        #     L(CenterCrop)(
        #         crop_size = CROP_SIZE,
        #         fields = {'image': 'image'})
        ]
    ),
    batch_size=BATCH_SIZE_PER_GPU_VALID,
    num_workers=NUM_WORKERS,
)

dataloader.evaluator = [
    L(FieldGnrtEvaluator)(
        output_dir = output_dir,
    )
]

# ================================================================
# 设置 model
# ================================================================
model = L(FieldGenerator)(
        backbone = L(BasicUNetBackbone)(
            input_channel=INPUT_CHANNEL,
            features = FEATS
        ),
        head = L(FieldGnrtUNetHead)(
            loss_function = LOSS_FUNCTION,
            in_channel = FEATS[0],
            out_channel = OUTPUT_CHANNEL,
            is_drop = False
        ),
        pixel_mean = NORM_PRAMS[0],
        pixel_std = NORM_PRAMS[1],
        resize_size = RESIZE_SIZE,
        output_dir = output_dir
)


# ================================================================
# 设置 optimizer 和 scheduler
# ================================================================
optimizer.lr = LEARNING_RATE
optimizer.weight_decay = WEIGHT_DECAY
optimizer.eps = 1e-6

# # multi step scheduler
lr_scheduler.values = LR_VALUES
lr_scheduler.milestones = LR_MILESTONES

# cosine step scheduler
# lr_scheduler.start = LEARNING_RATE
# lr_scheduler.end = LEARNING_RATE * 0.001

lr_scheduler.max_iter = MAX_ITERS
lr_scheduler.warmup_iter = WARMUP_ITER

# ================================================================
# 设置 train
# ================================================================
train=dict(
    output_dir=output_dir,
    init_checkpoint=INIT_CHECKPOINT,
    max_iter=MAX_ITERS,
    amp=dict(
        enabled=AMP_ENABLED,
        grad_clipper=GRAD_CLIPPER,
    ),
    ddp=dict(
        broadcast_buffer=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(
        period=EPOCH_ITERS * SAVE_EPOCH,
        max_to_keep=100,
    ),
    split_combine = dict(
        enabled = SPLIT_COMBINE_ENABLE,
        split_combiner = L(SplitCombiner)(
            crop_size = CROP_SIZE,
            stride = STRIDE,
            combine_method = COMBINE_METHOD,
            sigma = SIGMA,
            device = 'cuda'
        )
    ),
    eval_period=EPOCH_ITERS * EVAL_EPOCH,
    log_period=LOG_ITER,
    device='cuda',
)
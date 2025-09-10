# DATA
from torch.utils.data import DataLoader
from datareader import CUB_200_2011
from einops import rearrange
# MODEL
from model import LayerNormResNet50
# LOSS OPTIMIZER
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# SEED
from utils import SET_SEED

import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import json
import cv2
import os
'''
     Pre   Post  Acc
Adan False False 84.207
Adan True  False 84.104
Adan False True  84.639
Adan True  True  84.518
'''
MODEL_PATH = "./records/resnet50-PATCH_0_conlrmin_False_True_600_Adan/models/best_acc_model.pth" # resnet50-PATCH_0_mask_conlr_anchorlr
PRELAYER_NORM = False
POSTLAYER_NORM = True
CUB_200_2011_DATASETS_TEST  = CUB_200_2011(DATAROOT = '../20250807/CUB_200_2011/', MODE = 'test' , IMAGESIZE = 448, RANDOM_PATCH_SIZE = 112)
TEST_LOADER   = DataLoader(CUB_200_2011_DATASETS_TEST , batch_size = 16, shuffle = False)

CLASSES_NAME  = CUB_200_2011_DATASETS_TEST.CLASSES_NAME
CLASS_NUMBERS = len(CLASSES_NAME)

MODEL = LayerNormResNet50(
    num_classes = 200, 
    prelayernorm = PRELAYER_NORM,
    postlayernorm = POSTLAYER_NORM
)
MODEL.cuda()
# TEST INPUT OUTPUT AND INIT LAYERNORM
input_tensor = torch.randn(4, 3, 448, 448).cuda()
output = MODEL(input_tensor)
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL.cuda()
MODEL.eval()
ONE_EPOCH_LOSS, ONE_EPOCH_ACC = 0.0, 0.0
with torch.no_grad():
    for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL, _) in enumerate(tqdm(TEST_LOADER)):
        IMAGE_TF1, IMAGE_TF2, LABEL = IMAGE_TF1.cuda(), IMAGE_TF2.cuda(), LABEL.cuda()
        _, PREDICTIONS = MODEL(IMAGE_TF1)
        ONE_EPOCH_ACC  += (PREDICTIONS.argmax(dim=1) == LABEL).sum().item()
        # Visualization Mask
ONE_EPOCH_LOSS /= CUB_200_2011_DATASETS_TEST.__len__()
ONE_EPOCH_ACC  /= CUB_200_2011_DATASETS_TEST.__len__()
print(ONE_EPOCH_ACC)
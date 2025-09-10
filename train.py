# DATA
from torch.utils.data import DataLoader
from datareader import CUB_200_2011
from einops import rearrange
# from loss import SemiHardTripletLoss
# MODEL
from model import LayerNormResNet50
# LOSS OPTIMIZER
import torch
import torch.nn as nn
import torch.nn.functional as F
from adan_pytorch import Adan
import torch.optim as optim
# SEED
from utils import SET_SEED

import matplotlib.pyplot as plt
# from loss import Alignment_Loss
from tqdm import tqdm
import numpy as np
import json
import cv2
import os

if __name__ == '__main__':
    EPOCHS             = 70
    BATCH_SIZE         = 12
    SEED               = 42
    NUM_WORKERS        = 4
    LEARNING_RATE      = 1e-4
    END_LEARNING_RATE  = 1e-7
    PRELAYER_NORM      = False
    POSTLAYER_NORM     = True
    BACKBONE           = "resnet50"
    # PATCH CONTRASTIVE PRELAYERNORM POSTLAYERNORM
    RANDOM_PATCH_SIZE  = f'0_conlrmin_{PRELAYER_NORM}_{POSTLAYER_NORM}_600_Adan'
    SET_SEED(SEED)

    MODEL_NAME = f"{BACKBONE}-PATCH_{RANDOM_PATCH_SIZE}"
    CUB_200_2011_DATASETS_TRAIN = CUB_200_2011(DATAROOT = '../20250807/CUB_200_2011/', MODE = 'train', IMAGESIZE = 448, RANDOM_PATCH_SIZE = RANDOM_PATCH_SIZE)
    CUB_200_2011_DATASETS_TEST  = CUB_200_2011(DATAROOT = '../20250807/CUB_200_2011/', MODE = 'test' , IMAGESIZE = 448, RANDOM_PATCH_SIZE = RANDOM_PATCH_SIZE)
    TRAIN_LOADER  = DataLoader(CUB_200_2011_DATASETS_TRAIN, batch_size = BATCH_SIZE, shuffle = True , num_workers = NUM_WORKERS)
    TEST_LOADER   = DataLoader(CUB_200_2011_DATASETS_TEST , batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
    CLASSES_NAME  = CUB_200_2011_DATASETS_TRAIN.CLASSES_NAME
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

    pretrained_dict = torch.load('best_acc_model-old.pth')
    model_dict = MODEL.state_dict()
    filtered_dict = {}

    # UPDATE PRETRAINED MODEL
    for k, v in pretrained_dict.items():
        new_key =  f"model.{k}"# f"{k}"# f"model.{k}"
        if new_key in model_dict and model_dict[new_key].shape == v.shape:
            filtered_dict[new_key] = v
            print(f"Loading: {new_key} with shape {v.shape}")
        else:
            if new_key not in model_dict:
                print(f"Skipping: {k} -> {new_key} (not found in model)")
            else:
                print(f"Skipping: {k} -> {new_key} (shape mismatch: model={model_dict[new_key].shape}, pretrained={v.shape})")
    model_dict.update(filtered_dict)
    MODEL.load_state_dict(model_dict)
    print(f"Successfully loaded {len(filtered_dict)}/{len(pretrained_dict)} parameters for finetune")

    # LOSS and OPTIMIZER
    CRITERION = nn.CrossEntropyLoss()
    # OPTIMIZER = optim.AdamW(MODEL.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4)

    OPTIMIZER = Adan(
        MODEL.parameters(), 
        lr = LEARNING_RATE, 
        betas = (0.02, 0.08, 0.01), 
        weight_decay = 1e-4,
    )
    # optim.AdamW(MODEL.parameters(), lr = LEARNING_RATE, weight_decay = 1e-4)
    # TRIPLET = SemiHardTripletLoss()
    SCHEDULER = optim.lr_scheduler.CosineAnnealingWarmRestarts(OPTIMIZER, T_0 = 10, T_mult = 2, eta_min = END_LEARNING_RATE)

    # RECORD
    RECORDS_PATH        = f"./records/{MODEL_NAME}/"
    os.makedirs(f"{RECORDS_PATH}", exist_ok=True)
    MODEL_PATH          = f"./records/{MODEL_NAME}/models/"
    VISUALIZATION_PATH  = f"./records/{MODEL_NAME}/visualization/" # TEST DATASETS VISUALIZATION
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    RECORDS = {
        "TRAIN_LOSSES"       : [],
        "TEST_LOSSES"        : [],
        "TRAIN_ACCS"         : [],
        "TEST_ACCS"          : [],
        "TRAIN_BATCH_LOSSES" : [],
        "TEST_BATCH_LOSSES"  : [],
        "TRAIN_BATCH_ACCS"   : [],
        "TEST_BATCH_ACCS"    : [],
        "CONTRASTIVE_LOSSES" : [],
        "TRIPLET_LOSSES"     : [],
    }
    MIN_LOSS = float('inf')
    MAX_ACC  = 0.0

    for EPOCH in range(1, EPOCHS + 1):
        # TRAIN
        MODEL.train()
        ONE_EPOCH_LOSS, ONE_EPOCH_ACC = 0.0, 0.0
        for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL_1, LABEL_2) in enumerate(tqdm(TRAIN_LOADER)):
            IMAGE_TF1, IMAGE_TF2, LABEL_1, LABEL_2 = IMAGE_TF1.cuda(), IMAGE_TF2.cuda(), LABEL_1.cuda(), LABEL_2.cuda()
            IMAGE = torch.cat([IMAGE_TF1, IMAGE_TF2], dim=0)

            OPTIMIZER.zero_grad()
            FEATURE_AVG, PREDICTIONS = MODEL(IMAGE)
            B, C = FEATURE_AVG.shape# B, C, _, _ = FEATURE_AVG.shape
            LABEL_GATE = (LABEL_1 == LABEL_2)

            # same class
            CONTRASTIVE_LOSS = 1 - torch.squeeze((((F.cosine_similarity(FEATURE_AVG[:B//2,:], FEATURE_AVG[B//2:,:], dim = 1)/2) + 0.5))) * LABEL_GATE.float()
            # different class
            # CONTRASTIVE_LOSS = torch.squeeze((((F.cosine_similarity(FEATURE_AVG[:B//2,:,:,:], FEATURE_AVG[B//2:,:,:,:], dim = 1)/2) + 0.5))) * (1 - LABEL_GATE.float())
            # TRIPLET_LOSS = TRIPLET(torch.squeeze(FEATURE_AVG), torch.cat([LABEL, LABEL], dim = 0))
            PREDICTIONS = PREDICTIONS[:B//2]
            # if EPOCH > 1:
            #     FEATURE_AVG_FLAT = rearrange(FEATURE_AVG[:B//2], 'b c h w -> b (c h w)')
            #     ANCHOR_BATCH = torch.stack([ANCHOR_CENTERS[cls.item()].cuda() for cls in LABEL], dim=0)
            #     ANCHOR_LOSS = 1 - ((F.cosine_similarity(ANCHOR_BATCH, FEATURE_AVG_FLAT, dim=1) / 2) + 0.5)
            #     LOSS = CRITERION(PREDICTIONS, LABEL) + ANCHOR_LOSS.mean() + CONTRASTIVE_LOSS.mean()
            # else:
            LOSS = CRITERION(PREDICTIONS, LABEL_1) + CONTRASTIVE_LOSS.mean() # TRIPLET_LOSS # + CONTRASTIVE_LOSS.mean()
            LOSS.backward()
            OPTIMIZER.step()

            # RECORDS
            ONE_EPOCH_LOSS += LOSS.item() * IMAGE_TF1.size(0)
            ONE_EPOCH_ACC  += (PREDICTIONS.argmax(dim=1) == LABEL_1).sum().item()
            RECORDS["CONTRASTIVE_LOSSES"].append(CONTRASTIVE_LOSS.mean().item())
            RECORDS["TRAIN_BATCH_LOSSES"].append((LOSS.item() * IMAGE_TF1.size(0))/IMAGE_TF1.size(0))
            RECORDS["TRAIN_BATCH_ACCS"].append((PREDICTIONS.argmax(dim=1) == LABEL_1).sum().item()/IMAGE_TF1.size(0))
        ONE_EPOCH_LOSS /= CUB_200_2011_DATASETS_TRAIN.__len__()
        ONE_EPOCH_ACC  /= CUB_200_2011_DATASETS_TRAIN.__len__()
        RECORDS["TRAIN_LOSSES"].append(ONE_EPOCH_LOSS)
        RECORDS["TRAIN_ACCS"].append(ONE_EPOCH_ACC)
        tqdm.write(f"Epoch {EPOCH}/{EPOCHS} - Train Loss: {ONE_EPOCH_LOSS:.4f}, Train Acc: {ONE_EPOCH_ACC:.4f}, LR: {OPTIMIZER.param_groups[0]['lr']:.6f}")
        SCHEDULER.step()
        if EPOCH in [10, 30]:
            LEARNING_RATE = LEARNING_RATE / 10
            END_LEARNING_RATE = END_LEARNING_RATE / 10
            OPTIMIZER = Adan(
                MODEL.parameters(), 
                lr = LEARNING_RATE, 
                betas = (0.02, 0.08, 0.01), 
                weight_decay = 1e-4,
            )
            if EPOCH == 10:
                SCHEDULER = optim.lr_scheduler.CosineAnnealingWarmRestarts(OPTIMIZER, T_0 = 20, T_mult = 2, eta_min = END_LEARNING_RATE)
            if EPOCH == 30:
                SCHEDULER = optim.lr_scheduler.CosineAnnealingWarmRestarts(OPTIMIZER, T_0 = 40, T_mult = 2, eta_min = END_LEARNING_RATE)
        # FIND ANCHOR CENTER
        MODEL.eval()
        '''
        ANCHOR_CENTERS = {i: [] for i in range(CLASS_NUMBERS)}
        with torch.no_grad():
            for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL) in enumerate(tqdm(TRAIN_LOADER)):
                IMAGE_TF1, IMAGE_TF2, LABEL = IMAGE_TF1.cuda(), IMAGE_TF2.cuda(), LABEL.cuda()
                PREDICTIONS = MODEL.GET_LATENT_FEATURE(IMAGE_TF1)
                FEATURE_AVG = PREDICTIONS['feature']
                FEATURE_AVG = rearrange(FEATURE_AVG, 'b c h w -> b (c h w)')
                for i in range(LABEL.size(0)):
                    ANCHOR_CENTERS[LABEL[i].item()].append(FEATURE_AVG[i].detach().cpu())
        # CALCULATE ANCHOR CENTER
        for i in range(CLASS_NUMBERS):
            if len(ANCHOR_CENTERS[i]) > 0:
                ANCHOR_CENTERS[i] = torch.stack(ANCHOR_CENTERS[i]).mean(dim = 0)
        '''
        # TEST
        MODEL.eval()
        ONE_EPOCH_LOSS, ONE_EPOCH_ACC = 0.0, 0.0
        with torch.no_grad():
            for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL, _) in enumerate(tqdm(TEST_LOADER)):
                IMAGE_TF1, IMAGE_TF2, LABEL = IMAGE_TF1.cuda(), IMAGE_TF2.cuda(), LABEL.cuda()
                FEATURE_AVG, PREDICTIONS = MODEL(IMAGE_TF1)
                LOSS = CRITERION(PREDICTIONS, LABEL)
                ONE_EPOCH_LOSS += LOSS.item() * IMAGE_TF1.size(0)
                ONE_EPOCH_ACC  += (PREDICTIONS.argmax(dim=1) == LABEL).sum().item()
                RECORDS["TEST_BATCH_LOSSES"].append((LOSS.item() * IMAGE_TF1.size(0))/IMAGE_TF1.size(0))
                RECORDS["TEST_BATCH_ACCS"].append((PREDICTIONS.argmax(dim=1) == LABEL).sum().item()/IMAGE_TF1.size(0))
                # Visualization Mask
                '''
                if BATCH_IDX == 0:
                    LATENT_FEATURES = MODEL.GET_LATENT_FEATURE(IMAGE_TF1)
                    LATENT_FEATURES['l1'] = LATENT_FEATURES['l1'].cpu()[:, -1, :, :][0]
                    LATENT_FEATURES['l2'] = LATENT_FEATURES['l2'].cpu()[:, -1, :, :][0]
                    LATENT_FEATURES['l3'] = LATENT_FEATURES['l3'].cpu()[:, -1, :, :][0]
                    LATENT_FEATURES['l4'] = LATENT_FEATURES['l4'].cpu()[:, -1, :, :][0]
                    # Tensor to cv2
                    LATENT_FEATURES['l1'] = (LATENT_FEATURES['l1'].numpy() * 255).astype(np.uint8)
                    LATENT_FEATURES['l2'] = (LATENT_FEATURES['l2'].numpy() * 255).astype(np.uint8)
                    LATENT_FEATURES['l3'] = (LATENT_FEATURES['l3'].numpy() * 255).astype(np.uint8)
                    LATENT_FEATURES['l4'] = (LATENT_FEATURES['l4'].numpy() * 255).astype(np.uint8)
                    cv2.imwrite(f"{VISUALIZATION_PATH}/EPOCH_{EPOCH}_l1_mask.png", LATENT_FEATURES['l1'])
                    cv2.imwrite(f"{VISUALIZATION_PATH}/EPOCH_{EPOCH}_l2_mask.png", LATENT_FEATURES['l2'])
                    cv2.imwrite(f"{VISUALIZATION_PATH}/EPOCH_{EPOCH}_l3_mask.png", LATENT_FEATURES['l3'])
                    cv2.imwrite(f"{VISUALIZATION_PATH}/EPOCH_{EPOCH}_l4_mask.png", LATENT_FEATURES['l4'])
                '''
        ONE_EPOCH_LOSS /= CUB_200_2011_DATASETS_TEST.__len__()
        ONE_EPOCH_ACC  /= CUB_200_2011_DATASETS_TEST.__len__()
        RECORDS["TEST_LOSSES"].append(ONE_EPOCH_LOSS)
        RECORDS["TEST_ACCS"].append(ONE_EPOCH_ACC)
        tqdm.write(f"Epoch {EPOCH}/{EPOCHS} - Test Loss: {ONE_EPOCH_LOSS:.4f}, Test Acc: {ONE_EPOCH_ACC:.4f}")

        # SAVE MODEL
        if ONE_EPOCH_LOSS < MIN_LOSS:
            MIN_LOSS = ONE_EPOCH_LOSS
            torch.save(MODEL.state_dict(), f"{MODEL_PATH}/best_loss_model.pth")
        if ONE_EPOCH_ACC > MAX_ACC:
            MAX_ACC = ONE_EPOCH_ACC
            torch.save(MODEL.state_dict(), f"{MODEL_PATH}/best_acc_model.pth")
        with open(f"{RECORDS_PATH}/records.json", 'w') as f:
            json.dump(RECORDS, f)

    # EVALUATE BEST ACC MODEL
    # Y is Ground Truth, X is Prediction
    CONFUSE_MATRIX = np.zeros((CLASS_NUMBERS, CLASS_NUMBERS), dtype=np.int64)
    MODEL.load_state_dict(torch.load(f"{MODEL_PATH}/best_acc_model.pth"))
    MODEL.eval()
    with torch.no_grad():
        ACC = 0.0
        for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL, _) in enumerate(tqdm(TEST_LOADER)):
            IMAGE_TF1, IMAGE_TF2, LABEL = IMAGE_TF1.cuda(), IMAGE_TF2.cuda(), LABEL.cuda()
            _, PREDICTIONS = MODEL(IMAGE_TF1)
            PREDICTIONS = PREDICTIONS.argmax(dim=1)
            ACC += (PREDICTIONS == LABEL).sum().item()
            for i in range(LABEL.size(0)):
                CONFUSE_MATRIX[LABEL[i].item(), PREDICTIONS[i].item()] += 1

        ACC /= CUB_200_2011_DATASETS_TEST.__len__()
        print(f"Test Best Accuracy Model: {ACC:.4f}")

    CONFUSE_MATRIX = CONFUSE_MATRIX.astype(np.float32)
    for i in range(CLASS_NUMBERS):
        # normalize 0 to 1
        # divide zero problems
        if np.sum(CONFUSE_MATRIX[i, :]) != 0:
            CONFUSE_MATRIX[i, :] = CONFUSE_MATRIX[i, :] / np.sum(CONFUSE_MATRIX[i, :])
        else:
            CONFUSE_MATRIX[i, :] = 0.0
    plt.figure(figsize=(128, 128))
    plt.imshow(CONFUSE_MATRIX, cmap = "coolwarm")
    for i in range(CLASS_NUMBERS):
        # normalize 0 to 1
        # divide zero problems
        for j in range(CLASS_NUMBERS):
            plt.text(i - 0.42, j + 0.15, round(CONFUSE_MATRIX[j, i], 2), fontsize = 10)

    cbar = plt.colorbar(ticks = [round(i * 0.05, 2)  for i in range(21)], shrink = 0.7, aspect = 50, pad = 0.02)
    cbar.ax.tick_params(labelsize = 128)
    cbar.ax.set_ylabel("match accuracy", fontsize = 128, labelpad = 10)
    cbar.ax.set_yticklabels([round(i * 0.05, 2) for i in range(21)])
    plt.clim(0.0, 1.0)
    plt.title("CUB_200_2011 Image Classification Confuse Matrix", pad = 35, fontsize = 128)
    plt.xlabel("Model Prediction", labelpad = 20, fontsize = 128)
    plt.ylabel("Annotation Ground Truth", labelpad = 20, fontsize = 128)
    plt_classes_name = [f"${class_name.replace('_', r'\;')}$" for class_name in CLASSES_NAME]
    plt.xticks(np.arange(0, CLASS_NUMBERS), plt_classes_name, rotation = 90, fontsize = 16)
    plt.yticks(np.arange(0, CLASS_NUMBERS), plt_classes_name, fontsize = 16)
    plt.subplots_adjust(
        top=1.0,
        bottom=0.05,
        left=0.18,
        right=1.0,
        hspace=0.2,
        wspace=0.2
    )
    plt.savefig(f"{VISUALIZATION_PATH}/confuse_matrix_best_accuracy.png", dpi = 300)
    plt.close()
    plt.cla()
    plt.clf()

    # EVALUATE BEST LOSS MODEL
    # Y is Ground Truth, X is Prediction
    CONFUSE_MATRIX = np.zeros((CLASS_NUMBERS, CLASS_NUMBERS), dtype=np.int64)
    MODEL.load_state_dict(torch.load(f"{MODEL_PATH}/best_loss_model.pth"))
    MODEL.eval()
    with torch.no_grad():
        ACC = 0.0
        for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL, _) in enumerate(tqdm(TEST_LOADER)):
            IMAGE_TF1, IMAGE_TF2, LABEL = IMAGE_TF1.cuda(), IMAGE_TF2.cuda(), LABEL.cuda()
            _, PREDICTIONS = MODEL(IMAGE_TF1)
            PREDICTIONS = PREDICTIONS.argmax(dim=1)
            ACC += (PREDICTIONS == LABEL).sum().item()
            for i in range(LABEL.size(0)):
                CONFUSE_MATRIX[LABEL[i].item(), PREDICTIONS[i].item()] += 1
        ACC /= CUB_200_2011_DATASETS_TEST.__len__()
        print(f"Test Best Loss Model: {ACC:.4f}")

    CONFUSE_MATRIX = CONFUSE_MATRIX.astype(np.float32)
    for i in range(CLASS_NUMBERS):
        # normalize 0 to 1
        # divide zero problems
        if np.sum(CONFUSE_MATRIX[i, :]) != 0:
            CONFUSE_MATRIX[i, :] = CONFUSE_MATRIX[i, :] / np.sum(CONFUSE_MATRIX[i, :])
        else:
            CONFUSE_MATRIX[i, :] = 0.0
    plt.figure(figsize=(128, 128))
    plt.imshow(CONFUSE_MATRIX, cmap = "coolwarm")
    for i in range(CLASS_NUMBERS):
        # normalize 0 to 1
        # divide zero problems
        for j in range(CLASS_NUMBERS):
            plt.text(i - 0.42, j + 0.15, round(CONFUSE_MATRIX[j, i], 2), fontsize = 10)

    cbar = plt.colorbar(ticks = [round(i * 0.05, 2)  for i in range(21)], shrink = 0.7, aspect = 50, pad = 0.02)
    cbar.ax.tick_params(labelsize = 128)
    cbar.ax.set_ylabel("match accuracy", fontsize = 128, labelpad = 10)
    cbar.ax.set_yticklabels([round(i * 0.05, 2) for i in range(21)])
    plt.clim(0.0, 1.0)
    plt.title("CUB_200_2011 Image Classification Confuse Matrix", pad = 35, fontsize = 128)
    plt.xlabel("Model Prediction", labelpad = 20, fontsize = 128)
    plt.ylabel("Annotation Ground Truth", labelpad = 20, fontsize = 128)
    plt_classes_name = [f"${class_name.replace('_', r'\;')}$" for class_name in CLASSES_NAME]
    plt.xticks(np.arange(0, CLASS_NUMBERS), plt_classes_name, rotation = 90, fontsize = 16)
    plt.yticks(np.arange(0, CLASS_NUMBERS), plt_classes_name, fontsize = 16)
    plt.subplots_adjust(
        top=1.0,
        bottom=0.05,
        left=0.18,
        right=1.0,
        hspace=0.2,
        wspace=0.2
    )
    plt.savefig(f"{VISUALIZATION_PATH}/confuse_matrix_best_loss.png", dpi = 300)
    plt.close()
    plt.cla()
    plt.clf()
from random import random
from torchvision import transforms as TF
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import random
import torch
import cv2
import os

class SHUFFLE_PATCHES(object):
    def __init__(self, PATCH_SIZE = 224, RANDOM_THRESHOLD = 0.5):
        # PATCH_SIZE random or int
        self.PATCH_SIZE_MODE  = np.random.choice([112, 224]) # PATCH_SIZE # np.random.choice([112, 56, 28, 14, 7])
        self.RANDOM_THRESHOLD = RANDOM_THRESHOLD

    def __call__(self, IMAGE):
        if self.PATCH_SIZE_MODE == 'random':
            self.PATCH_SIZE = np.random.choice([224, 112, 56, 28])
        else:
            self.PATCH_SIZE = self.PATCH_SIZE_MODE
        if torch.rand(1).item() < self.RANDOM_THRESHOLD:
            UNFOLD_PATCH = F.unfold(IMAGE.unsqueeze(dim = 0), kernel_size = self.PATCH_SIZE, stride = self.PATCH_SIZE, padding = 0)
            UNFOLD_PATCH_SHUFFLE = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in UNFOLD_PATCH], dim = 0)
            IMAGE = F.fold(UNFOLD_PATCH_SHUFFLE, output_size = (IMAGE.shape[1], IMAGE.shape[2]), kernel_size = self.PATCH_SIZE, stride = self.PATCH_SIZE, padding = 0)
        return IMAGE.squeeze(0)

def READFILE(FILEPATH):
    f = open(FILEPATH, 'r')
    FILE_ = f.read().split('\n')[:-1]
    f.close()
    return FILE_

def FILE2DICT(FILE_):
    return {LINE.split(' ')[0]:LINE.split(' ')[1] for LINE in FILE_}

def TENSOR2IMAGE(TENSOR):
    '''
        B, C, H, W -> B, H, W, C
    '''
    return rearrange(TENSOR, 'b c h w -> b h w c')

def UNNORMALIZE(TENSOR):
    '''
        B, C, H, W -> B, C, H, W
    '''
    TENSOR = TENSOR * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    TENSOR = TENSOR + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    # TO NUMPY
    TENSOR = TENSOR.detach().cpu().numpy()*255
    TENSOR = TENSOR.astype(np.uint8)
    return TENSOR


class CUB_200_2011(Dataset):
    def __init__(self, DATAROOT = './CUB_200_2011/', MODE = 'train', IMAGESIZE = 384, RANDOM_PATCH_SIZE = 112):
        self.DATAROOT   = DATAROOT
        self.MODE       = MODE
        self.IMAGEFILE  = os.path.join(DATAROOT, 'images.txt')
        self.LABELFILE  = os.path.join(DATAROOT, 'image_class_labels.txt')
        self.TRAINSPLIT = os.path.join(DATAROOT, 'train_test_split.txt')
        CLASSFILE       = os.path.join(DATAROOT, 'classes.txt')

        # AUGMENTATION
        self.IMAGE_SIZE        = IMAGESIZE
        self.RANDOM_PATCH_SIZE = RANDOM_PATCH_SIZE

        self.DATA_DICT = {}
        self.TRAINSPLIT_LIST = FILE2DICT(READFILE(self.TRAINSPLIT))
        self.IMAGEFILE_DICT  = FILE2DICT(READFILE(self.IMAGEFILE))
        self.LABELFILE_DICT  = FILE2DICT(READFILE(self.LABELFILE))
        CLASSFILE_DICT       = FILE2DICT(READFILE(CLASSFILE))
        self.CLASSES_NAME    = [CLASSNAME.split('.')[1] for CLASSNAME in CLASSFILE_DICT.values()]

        for FILEPATH, LABEL, TRAIN_SPLIT in zip(self.IMAGEFILE_DICT.values(), self.LABELFILE_DICT.values(), self.TRAINSPLIT_LIST.values()):
            LABEL = int(LABEL) - 1
            if TRAIN_SPLIT == '1' and MODE == 'train':
                if LABEL not in self.DATA_DICT:
                    self.DATA_DICT[LABEL] = [FILEPATH]
                else:
                    self.DATA_DICT[LABEL].append(FILEPATH)
            elif TRAIN_SPLIT == '0' and MODE == 'test':
                if LABEL not in self.DATA_DICT:
                    self.DATA_DICT[LABEL] = [FILEPATH]
                else:
                    self.DATA_DICT[LABEL].append(FILEPATH)

        if MODE == 'train':
            self.TRAINSPLIT_LIST = [k for k, v in self.TRAINSPLIT_LIST.items() if v == '1']
            
            self.TFs = TF.Compose([
                TF.ToPILImage(),
                TF.Resize((600, 600)),
                TF.RandomCrop((IMAGESIZE, IMAGESIZE)),

                # TF.RandomResizedCrop(size = (self.IMAGE_SIZE, self.IMAGE_SIZE), scale = (0.8, 1.0), ratio = (0.75, 1.333)),
                # TF.Resize((224, 224)),
                TF.RandomHorizontalFlip(),
                # TF.RandomVerticalFlip(),
                # TF.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
                TF.RandomApply([TF.GaussianBlur(kernel_size = (5, 5), sigma = (0.1, 5))], p = 0.1),
                TF.RandomAdjustSharpness(sharpness_factor = 1.5, p = 0.1),
                TF.ToTensor(),
                # SHUFFLE_PATCHES(PATCH_SIZE = self.RANDOM_PATCH_SIZE, RANDOM_THRESHOLD = 0.5),
                TF.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        elif MODE == 'test':
            self.TRAINSPLIT_LIST = [k for k, v in self.TRAINSPLIT_LIST.items() if v == '0']
            self.TFs = TF.Compose([
                TF.ToPILImage(),
                TF.Resize((600, 600)),
                TF.CenterCrop((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                # TF.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                TF.ToTensor(),
                TF.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.TRAINSPLIT_LIST)

    def __getitem__(self, idx):
        IMAGE_ID   = self.TRAINSPLIT_LIST[idx]
        IMAGE_PATH = os.path.join(self.DATAROOT, 'images', self.IMAGEFILE_DICT[IMAGE_ID])
        IMAGE      = cv2.imread(IMAGE_PATH)
        IMAGE      = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
        # if random.random() < 0.5:
        LABEL_ID   = int(self.LABELFILE_DICT[IMAGE_ID]) - 1# np.random.randint(0, 199) # int(self.LABELFILE_DICT[IMAGE_ID]) - 1 # 1 - 200 to 0 - 199
        # else:
        #     LABEL_ID   = int(self.LABELFILE_DICT[IMAGE_ID]) - 1
        '''
        # 448:600
        # 384:510
        '''
        if self.MODE == 'train':
            IMAGE_TF1  = self.TFs(IMAGE)
            IMAGE_PATH2 = self.DATA_DICT[LABEL_ID][np.random.randint(0, len(self.DATA_DICT[LABEL_ID]))]
            IMAGE_PATH2 = os.path.join(self.DATAROOT, 'images', IMAGE_PATH2)
            IMAGE2     = cv2.imread(IMAGE_PATH2)
            IMAGE2     = cv2.cvtColor(IMAGE2, cv2.COLOR_BGR2RGB)
            IMAGE_TF2  = self.TFs(IMAGE2)
        else:
            IMAGE_TF1  = self.TFs(IMAGE)
            IMAGE_PATH2 = self.DATA_DICT[LABEL_ID][np.random.randint(0, len(self.DATA_DICT[LABEL_ID]))]
            IMAGE_PATH2 = os.path.join(self.DATAROOT, 'images', IMAGE_PATH2)
            IMAGE2     = cv2.imread(IMAGE_PATH2)
            IMAGE2     = cv2.cvtColor(IMAGE2, cv2.COLOR_BGR2RGB)
            IMAGE_TF2  = self.TFs(IMAGE2)
        LABEL_1 = torch.tensor(int(self.LABELFILE_DICT[IMAGE_ID]) - 1, dtype=torch.long)
        LABEL_2 = torch.tensor(LABEL_ID, dtype=torch.long)
        return IMAGE_TF1, IMAGE_TF2, LABEL_1, LABEL_2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    CUB_DATASETS = CUB_200_2011(DATAROOT = '../20250807/CUB_200_2011/', MODE = 'train', RANDOM_PATCH_SIZE = 'random')
    CUB_DATALOADER = DataLoader(CUB_DATASETS, batch_size = 16, shuffle = True)
    CLASSES_LIST = CUB_DATASETS.CLASSES_NAME
    for BATCH_IDX, (IMAGE_TF1, IMAGE_TF2, LABEL_1, LABEL_2) in enumerate(CUB_DATALOADER):
        print(BATCH_IDX, IMAGE_TF1.shape, IMAGE_TF2.shape, LABEL_1.shape)
        IMAGE_TF1 = UNNORMALIZE(IMAGE_TF1)
        IMAGE_TF2 = UNNORMALIZE(IMAGE_TF2)
        IMAGE_TF1 = TENSOR2IMAGE(IMAGE_TF1)
        IMAGE_TF2 = TENSOR2IMAGE(IMAGE_TF2)
        LABEL_1 = LABEL_1.detach().cpu().numpy()
        LABEL_2 = LABEL_2.detach().cpu().numpy()
        print(LABEL_1, LABEL_2)
        for i in range(16):
            plt.subplot(4, 8, i*2 + 1)
            plt.imshow(IMAGE_TF1[i])
            plt.title(CLASSES_LIST[LABEL_1[i]])
            plt.axis('off')
            plt.subplot(4, 8, i*2 + 2)
            plt.imshow(IMAGE_TF2[i])
            plt.title(CLASSES_LIST[LABEL_2[i]])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        # break

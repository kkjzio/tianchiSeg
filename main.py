import ssl
import numpy as np
from numpy.core.arrayprint import set_string_function
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
from torch.nn.modules.loss import NLLLoss
from tqdm import tqdm

import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


import albumentations as A


from tools.Rle import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

import segmentation_models_pytorch as smp

EPOCHES = 20
BATCH_SIZE = 32 
IMAGE_SIZE = 512
# IMAGE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 利用albumentations数据扩增
trfm1 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
])

trfm2 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

trfm3 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    # 垂直、左右翻转
    A.transforms.Transpose(p=0.5),
    # 随机旋转+缩放
    A.ShiftScaleRotate(p=0.5),

])

trfm4 = A.Compose([
    A.augmentations.crops.transforms.RandomSizedCrop(min_max_height=[512,512],height=IMAGE_SIZE, width=IMAGE_SIZE, w2h_ratio=1.0, interpolation=1, always_apply=True, p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

trfm5 = A.Compose([
    A.augmentations.crops.transforms.RandomSizedCrop(min_max_height=[512,512],height=IMAGE_SIZE, width=IMAGE_SIZE, w2h_ratio=1.0, interpolation=1, always_apply=True, p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # 随机旋转+缩放
    A.ShiftScaleRotate(p=0.5),
    A.RandomRotate90(),
])

class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode
        
        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])
        
    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            #只对训练集使用T的变换 别对mask使用
            # None给mask加1维度变成[1,512,512]
            return self.as_tensor(augments['image']), augments['mask'][None]
        else:
            return self.as_tensor(img), ''        
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

train_mask = pd.read_csv('train_mask.csv', sep='\t', names=['name', 'mask'])
# 这给数据名前加上索引
train_mask['name'] = train_mask['name'].apply(lambda x: 'train/' + x)

dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm1, False
)

dataset2 = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm2, False
)
dataset3 = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm3, False
)
dataset4 = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm4, False
)
dataset5 = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm5, False
)

dataset = D.ConcatDataset([dataset, dataset2, dataset3,dataset4,dataset5])

valid_idx, train_idx = [], []
for i in range(len(dataset)):
    if i % 7 == 0:
        valid_idx.append(i)
#     else:
    elif i % 7 == 1:
        train_idx.append(i)
        
train_ds = D.Subset(dataset, train_idx)
valid_ds = D.Subset(dataset, valid_idx)

# define training and validation data loaders
loader = D.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

vloader = D.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# def get_model():
#     model = torchvision.models.segmentation.deeplabv3_resnet50(True)
#     # model = torchvision.models.segmentation.fcn_resnet50(True)
# #     pth = torch.load("../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
# #     for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
# #         del pth[key]
#     # 最后输出改成1通道
#     # model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # 将容器中[4]重新初始化 
#     model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) #dlV3
#     return model

model = smp.Unet(
    encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        # output = model(image)['out']
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())
        
    return np.array(losses).mean()


# model = get_model()
model = nn.DataParallel(model)
model.to(DEVICE);

optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(),
#                   lr=1e-4, weight_decay=1e-3,alpha=0.99, eps=1e-04, momentum=0.001)



import lossf.lovasz_losses as L


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits	# 如果BEC带logits则损失函数在计算BECloss之前会自动计算softmax/sigmoid将其映射到[0,1]
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        #x,y=[1,256,256]
        #tp fp fn=[1]  这种sum的用法貌似只能在这里用
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

    
bce_fn=FocalLoss(logits=True)
# bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    # print(y_pred.squeeze(1).shape)
    # print(y_true.shape)
    lov=L.lovasz_softmax(y_pred.sigmoid().squeeze(1), y_true.squeeze(1),classes=[1])
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.7*bce+0.3*dice

# def lov_fn(y_pred, y_true):
#     lov=L.lovasz_softmax(y_pred.sigmoid().squeeze(1), y_true.squeeze(1),classes=[1])
#     return lov


from tools.pytorchtools import EarlyStopping
patience = 7

header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
print(header)


EPOCHES = 200
best_loss = 10
early_stopping = EarlyStopping(patience=patience, verbose=False)
# model.load_state_dict(torch.load('stemp2.pth'))
now_fn=loss_fn
stt=0

for epoch in range(1, EPOCHES+1):
    losses = []
    start_time = time.time()
    # drop和和bn开
    model.train()
    for image, target in tqdm(loader):
        
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        # 清空上次的优化值
        optimizer.zero_grad()     
        # 把数据喂给model
        # [1,256,256]
        # output = model(image)['out']
        output = model(image)

        # print(output.shape)
        # print(target.squeeze(1).shape)
        # 算出loss值
        # loss = loss_fn(output, target)
        loss = now_fn(output, target)
        # 计算参数更新值
        loss.backward()
        # 将参数更新值施加到 model 的 parameters 上
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())
        
    vloss = validation(model, vloader, loss_fn)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time()-start_time)/60**1))
    losses = []
    
    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), 'unet2.pth')
    

    # 早停
    early_stopping(vloss, model,initia=False)
    

    if early_stopping.early_stop:
        print("Early stopping")
        break

    #第一次早停时更换loss
    # if early_stopping.early_stop:
    #     if stt==0 :
    #         print("change loss to lov_fn")
    #         now_fn=lov_fn
    #         stt=1
    #         early_stopping(vloss, model,initia=True)
    #     else:
    #         print("Early stopping")
    #         break


# 训练结束

import os, sys,shutil, gc

from glob import glob
from tqdm import tqdm

import math
import random
from collections import OrderedDict
import warnings

import albumentations as A
import cv2
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW,lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import fbeta_score, average_precision_score, roc_auc_score

def load_image(filename):
    return cv2.imread(filename, 0)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def load_fragment(frag_id):
    frag_dir = f'./frags/train_{frag_id}'
    img_fn_list = sorted(glob(f'{frag_dir}/surface_*'))
    images = np.array([load_image(fn) for fn in img_fn_list])
    frag_mask = load_image(f"{frag_dir}/frag_mask.png").astype(bool).astype(np.uint8)
    ink_mask = load_image(f"{frag_dir}/ink_mask.png").astype(bool).astype(np.uint8)
    
    return images,frag_mask,ink_mask


class DiceLoss(nn.Module):
    """Calculate dice loss."""
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, use_logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_logits = use_logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.use_logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def dice_loss(pred, target):
    smooth = 1e-5  # small constant to avoid division by zero
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice
    return dice_loss

def dice_bce_loss(pred, target, bce_weight=0.5):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    combined_loss = bce_weight * bce_loss + (1 - bce_weight) * dice
    return combined_loss

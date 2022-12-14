import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from utils.dice_score_org import multiclass_dice_coeff_org, dice_coeff
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import argparse
import logging
import os
from torchvision.transforms import functional as R
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.dice_score_org import multiclass_dice_coeff_org, dice_coeff
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
''''Permet d'effectuer la phase de test des images presentes dans dir_img'''

dir_img = Path('./data_test_final/3/imgs/')
dir_mask = Path('./data_test_final/3/masks/')
dir_model= Path('./model5.pth')
img_scale=0.5

def compute_dice(res, gt, label):
    A = gt == label
    B = res == label
    TP = len(np.nonzero(A*B)[0])
    FN = len(np.nonzero(A*(~B))[0])
    FP = len(np.nonzero((~A)*B)[0])
    DICE = 0
    if (FP+2*TP+FN) != 0:
        DICE = float(2)*TP/(FP+2*TP+FN)
    return DICE*100


def compute_dice_exam(seg, mask):
    dice_liver = compute_dice(seg, mask, 1)
    dice_rkidney = compute_dice(seg, mask, 2)
    dice_lkidney = compute_dice(seg, mask, 3)
    dice_spleen = compute_dice(seg, mask, 4)
    return dice_liver, dice_rkidney, dice_lkidney, dice_spleen


def test():

    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=True, **loader_args)


    net = UNet(n_channels=1, n_classes=5, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    net.to(device=device)
    net.load_state_dict(torch.load(dir_model, map_location=device))


    net.eval()
    num_val_batches = len(dataloader)
    dice_score = np.zeros(shape=4)
    compt=0
    list_true=[]
    list_pred=[]
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Testing round', unit='batch', leave=False):

        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type


        with torch.no_grad():
            output = net(image.to(device=device))

            if net.n_classes > 1:
                probs_ = F.softmax(output, dim=1)[0]
                probs = torch.argmax(probs_, dim=0)
                probs = probs.to(torch.uint8)

            else:
                probs = torch.sigmoid(output)[0]


            true_mask=np.array(mask_true[0])

            full_mask = np.array(probs.cpu())

            list_true.append(true_mask)
            list_pred.append(full_mask)

    conc_true=np.stack(list_true)
    conc_pred=np.stack(list_pred)
    print(np.shape(conc_true))


    return(compute_dice_exam(conc_pred,conc_true))



    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    # return dice_score / num_val_batches


if __name__ == '__main__':

    output= test()
    print('Foie:  ', output[0])
    print('Rein droit:  ', output[1])
    print('Rein gauche:  ', output[2])
    print('Rate:  ', output[3])
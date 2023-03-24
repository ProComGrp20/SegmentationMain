#This code is used to evaluate the performance of our U-Net model during training by computing the DICE coefficient 
#and Hausdorff distance between the true and predicted masks 
#It was based on the following project by "milesial" : https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, multiclass_hausdorff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    hausdorff = 0

    #Iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        #Move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            #Predict the mask
            mask_pred = net(image)

            #Convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #Compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                #Compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                #Compute the Hausdorff distance between the true and predicted masks
                hausdorff += multiclass_hausdorff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...])

           

    net.train()

 #Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, hausdorff
    return dice_score / num_val_batches, hausdorff / num_val_batches

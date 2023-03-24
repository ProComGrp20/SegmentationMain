#This code is used to compute the DICE coefficient and Hausdorff distance between the true and predicted masks
#It was based on the following project by "milesial" : https://github.com/milesial/Pytorch-UNet

import torch
from torch import Tensor
import numpy as np

from scipy.spatial.distance import directed_hausdorff


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    #Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    #Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]



def multiclass_hausdorff(input: Tensor, target: Tensor):
    #Average of Hausdorff distance for all classes
    assert input.size() == target.size()
    hausdorff = 0
    for channel in range(input.shape[1]):
        hausdorff += directed_hausdorff(np.asarray(input[0, channel, ...].cpu()), np.asarray(target[0, channel, ...].cpu()))[0]

    return hausdorff / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    #Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
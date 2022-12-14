from os import listdir
from os.path import isfile, join
from PIL.Image import *
from matplotlib import pyplot as plt, image as mimg
import torch
from os import listdir
from os.path import isfile, join
import torchvision.transforms.functional as TF
import random
import numpy as np
from skimage.util import random_noise
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

''' Permet d'effectuer une augmentation de données: rotation, zoom, bruit, translation'''


class SaltPepperTransform:
    """
    Define a custom PyTorch transform to implement
    Salt and Pepper Data Augmentation
    """

    def __init__(self, amount):
        """
        Pass custom parameters to the transform in init

        Parameters
        ----------
        amount : float
            The amount of salt and pepper noise to add to the image sample
        """
        super().__init__()
        self.amount = amount

        # conversion transforms we will use
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, sample):
        """
        Transform the sample when called

        Parameters
        ----------
        sample : PIL.Image
            The image to augment with noise

        Returns
        -------
        noise_img : PIL.Image
            The image with noise added
        """
        salt_img = torch.tensor(random_noise(self.to_tensor(sample),
                                             mode='salt', amount=self.amount))

        return self.to_pil(salt_img)


def My_transforms(image, segmentation):

    #angle = random.randint(-180, 180)
    angle=0
    shift = [random.randint(-30, 30),random.randint(-30, 30)]
    scale=0.8+random.random()*0.5
    image = TF.affine(image, angle,shift,scale,0)
    segmentation = TF.affine(segmentation, angle,shift,scale,0)


    return image, segmentation

def My_transforms_salt(image, segmentation):


    trans=SaltPepperTransform(0.025)
    image=trans(image)

    return image, segmentation



dir_img = "./data3/imgs/"
dir_mask = "./data3/masks/"
out_img="./data3/aug_imgs/"
out_mask="./data3/aug_masks/"


fichiers = [f for f in listdir(dir_img) ]

for name in fichiers:
    for i in [1,2]:

        image=open(dir_img + str(name))
        mask=open(dir_mask + str(name))
        image,mask=My_transforms(image, mask)
        image.save(out_img+"aug_2_"+str(i)+str(name), 'png')
        mask.save(out_mask+"aug_2_"+str(i)+str(name), 'png')



    image=open(dir_img + str(name))
    mask=open(dir_mask + str(name))
    image.save(out_img+str(name), 'png')
    mask.save(out_mask+str(name), 'png')
    image,mask=My_transforms_salt(image, mask)
    image.save(out_img+"aug_2_salt_"+str(name), 'png')
    mask.save(out_mask+"aug_2_salt_"+str(name), 'png')
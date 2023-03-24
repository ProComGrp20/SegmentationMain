#This code is used to prepare our dataset of hand CT images and corresponding masks by converting nifti files into slices
#It was inspired by the following code by "madsendennis" : https://github.com/madsendennis/notebooks/blob/master/volume_segmentation_with_unet/01_Volume-Segmentation-with-UNET_Pre-processing.ipynb

import os, glob
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2


# Data paths
InputPath = './Data/'
imagePath = os.path.join(InputPath, 'Img/')
maskPath = os.path.join(InputPath, 'Mask/')

OutputPath = './data_bones/Slices'
imageSlicePath = os.path.join(OutputPath, 'Img/')
maskSlicePath = os.path.join(OutputPath, 'Mask/')

# CT images normalization constants (HU = Hounsfield units)
MIN_HU = -1000
MAX_HU = 2000
RANGE_HU = MAX_HU - MIN_HU

SLICE_ID = 3

# Normalize image
def normalizeImageIntensity(img):
    img[img < MIN_HU] = MIN_HU
    img[img > MAX_HU] = MAX_HU
    return (img - MIN_HU) / RANGE_HU


# Read image or mask nifti files 
def readNiftiFile(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensity(img)
    else:
        return img
    

# Save image slice to file
def saveImageSlice(img, filename, path):
    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{filename}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')

# Save mask slice to file 
def saveMaskSlice(mask, filename, path):
    mask = np.uint8(mask)  
    fout = os.path.join(path, f'{filename}.png')
    cv2.imwrite(fout, mask)
    print(f'[+] Slice saved: {fout}', end='\r')
    

# Slice image along the Z axis and save
def sliceAndSaveVolumeImage(img, filename, path):
    nb_slices = 0
    (dimx, dimy, dimz) = img.shape
    print(dimx, dimy, dimz)
    nb_slices += dimz
    print('Slicing Z: ')
    for i in range(dimz):
        saveImageSlice(img[:,:,i], filename+f'-slice{str(i).zfill(SLICE_ID)}_z', path)
    return nb_slices

# Slice mask along the Z axis and save
def sliceAndSaveVolumeMask(mask, filename, path):
    nb_slices = 0
    (dimx, dimy, dimz) = mask.shape
    print(dimx, dimy, dimz)
    nb_slices += dimz
    print('Slicing Z: ')
    for i in range(dimz):
        saveMaskSlice(mask[:,:,i], filename+f'-slice{str(i).zfill(SLICE_ID)}_z', path)
    return nb_slices

# Read and process nifti image files
for index, filename in enumerate(sorted(glob.iglob(imagePath+'*.nii'))):
    img = readNiftiFile(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'Patient-'+str(index), imageSlicePath)
    print(f'\n{filename}, {numOfSlices} slices were created \n')
    
# Read and process nifti mask files
for index, filename in enumerate(sorted(glob.iglob(maskPath+'*.nii'))):
    mask = readNiftiFile(filename, False)
    print(filename, mask.shape, np.sum(mask.shape), np.min(mask), np.max(mask), np.unique(mask))
    numOfSlices = sliceAndSaveVolumeMask(mask, 'Patient-'+str(index), maskSlicePath)
    print(f'\n{filename}, {numOfSlices} slices were created \n')

#This code is used to predict bones masks on a set of CT images with a trained U-Net model
#It was based on the following project by "milesial" : https://github.com/milesial/Pytorch-UNet

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

#This function takes an image and a trained model as inputs and returns the mask predicted by the model
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs_ = F.softmax(output, dim=1)[0]  #The softmax function converts the model's output into the probabilities of each pixel to belong to each class
            probs = torch.argmax(probs_, dim=0)   #We then attribute the class with the maximum probability to the pixel
            probs = probs.to(torch.uint8)
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])


        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args): 
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

def get_input_filenames(directory): 
    return os.listdir(directory)

def mask_to_image(mask: np.ndarray):     #This function allows us to visualize a mask 
    if mask.ndim == 2:
        return Image.fromarray((mask.numpy() * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':

    args = get_args()
    in_files = get_input_filenames(args.input[0])
    out_directory = args.output[0]

    net = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device)) #Load a U-Net trained model 

    logging.info('Model loaded!')

    for filename in in_files :         #Prediction of a mask for each image of a chosen directory 
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(args.input[0]+'/'+filename)

        true_mask = Image.open('./data_test_bones/masks/'+filename)    

        mask = predict_img(net=net,                               
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)                        #Mask prediction 

        if not args.no_save:
            out_filename = out_directory+'/'+filename
            result = mask_to_image(mask)
            result.save(out_filename)                      #Saving the mask file
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask, true_mask)        #Image and mask visualization

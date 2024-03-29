#This code is used to plot the image and predicted mask 
#It was based on the following project by "milesial" : https://github.com/milesial/Pytorch-UNet

import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask, true_mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'True mask')
        ax[1].imshow(true_mask)
        ax[2].set_title(f'Output mask')
        ax[2].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

'''Sub module for functions that do visualisation, mainly plotting images'''

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import nibabel as nib


def array_from_path(full_path):
    raise NotImplementedError("Aa yeah, don't know what to tell you, it's not done yet.")
    '''Takes a string path to a .nii.gz image, and returns it as a numpy array.'''
    img = nib.load(full_path).get_fdata()
    return(img)


def show_image(filename=None, full_path=None, z=75, title=None, images=None):
    '''Function that takes a ".nii.gz" filename (from a specific dir) representing an image, and displays it.
    
    There are multiple "modes", or "ways" of using this function:
    
    With filename: Put the name of the file (like \"PP_C123456_..._.nii.gz\") (works on files from our CBTN data)
    
    With full_path: Put the full path of the file. This can be any image.
    
    With images: Put the images in a 4D numpy array'''

    # Load image
    if filename != None:
        img = nib.load("/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/" + filename).get_fdata()
    elif full_path != None:
        img = nib.load(full_path).get_fdata()
    elif images is not None:
        shape = images.shape
        if len(shape) in [0, 1, 2, 3]: raise AssertionError("Incorrect dimensions of images, got " + str(shape))
        if shape[0] == 1 and shape[1] == 1: # Just one image was entered
            fig, axs = plt.subplots(nrows=1, ncols=1)
            axs.imshow(images[0, 0, :, :])
            axs.title("Image " + str(0) + ", " + str(0))
            fig.show
        elif shape[0] == 1 and shape[1] > 1: # One row of images
            fig, axs = plt.subplots(nrows=1, ncols=shape[1])
            for col in range(shape[1]):
                print(col)
                axs[col].imshow(images[0, col, :, :])
                #axs[col].title("Image " + str(0) + ", " + str(col))
            fig.suptitle("Image " + str(0) + ", " + str(col))
            fig.show
        elif shape[0] > 1 and shape[1] == 1: # One column of images
            fig, axs = plt.subplots(nrows=shape[0], ncols=1)
            for row in range(shape[0]):
                axs[row].imshow(images[row, 0, :, :])
                #axs[row].title("Image " + str(row) + ", " + str(0))
            fig.suptitle("Image " + str(row) + ", " + str(0))
            fig.show
        elif shape[0] > 1 and shape[1] > 1: # Multiple rows and columns
            fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1])
            for col in range(shape[1]):
                for row in range(shape[0]):
                    axs[row, col].imshow(images[row, col, :, :], cmap="gray")
                    #axs[row, col].title("Image " + str(row) + ", " + str(col))
                fig.suptitle("Image " + str(row) + ", " + str(col))
                fig.show()
        
        return
    # Plot image
    fig, axs = plt.subplots(nrows=1)
    plt.imshow(img[:, :, z], cmap="gray")

    # Configure title
    if title == None: title = "Image of dim " + str(img.shape) + " at z=" + str(z)
    plt.title(title)
    plt.show()


def show_image_v2(images=None, z=75):
    '''Takes a list structure, representing a grid of images to be plotted. When converted to numpy array, it must be at most 5-dimensional.
    Two dims for rows and columns in the plot, and 3 dims for voxels.'''
    raise NotImplementedError("Sorry, not done")






####################################################################################
#####################   Potential utils functions   ################################
####################################################################################
'''One that takes like a string, or set of strings represtening t1, t1gd, t2, flair
images, and then runs the BSF model on them and produces a segmentation, and plots some
useful plots along the way'''
####################################################################################
####################################################################################
####################################################################################



# DEPRECATED

# def in_out(in_img, out_img):
#     '''Function that takes an input and an output image and plots them side by side'''
#     from matplotlib import pyplot as plt

#     fig, axs = plt.subplots(nrows=1, ncols=2)
#     axs[0].imshow(in_img, cmap="gray")

#     axs[1].imshow(out_img, cmap="gray")
# %%

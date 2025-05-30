'''Sub module for functions that do visualisation, mainly plotting images'''

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import nibabel as nib


def array_from_path(full_path):
    #raise NotImplementedError("Aa yeah, don't know what to tell you, it's not done yet.")
    '''Takes a full path (string) to a .nii.gz image, and returns it as a numpy array.'''
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


# %%

def show_image_v2(images=None, titles=None, maintitle=None, z=75, minimal=True):
    '''Takes a list structure, representing a grid of images to be plotted. When converted to numpy array, it must be at most 5-dimensional.
    Two dims for rows and columns in the plot, and 3 dims for voxels.
    
    images: A list structure like [[img1, img2], [img3, "none"]], where img1, img2, img3 may be numpy arrays, or string paths to images. "none" indicates an empty spot in the grid.
    
    minimal: Whether to remove axis labels, ticks, making the plot minimal'''

    # Change any potential paths to tensors
    for row in range(0, len(images)):
        for col in range(0, len(images[0])):
            # Check if it is an image or a path to an image
            obj = images[row][col]
            if type(obj) == str and obj != "none":
                images[row][col] = array_from_path(obj)
            elif type(obj) == np.ndarray:
                images[row][col] = torch.from_numpy(obj)


    fig = plt.figure()
    
    nrows = len(images)
    ncols = len(images[0])

    # Make the plots
    for index in range(nrows*ncols):
        axs = fig.add_subplot(nrows,ncols,index+1)

        row_index = index % ncols
        col_index = index % nrows

        row_index = (index+1 - 1) // ncols  # Integer division for row index
        col_index = (index+1 - 1) % ncols

        #print("Index+1 is "+str(index+1), " , r_ind: "+str(row_index), " , c_ind: "+str(col_index))

        if type(images[row_index][col_index]) == str: #"none": # Hide empty figs
            axs.axis("off")
        else:
            axs.imshow(images[row_index][col_index][:, :, z], cmap="gray")

            if titles != None: axs.set_title(titles[row_index][col_index])
            else: axs.set_title("Fig "+str(index+1))
        
        # TEst
        #axs.spines["left"].set_visible(False)
        if minimal:
            axs.set_axis_off()
            plt.tight_layout(rect=(0,0,1,0.9))
            
    
    fig.suptitle(maintitle if maintitle != None else "Title missing due to pixel shortage\n", fontsize=16)
    fig.show()


# Data
import numpy as np
import matplotlib.pyplot as plt
import torch


#data = torch.rand(20,20, 100)
#show_image_v2([[data, data, data], [data, data, "none"]], [["a", "b", "c"], ["d", "e", "f"]])


#%%
#[["path", 123], [123, 123]]

# %%

def list_shape(lis):
    shape_obj = [None for i in range(len(lis))]

    n_objects = 0
    for i in range(len(lis)):
        if type(lis[i]) == list:
            shape_obj[i] = len(lis[i])
        else: n_objects += 1
    
    return(shape_obj)

#print(list_shape([1,2,3]))


# %%

def get_row_col(num, nrows, ncols):
    row = (num - 1) // ncols  # Integer division for row index
    col = (num - 1) % ncols   # Modulo for column index
    return row, col

# Example usage:
# nrows, ncols = 6, 2  # 3 rows, 4 columns
# num = 9

# row, col = get_row_col(num, nrows, ncols)
# print(f"Number {num} is at row {row}, column {col}")


import numpy as np
import pandas as pd

def unique(series, ascending=False):
    '''Returns the unique values, along with the corresponding counts for a series'''

    # If at least one element is str, then the others should be treated as str as well.
    # The others might be nan for example, which does not play nicely with str.
    if str in set(type(i) for i in series):
        series = [str(i) for i in series]
    
    # Count frequencies
    counts = np.unique(series, return_counts=True)
    df = pd.DataFrame({"Values": counts[0], "Counts":counts[1]})
    df.sort_values(inplace=True, by="Counts", ascending=ascending)
    return df



def observation_summary(observations, title=""):
    '''
    Provide a summary of a dataframe representing our observations.

    Assumes columns ..... exist.
    '''
    print(f"############################   {title} observation summary    ############################")
    print(f"Total number of rows (observations): {observations.shape[0]}")
    print(f"Total number of unique (patient, session): {observations.drop_duplicates(subset=["subjetID", "session_name"]).shape[0]}")
    print(f"Total number of unique patients: {observations.drop_duplicates(subset=["subjetID"]).shape[0]}")
    print("")

    print(f"Number of T1W volumes: {observations[observations["T1W"] != "---"].shape[0]}")
    print(f"Number of T1W-GD volumes: {observations[observations["T1W-GD"] != "---"].shape[0]}")
    print(f"Number of T2W volumes: {observations[observations["T2W"] != "---"].shape[0]}")
    print("")

    print(f"Number of patients with T1W, T1W-GD and T2W: {observations[(observations["T1W"] != "---") & (observations["T1W-GD"] != "---") & (observations["T2W"] != "---")].shape[0]}")
    print(f"Number of patients with T1W and T1W-GD: {observations[(observations["T1W"] != "---") & (observations["T1W-GD"] != "---")].shape[0]}")
    print(f"Number of patients with T1W and T2W: {observations[(observations["T1W"] != "---") & (observations["T2W"] != "---")].shape[0]}")
    print(f"Number of patients with T1W-GD and T2W: {observations[(observations["T1W-GD"] != "---") & (observations["T2W"] != "---")].shape[0]}")
    print("")

    print("\nNumber of observations with 2, or 3 out of (T1, T1GD and T2) for each diagnose:\n")
    print(observations.groupby("class_label")[["T1_and_T1GD", "T1GD_and_T2", "T1_and_T2", "all_three"]].sum().reset_index())
    print("\n")

#%%

def get_conf_matrix(all_preds, all_targets, n_classes=3) -> list:
    '''
    Takes two integer lists of all target classes, and all predictions by some classifier.

    Returns a confusion matrix, with true class on rows, and predicted class on the columns,
    as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    '''
    matrix = [[0 for i in range(n_classes)] for i in range(n_classes)]

    print(f"There are {len(all_preds)} predictions in all_preds.")

    for i in range(n_classes):
        for j in range(n_classes):
            for tar, pre in zip(all_targets, all_preds):
                if tar == i and pre == j:
                    matrix[i][j] += 1
    
    return(matrix)



# %%
import re

def create_conf_matrix_fig(conf_matrix, save_fig_as=None, epoch=None, title="", n_classes=3, subtitle=None):
    '''
    Takes confusion matrices (on training and validation data),
    and creates a figure with them. Saves as a png.

    The true classes are on the rows, and the predicted values on the columns.
    '''
    fig, axs = plt.subplots(ncols=1)
    #fig.tight_layout(rect=(0,0,1,0.999))

    # axs[0].matshow(train_mat)
    # for (i, j), z in np.ndenumerate(train_mat):
    #     axs[0].text(j, i, '{}'.format(z), ha='center', va='center')
    # axs[0].set_yticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    # axs[0].set_xticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    # axs[0].set_xlabel("True value")
    # axs[0].set_ylabel("Prediction")
    # axs[0].set_title("Training data")

    if n_classes == 3:
        labels = ["Gli", "Epe", "Med"]
    elif n_classes == 2:
        labels = ["Supra", "Infra"]
    else:
        raise ValueError("Unsupported number of classes")

    axs.matshow(conf_matrix, alpha=0.3, cmap="Blues")
    for (i, j), z in np.ndenumerate(conf_matrix):
        axs.text(j, i, '{}'.format(z), ha='center', va='center', size="xx-large")
    axs.set_yticks(ticks=[i for i in range(n_classes)], labels=labels, fontsize=12)
    axs.set_xticks(ticks=[i for i in range(n_classes)], labels=labels, fontsize=12)
    axs.set_ylabel("True value", fontsize=16)
    axs.set_xlabel("Prediction", fontsize=16)
    #axs.set_title("Validation data")

    if subtitle != None: plt.title(subtitle)

    if save_fig_as != None:
        #start_date = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}", save_fig_as).group(0)
        #fig.suptitle(title+" (Epoch "+str(epoch)+")", fontsize=16)
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.savefig(save_fig_as)
    else:
        if title == "": title = "Confusion matrix"
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.show()


#create_conf_matrix_fig([[3,4, 9], [1,2, 0], [1,2, 0]])

def create_lr_schedule_fig(scheduler, optimizer, max_epochs, save_as):
    '''
    Takes a learning rate scheduler, an optimizer, a maximum number of epochs, and a path.

    Saves a figure that illustrates the learning rate across all epochs.
    '''
    # Make a copy of the scheduler to not change the one that will be used for training
    lr_values = []

    for epoch in range(max_epochs):
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lr_values.append(current_lr)

    # Plot the learning rate over epochs
    plt.plot(range(max_epochs), lr_values, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.savefig(save_as)


# %%


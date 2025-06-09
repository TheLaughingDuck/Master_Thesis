#%%
# SETUP
import os
import sys
import re
import pickle
import datetime
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter


from torchsummary import summary

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Activations,
    LoadImaged,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotated,
    Rand3DElasticd,
    EnsureTyped,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SpatialPadd,
    MapTransform,
    DivisiblePadd,
    Orientationd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    AsDiscrete,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandRotate90d,
    ToTensord,
    Lambdad,
    ScaleIntensityRanged,
)



# %% UTILITIES ################## DATA
def get_train_transform(
    roi_size=[128, 128, 128], scaled_intensity_min=-4.6, scaled_intensity_max=4.6
):
    train_transform = Compose(
        [
            # load  images and stack them together
            LoadImaged(keys="images"),
            LoadImaged(keys="image_not_augmented"),
            EnsureTyped(keys="images"),
            EnsureTyped(keys="image_not_augmented"),
            # add channel dim
            EnsureChannelFirstd(
                keys="images"
            ),  # NOTE: this must be present if not Resized does not work
            EnsureChannelFirstd(
                keys="image_not_augmented"
            ),  # NOTE: this must be present if not Resized does not work
            ScaleIntensityRanged(
                keys="images",
                a_min=scaled_intensity_min,
                a_max=scaled_intensity_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            ScaleIntensityRanged(
                keys="image_not_augmented",
                a_min=scaled_intensity_min,
                a_max=scaled_intensity_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Resized(keys="images", spatial_size=roi_size, mode="area"),
            Resized(keys="image_not_augmented", spatial_size=roi_size, mode="area"),
            RandFlipd(keys="images", prob=0.5, spatial_axis=-1),
            RandFlipd(keys="images", prob=0.5, spatial_axis=-2),
            RandFlipd(keys="images", prob=0.5, spatial_axis=-3),
            
            # transform the label saved in "extra" to one-hot
            # transform the label saved in "extra" to one-hot
            Lambdad(
                keys="label",
                func=lambda x: torch.nn.functional.one_hot(
                    torch.tensor(x), num_classes=config["training"]["num_classes"]
                ).type(torch.float32),
            ),
        ]
    )

    # ###################### Simon version
    # train_transform = Compose(
    #     [
    #         LoadImaged(keys="images"),
    #         NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
    #         Resized(keys="images", spatial_size=(128, 128, 128)),
    #         RandRotate90d(
    #             keys="images", prob=1, max_k=3, spatial_axes=(0, 1)
    #         ),
    #         RandFlipd(keys="images", prob=1, spatial_axis=2),
    #         ToTensord(keys="images", track_meta=False),
    #     ]
    # )

    return train_transform


def get_valid_transform(
    roi_size=[128, 128, 128], scaled_intensity_min=-4.6, scaled_intensity_max=4.6
):
    val_transform = Compose(
        [
            # load  images and stack them together
            LoadImaged(keys="images"),
            EnsureTyped(keys="images"),
            # # add channel dim
            EnsureChannelFirstd(
                keys="images"
            ),  # NOTE: this must be present if not Resized does not work
            ScaleIntensityRanged(
                keys="images",
                a_min=scaled_intensity_min,
                a_max=scaled_intensity_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Resized(keys="images", spatial_size=roi_size, mode="area"),
            # transform the label saved in "extra" to one-hot
            # transform the label saved in "extra" to one-hot
            Lambdad(
                keys="label",
                func=lambda x: torch.nn.functional.one_hot(
                    torch.tensor(x), num_classes=config["training"]["num_classes"]
                ).type(torch.float32),
            ),
        ]
    )
    return val_transform


# %% CONFIGURATIONS (mimics the hydra config file)

config = {
    "work_dir": "/local/data2/simjo484/Training_outputs/3D_ResNet", #"/local/data1/iulta54/Code/Master_thesis/IET_3D_ResNet",
    "data_dir": "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files", #"/local/data1/iulta54/Data/CBTN_RADIOLOGY_V2/Final_preprocessed_files",
    "splits_dir": "/local/data1/simjo484/mt_data/all_data/MRI/simon", #"/local/data1/iulta54/Data/CBTN_RADIOLOGY_V2",
    "dataset": {
        "modalities": ["T2W"], #["T1W", "T1W-GD", "T2W"] # EDIT HERE
        "roi": [128, 128, 128],
    },
    "global_seed": 20251013,
    "training": {
        "num_classes": 3, # EDIT HERE
        "batch_size": 3,
        "learning_rate": 0.00001,#0.00001,
        "num_epoch": 200,
        "num_workers": 18,
        "dropout": 0.5,
        "weight_decay": 0.001,
        "early_stopping_patience": 50,
    },
    "models": {
        "model": "resnet_mixed_conv",  # resnet2p1, resnet_mixed_conv, 3dconv # EDIT HERE
        "stem_type": "None", # NOT USED  # resnet_mixed_convStem, conv3dStem, resnet2p1Stem
        "pretrain": True, # EDIT HERE
    },
}

# %% DATASET - load file paths and labels

# image files
IMG_ROOT_DIR = config["data_dir"]
# check if it exists
if not os.path.exists(IMG_ROOT_DIR):
    raise ValueError("ROOT_DIR does not exist")

# labels
LABEL_ROOT_DIR = config["splits_dir"]
if config["training"]["num_classes"] == 2: label_column = "loc_label" # The name of the label column in the data
elif config["training"]["num_classes"] == 3: label_column = "class_label"

# load the dataframes
if config["training"]["num_classes"] == 2:
    train_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "train_df_loc.csv"))
    valid_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "valid_df_loc.csv"))
elif config["training"]["num_classes"] == 3:
    train_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "train_df.csv"))
    valid_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "valid_df.csv"))

# filter out those that do not have the modalities that we need
for modality in config["dataset"]["modalities"]:
    train_df = train_df[train_df[modality].notnull()]
    valid_df = valid_df[valid_df[modality].notnull()]
    # and different from "---"
    train_df = train_df[train_df[modality] != "---"]
    valid_df = valid_df[valid_df[modality] != "---"]

# check that the files exist
for modality in config["dataset"]["modalities"]:
    for idx, row in train_df.iterrows():
        if not os.path.isfile(os.path.join(IMG_ROOT_DIR, row[modality])):
            raise ValueError(
                f"File {os.path.join(IMG_ROOT_DIR, row[modality])} does not exist"
            )

    for idx, row in valid_df.iterrows():
        if not os.path.isfile(os.path.join(IMG_ROOT_DIR, row[modality])):
            raise ValueError(
                f"File {os.path.join(IMG_ROOT_DIR, row[modality])} does not exist"
            )

# build list of dictionaries with image and integer label
train_dataset = [
    {
        "images": [
            os.path.join(IMG_ROOT_DIR, row[modality])
            for modality in config["dataset"]["modalities"]
        ],
        "image_not_augmented": [
            os.path.join(IMG_ROOT_DIR, row[modality])
            for modality in config["dataset"]["modalities"]
        ],
        "label": row[label_column],
    }
    for idx, row in train_df.iterrows()
]
valid_dataset = [
    {
        "images": [
            os.path.join(IMG_ROOT_DIR, row[modality])
            for modality in config["dataset"]["modalities"]
        ],
        "label": row[label_column],
    }
    for idx, row in valid_df.iterrows()
]
# print the number of samples
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in valid dataset: {len(valid_dataset)}")

# compute class weights based on the training dataset. Larger weights to the less represented classes
# get unique class labels
class_labels = train_df[label_column].unique()
# sort them
class_labels = np.sort(class_labels)
# get the number of subjects per class
class_counts = [len(train_df[train_df[label_column] == i]) for i in class_labels]
# compute the class weights
class_weights = np.array(
    [
        sum(class_counts) / (len(class_counts) * class_counts[i])
        for i in range(len(class_counts))
    ]
)
# normalize the weights
class_weights = class_weights / class_weights.sum()

# print the class weights (label : nbr subjects : class weight)
for i in range(len(class_labels)):
    print(
        f"Label {class_labels[i]}: {class_counts[i]:3d} subjects, weight {class_weights[i]:0.4f}"
    )

# convert to tensor
class_weights = torch.tensor(class_weights).float()

# %% DATASET - create the dataset and dataloaders

# set the seed
torch.manual_seed(config["global_seed"])
# create the dataset
train_dataset = Dataset(
    data=train_dataset,
    transform=get_train_transform(roi_size=config["dataset"]["roi"]),
)
valid_dataset = Dataset(
    data=valid_dataset,
    transform=get_valid_transform(roi_size=config["dataset"]["roi"]),
)
# create the dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["training"]["num_workers"],
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["training"]["num_workers"],
)

# %% CHECK DATALOADER

from monai.utils import set_determinism

# get next batch
it = iter(train_dataloader)
batch = next(it)
batch = next(it)
batch = next(it)


# get image and label
image = batch["images"]
image_not_augmented = batch["image_not_augmented"]
label = batch["label"]

# %%
# SIMON
for i in range(2):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image[i, 0, :, :, 70].numpy().squeeze(), cmap="gray")
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image_not_augmented[i, 0, :, :, 70].numpy().squeeze(), cmap="gray")
    plt.show()

# Difference
#img = image[i, 0, :, :, 70].numpy().squeeze()
img = np.array([[i for i in range(128)] for i in range(128)])
img = image[0, 0, :, :, 70].numpy().squeeze() - image_not_augmented[0, 0, :, :, 70].numpy().squeeze()
plt.imshow(img, cmap="gray")
plt.suptitle("WEIRD")
plt.show()
#%%

# check shapes
print(image.shape)
print(label)

# plot image
fig, ax = plt.subplots(1, len(config["dataset"]["modalities"]), figsize=(20, 20))
if len(config["dataset"]["modalities"]) == 1:
    ax = [ax]
else:
    ax = ax.flatten()
# plot each modality
for midx, m in enumerate(config["dataset"]["modalities"]):
    ax[midx].imshow(image[0, midx, :, :, 70].numpy().squeeze(), cmap="gray")
    ax[midx].set_title(m)
plt.show()

# # plot histogram
# fig, ax = plt.subplots(1, len(config["dataset"]["modalities"]), figsize=(20, 20))
# if len(config["dataset"]["modalities"]) == 1:
#     ax = [ax]
# else:
#     ax = ax.flatten()
# # plot each modality
# for midx, m in enumerate(config["dataset"]["modalities"]):
#     ax[midx].hist(image[0, midx, :, :, :].numpy().squeeze().flatten(), bins=100)
#     ax[midx].set_title(m)
#     ax[midx].set_yscale("log")

# plt.show()
# %%

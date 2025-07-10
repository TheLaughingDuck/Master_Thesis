'''
Module for creating a data loader.
'''

import pickle

import numpy as np
import torch


from monai import data, transforms
from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

#from utils import *
import pickle


from monai.transforms import (
    LoadImaged,
    EnsureChannelFirst,
    ScaleIntensity,
    NormalizeIntensityd,
    Resized,
    ToTensord,
    Compose,
    Rotate90d,
    Lambda,
    ToDeviced,
    RandRotate90d,
    RandFlipd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureChannelFirstd,
    Lambdad
)

def get_loader(config, sequences):
    print("\nTHIS IS THE TENTORIAL Data Loader\n")

    train_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/train_df_loc.csv")
    valid_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/valid_df_loc.csv")
    
    # Debug mode: Train on very few examples in order to achieve massive speedup, allowing debugging.
    if config["debug_mode"] == True:
        print("\nDebug mode!\n")
        train_df = train_df[0:10]
        valid_df = valid_df[0:10]
    
    # # Get loss weights (proportion of each class in training data)
    # n_diags = len(set(train_df["class_label"]))
    # class_counts = collections.Counter(train_df["class_label"]) # A dict with the class counts
    # loss_weights = torch.tensor([1/class_counts[i] for i in range(n_diags)])

    # Does almost the same thing as the commented out code above
    from sklearn.utils import class_weight
    labels = torch.tensor(train_df["loc_label"].tolist()).long()
    class_weights=class_weight.compute_class_weight('balanced',classes=np.unique(labels),y=labels.numpy())
    loss_weights=torch.tensor(class_weights,dtype=torch.float).to("cuda")
    print(f"\nThe loss weights are: {loss_weights}\n")

    # Format data paths
    img_root = "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files"
    train_data_paths = format_paths(train_df, sequences=sequences, root=img_root)
    valid_data_paths = format_paths(valid_df, sequences=sequences, root=img_root)

    # print(train_data_paths)
    # print(valid_data_paths)


    # Define train transform
    train_transform = Compose(
        [
            # load  image and stack them together
            LoadImaged(keys="image"),
            EnsureTyped(keys="image"),
            # add channel dim
            EnsureChannelFirstd(
                keys="image"
            ),  # NOTE: this must be present if not Resized does not work
            ScaleIntensityRanged(
                keys="image",
                a_min=-4.6,
                a_max=4.6,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Resized(keys="image", spatial_size=(128,128,128), mode="area"),
            RandFlipd(keys="image", prob=0.5, spatial_axis=-1),
            RandFlipd(keys="image", prob=0.5, spatial_axis=-2),
            RandFlipd(keys="image", prob=0.5, spatial_axis=-3),
            # transform the label saved in "extra" to one-hot
            # transform the label saved in "extra" to one-hot
            # Lambdad(
            #     keys="label",
            #     func=lambda x: torch.nn.functional.one_hot(
            #         torch.tensor(x), num_classes=2
            #     ).type(torch.float32),
            # ),
        ]
    )


    # Define valid transform
    valid_transform = Compose(
        [
            # load  image and stack them together
            LoadImaged(keys="image"),
            EnsureTyped(keys="image"),
            # # add channel dim
            EnsureChannelFirstd(
                keys="image"
            ),  # NOTE: this must be present if not Resized does not work
            ScaleIntensityRanged(
                keys="image",
                a_min=-4.6,
                a_max=4.6,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Resized(keys="image", spatial_size=(128,128,128), mode="area"),
            # transform the label saved in "extra" to one-hot
            # transform the label saved in "extra" to one-hot
            # Lambdad(
            #     keys="label",
            #     func=lambda x: torch.nn.functional.one_hot(
            #         torch.tensor(x), num_classes=2
            #     ).type(torch.float32),
            # ),
        ]
    )

    train_ds = data.Dataset(data=train_data_paths, transform=train_transform)
    train_dataloader = data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"], pin_memory=True, prefetch_factor=4
    )

    valid_ds = data.Dataset(data=valid_data_paths, transform=valid_transform)
    valid_dataloader = data.DataLoader(
        valid_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], pin_memory=True, prefetch_factor=4
    )

    return([train_dataloader, valid_dataloader], loss_weights)


def format_paths(df, sequences, root, device="cpu"):
    '''
    Takes a pd dataframe featuring three columns of T1W, T1W-GD, and T2W image paths and a class label column,
    and formats the remaining rows in the following manner:
    
    [
    {"image": ["path1", "path2", "path3", "path4"], "label":tensor(0)},
    ...
    {"image": ["path1", "path2", "path3", "path4"], "label":tensor(0)}
    ]

    === Arguments ===
    df: A pandas dataframe with at least columns T1W, T2W, T1W-GD, class_label.

    sequences: A list specifying the desired combination of sequences, for example
        ["T2W", "T2W", "T2W", "T2W"] for a T2W only classifier.
        ["T1W-GD", "T1W-GD", "T2W", "T2W"] for a T1W-GD and T2W classifier.
    '''

    paths = []
    for id, row in df.iterrows():
        observation = {"image":[root+"/"+row[i] for i in sequences],
                       "label":torch.tensor(row["loc_label"]).to(device).long()}
        paths.append(observation)
    
    return(paths)
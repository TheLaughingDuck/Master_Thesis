'''
Helper functions for handling data in a training pipeline
'''

import json
import math
import os

import pickle

import numpy as np
import torch


from monai import data, transforms


from torch import nn
import torch
from torch.nn import Threshold

import numpy as np
import nibabel as nib

from utils import *
import pickle

import matplotlib.pyplot as plt

import collections

from monai.networks.nets import SwinUNETR
from monai import data
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
    RandFlipd
)

def get_loader(args):
    
    # save_dir = "/home/simjo484/master_thesis/Master_Thesis/classifier_training/" 
    # with open(save_dir+"t2_training_paths.pkl", "rb") as f:
    #     train_data_paths = pickle.load(f)

    # with open(save_dir+"t2_valid_paths.pkl", "rb") as f:
    #     valid_data_paths = pickle.load(f)

    # # Debug mode: Train on very few examples in order to achieve massive speedup, allowing debugging.
    # if args.debug_mode == True:
    #     print("\nDebug mode!\n")
    #     train_data_paths = train_data_paths[0:10]
    #     valid_data_paths = valid_data_paths[0:10]
    

    meta_root = "/local/data1/simjo484/mt_data/all_data/MRI/simon/"
    with open(meta_root+"train_df.pkl", "rb") as f:
        train_df = pickle.load(f)

    with open(meta_root+"valid_df.pkl", "rb") as f:
        valid_df = pickle.load(f)

    # Debug mode: Train on very few examples in order to achieve massive speedup, allowing debugging.
    if args.debug_mode == True:
        print("\nDebug mode!\n")
        train_df = train_df[0:20]
        valid_df = valid_df[0:20]
    
    # Get loss weights (proportion of each class in training data)
    n_diags = len(set(train_df["class_label"]))
    class_counts = collections.Counter(train_df["class_label"]) # A dict with the class counts
    loss_weights = torch.tensor([1/class_counts[i] for i in range(n_diags)])

    # Format data paths
    img_root = "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files"
    train_data_paths = format_paths(train_df, sequences=["T2W", "T2W", "T2W","T2W"], root=img_root)
    valid_data_paths = format_paths(valid_df, sequences=["T2W", "T2W", "T2W","T2W"], root=img_root)


    # Define train transform
    train_transform = Compose([
        LoadImaged(keys="images"),
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        Resized(keys="images", spatial_size=(128, 128, 128)),

        RandRotate90d(keys="images", prob=0.5, max_k=3, spatial_axes=(0,1)),
        RandFlipd(keys="images", prob=0.5, spatial_axis=2),

        ToTensord(keys="images", track_meta=False)
    ])


    # Define valid transform
    valid_transform = Compose([
        LoadImaged(keys="images"),
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        Resized(keys="images", spatial_size=(128, 128, 128)),
        ToTensord(keys="images", track_meta=False),
    ])


    if args.test_mode:
        raise NotImplementedError("A test mode transformer has not been implemented yet!")
    else:
        train_ds = data.Dataset(data=train_data_paths, transform=train_transform)
        train_dataloader = data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, prefetch_factor=4
        )

        valid_ds = data.Dataset(data=valid_data_paths, transform=valid_transform)
        valid_dataloader = data.DataLoader(
            valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, prefetch_factor=4
        )

    loss_weights = None
    return([train_dataloader, valid_dataloader], loss_weights)


def format_paths(df, sequences, root, device="cpu"):
    '''
    Takes a pd dataframe featuring three columns of T1W, T1W-GD, and T2W image paths and a class label column,
    and formats the remaining rows in the following manner:
    
    [
    {"images": ["path1", "path2", "path3", "path4"], "label":tensor(0)},
    ...
    {"images": ["path1", "path2", "path3", "path4"], "label":tensor(0)}
    ]

    === Arguments ===
    df: A pandas dataframe with at least columns T1W, T2W, T1W-GD, class_label.

    sequences: A list specifying the desired combination of sequences, for example
        ["T2W", "T2W", "T2W", "T2W"] for a T2W only classifier.
        ["T1W-GD", "T1W-GD", "T2W", "T2W"] for a T1W-GD and T2W classifier.
    '''
    paths = []
    for id, row in df.iterrows():
        observation = {"images":[root+"/"+row[i] for i in sequences],
                       "label":torch.tensor(row["class_label"]).to(device).long()}
        paths.append(observation)
    
    return(paths)
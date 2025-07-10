'''
Helper functions for handling data loading in a training pipeline.
'''

import json
import math
import os

import pickle

import pandas as pd
import numpy as np
import torch


from monai import data, transforms


from torch import nn
import torch
from torch.nn import Threshold

import numpy as np
import random
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

def get_loader(args, seq_types = "T2W", label_column="class_label", dataset_paths=None, seed=None):
    '''
    
    ================ Arguments ================
    seed: If doing evaluation, that should be reproducible, set the seed to some integer value, otherwise leave as None.
    '''
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


    # Set seed if None
    if seed == None: seed = random.randint(1, 10000000)

    # Process and load the path data sets
    if dataset_paths != None:
        train_df = pd.read_csv(dataset_paths[0])
        valid_df = pd.read_csv(dataset_paths[1])
        test_df = pd.read_csv(dataset_paths[2])
    else:
        if label_column == "class_label":
            meta_root = "/local/data1/simjo484/mt_data/all_data/MRI/simon/"
            with open(meta_root+"train_df.pkl", "rb") as f:
                train_df = pickle.load(f)

            with open(meta_root+"valid_df.pkl", "rb") as f:
                valid_df = pickle.load(f)
            
            with open(meta_root+"test_df.pkl", "rb") as f:
                test_df = pickle.load(f)

        elif label_column == "loc_label":
            train_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/train_df_loc.csv")
            valid_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/valid_df_loc.csv")
            test_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/test_df_loc.csv")
        else:
            raise ValueError("Unsupported \"label_column\" argument.")
    
    
    # Debug mode: Train on very few examples in order to achieve massive speedup, allowing debugging.
    if args.debug_mode == True:
        print("\nDebug mode!\n")
        train_df = train_df[0:10]
        valid_df = valid_df[0:10]
        test_df = test_df[0:10]
    
    # # Get loss weights (proportion of each class in training data)
    # n_diags = len(set(train_df["class_label"]))
    # class_counts = collections.Counter(train_df["class_label"]) # A dict with the class counts
    # loss_weights = torch.tensor([1/class_counts[i] for i in range(n_diags)])

    # Does almost the same thing as the commented out code above
    from sklearn.utils import class_weight
    labels = torch.tensor(train_df["class_label"].tolist()).long()
    class_weights=class_weight.compute_class_weight('balanced',classes=np.unique(labels),y=labels.numpy())
    loss_weights=torch.tensor(class_weights,dtype=torch.float).to(args.cl_device)
    print(f"\nThe loss weights are: {loss_weights}\n")

    # Format data paths
    img_root = "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files"

    if seq_types == "T2W": seq_types = ["T2W", "T2W", "T2W", "T2W"]
    elif seq_types == "T1W-GD": seq_types = ["T1W-GD", "T1W-GD", "T1W-GD", "T1W-GD"]
    elif seq_types == "T1W-GD_T2W": seq_types = ["T1W-GD", "T1W-GD", "T2W", "T2W"]
    else:
        raise ValueError("Incorrect sequence type specification.")

    train_data_paths = format_paths(train_df, sequences=seq_types, root=img_root, label_column=label_column)
    valid_data_paths = format_paths(valid_df, sequences=seq_types, root=img_root, label_column=label_column)
    test_data_paths = format_paths(test_df, sequences=seq_types, root=img_root, label_column=label_column)

    # Define train transform
    train_transform = Compose([
        LoadImaged(keys="images"),
        
        #Resized(keys="images", spatial_size=(128, 128, 128)), # This changes the size through scaling

        # transforms.CropForegroundd(
        #     keys="images", source_key="images", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
        # ),
        transforms.SpatialCropd(
            keys="images", roi_center=[120,120,52], roi_size=[128,128,128]
        ),
        # transforms.RandSpatialCropd(
        #     keys="images", roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False, random_center=False
        # ),
        
        # New augmentations
        transforms.RandFlipd(keys="images", prob=0.5, spatial_axis=0).set_random_state(seed),
        transforms.RandFlipd(keys="images", prob=0.5, spatial_axis=1).set_random_state(seed),
        transforms.RandFlipd(keys="images", prob=0.5, spatial_axis=2).set_random_state(seed),
        
        #RandRotate90d(keys="images", prob=args.data_aug_prob, max_k=3, spatial_axes=(0,1)),
        #RandFlipd(keys="images", prob=args.data_aug_prob, spatial_axis=2),

        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),

        # Intensity augmentations
        transforms.NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="images", factors=0.1, prob=1.0).set_random_state(seed),
        transforms.RandShiftIntensityd(keys="images", offsets=0.1, prob=1.0).set_random_state(seed),

        ToTensord(keys="images", track_meta=False)
    ])


    # Define valid transform
    valid_transform = Compose([
        LoadImaged(keys="images"),

        # New transformations
        # transforms.CropForegroundd(
        #     keys="images", source_key="images", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
        # ),
        # transforms.RandSpatialCropd(
        #     keys="images", roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
        # ),
        
        # New transformation
        transforms.SpatialCropd(
            keys="images", roi_center=[120,120,52], roi_size=[128,128,128]
        ),

        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        ToTensord(keys="images", track_meta=False),
    ])

    test_transform = Compose([
        LoadImaged(keys="images"),

        # New transformations
        # transforms.CropForegroundd(
        #     keys="images", source_key="images", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
        # ),
        # transforms.RandSpatialCropd(
        #     keys="images", roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
        # ),
        
        # New transformation
        transforms.SpatialCropd(
            keys="images", roi_center=[120,120,52], roi_size=[128,128,128]
        ),

        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
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

        test_ds = data.Dataset(data=test_data_paths, transform=test_transform)
        test_dataloader = data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, prefetch_factor=4
        )

    #loss_weights = None
    return([train_dataloader, valid_dataloader, test_dataloader], loss_weights)


def format_paths(df, sequences, root, device="cpu", label_column="class_label"):
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
    
    label_column: The name of the column in the data where the labels are.
        "class_label" refers to the diagnose label (0,1,2), while "loc_label" refers to the location label (0,1) which is only present in the specific train_df_loc.csv etc files.
    '''
    paths = []
    for id, row in df.iterrows():
        observation = {"images":[root+"/"+row[i] for i in sequences],
                       "label":torch.tensor(row[label_column]).to(device).long()}
        paths.append(observation)
    
    return(paths)
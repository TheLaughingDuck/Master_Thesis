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
    ToDeviced
)

def get_loader(args):
    
    save_dir = "/local/data2/simjo484/Classifier_training/"
    with open(save_dir+"t2_training_paths.pkl", "rb") as f:
        train_data_paths = pickle.load(f)

    with open(save_dir+"t2_valid_paths.pkl", "rb") as f:
        valid_data_paths = pickle.load(f)

    # Debug mode: Train on very few examples in order to achieve massive speedup, allowing debugging.
    if args.debug_mode == True:
        print("\nDebug mode!\n")
        train_data_paths = train_data_paths[0:10]
        valid_data_paths = valid_data_paths[0:10]


    # Define train transform
    train_transform = Compose([
        LoadImaged(keys="images"),
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        Resized(keys="images", spatial_size=(128, 128, 128)),
        ToTensord(keys="images", track_meta=False),
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

    return([train_dataloader, valid_dataloader])
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


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    
    save_dir = "/local/data2/simjo484/Classifier_training/"
    with open(save_dir+"t2_training_paths.pkl", "rb") as f:
        train_data_paths = pickle.load(f)

    with open(save_dir+"t2_valid_paths.pkl", "rb") as f:
        valid_data_paths = pickle.load(f)

    # Debug mode: Train on very few examples in order to achieve massive speedup, allowing debugging.
    if args.debug_mode == True:
        print("\nDebug mode!\n")
        train_data_paths = train_data_paths[0:4]
        valid_data_paths = valid_data_paths[0:4]


    # Define train transform
    train_transform = Compose([
        LoadImaged(keys="images"),
        #EnsureChannelFirst(), # Did not cause problem or worse segmentation when removed
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        Resized(keys="images", spatial_size=(128, 128, 128)),
        #EnsureType(),
        #Rotate90d(keys="images", k=3, spatial_axes=(0,1)), # *Should* not be necessary
        #EnsureType(),
        ToTensord(keys="images", track_meta=False),

        #Lambda(lambda x: [{"images": BSF_embedder(x["images"].view(1,4,128,128,128)), "label": x["label"]}])
        #Lambda(lambda x: print("Before ToDeviced:", type(x["images"])) or x),
        #ToDeviced(keys="images", device=device), # For some reason this causes a RuntimeError
        #Lambda(lambda x: print("After ToDeviced:", type(x["images"])) or x)
    ])


    # Define valid transform
    valid_transform = Compose([
        LoadImaged(keys="images"),
        #EnsureChannelFirst(), # Did not cause problem or worse segmentation when removed
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        Resized(keys="images", spatial_size=(128, 128, 128)),
        #EnsureType(),
        #Rotate90d(keys="images", k=3, spatial_axes=(0,1)), # *Should* not be necessary
        #EnsureType(),
        ToTensord(keys="images", track_meta=False),

        #Lambda(lambda x: [{"images": BSF_embedder(x["images"].view(1,4,128,128,128)), "label": x["label"]}])
        #Lambda(lambda x: print("Before ToDeviced:", type(x["images"])) or x),
        #ToDeviced(keys="images", device=device), # For some reason this causes a RuntimeError
        #Lambda(lambda x: print("After ToDeviced:", type(x["images"])) or x)
    ])


    if args.test_mode:
        raise NotImplementedError("A test mode transformer has not been implemented yet!")
    else:
        train_ds = data.Dataset(data=train_data_paths, transform=train_transform)
        train_dataloader = data.DataLoader(
            train_ds, batch_size=args.n_batches, shuffle=False, num_workers=args.workers, pin_memory=True
        )

        valid_ds = data.Dataset(data=valid_data_paths, transform=valid_transform)
        valid_dataloader = data.DataLoader(
            valid_ds, batch_size=args.n_batches, shuffle=False, num_workers=args.workers, pin_memory=True
        )

    return([train_dataloader, valid_dataloader])




# def get_loader(args):
#     data_dir = args.data_dir
#     datalist_json = args.json_list
#     train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
#     train_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.CropForegroundd(
#                 keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
#             ),
#             transforms.RandSpatialCropd(
#                 keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
#             ),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#             transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#             transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )
#     val_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )

#     test_transform = transforms.Compose(
#         [
#             transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#             transforms.ToTensord(keys=["image", "label"]),
#         ]
#     )

#     if args.test_mode:
#         val_ds = data.Dataset(data=validation_files, transform=test_transform)
#         val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
#         test_loader = data.DataLoader(
#             val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
#         )

#         loader = test_loader
#     else:
#         train_ds = data.Dataset(data=train_files, transform=train_transform)

#         train_sampler = Sampler(train_ds) if args.distributed else None
#         train_loader = data.DataLoader(
#             train_ds,
#             batch_size=args.batch_size,
#             shuffle=(train_sampler is None),
#             num_workers=args.workers,
#             sampler=train_sampler,
#             pin_memory=True,
#         )
#         val_ds = data.Dataset(data=validation_files, transform=val_transform)
#         val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
#         val_loader = data.DataLoader(
#             val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
#         )
#         loader = [train_loader, val_loader]

#     return loader


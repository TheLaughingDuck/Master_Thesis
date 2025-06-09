'''
I ran some training in /local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture
The validation confusion matrices reported there are from the latest saved model, and since the models were all overfitting, those
confusion matrices are not really interesting. Instead, we should load the validation data, and run it through the best model "model.pt"
in those directories, and generate a validation matrix from that.

This script does that.
'''

#%%
# SETUP

# load data
# load models
# run the data through the models
# create conf matrices

import torch
import os

os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import EmbedSwinUNETR, get_loader

os.chdir("/home/simjo484/master_thesis/Master_Thesis/BSF_finetuning")
#from bsf_data_utils import get_loader


import matplotlib.pyplot as plt
import argparse

from itertools import islice

import numpy as np


from torch import nn
import numpy as np
from monai.networks.nets import SwinUNETR
import torch


import torch
from torch.nn import Threshold

import numpy as np
import nibabel as nib

from utils import *

import matplotlib.pyplot as plt

from monai.networks.nets import SwinUNETR
#from monai import data
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    NormalizeIntensity,
    Resize,
    ToTensor,
    Compose,
    Rotate90,
    Lambda
)


# Arguments
class Args(argparse.Namespace):
    logdir = ""
    optim_lr = 1e-4
    reg_weight = 1e-5
    roi_x = 128
    roi_y = 128
    roi_z = 128
    distributed = False
    workers = 18
    data_dir='/local/data2/simjo484/BRATScommon/BRATS21/'
    json_list = "./jsons/brats21_folds.json"
    fold = 4
    test_mode = False
    batch_size = 3
    debug_mode = False
    device = "cuda"
    cl_device = "cuda"
    pp_device = "cpu"
    data_aug_prob = 0.3
    freeze_blocks = [3,4]

args = Args()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(768, 10),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(10, 3)#,
            #nn.ReLU(),

            # nn.Dropout(p=0.4),
            # nn.Linear(300, 3)#,

            #nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

# Define Feature Extractinator
class EmbedSwinUNETR(SwinUNETR):
    '''
    Class that represents the 4-channel BSF architecture intended for BraTS.
    '''

    def __init__(self, **args):
        super(EmbedSwinUNETR, self).__init__(
            img_size=(128, 128, 128), # Essentially trivial
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
            **args)
        
        #self.load_state_dict(torch.load(weights, map_location=device))

    def forward(self, x_in):

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
    
        out = self.swinViT(x_in, self.normalize)[4]

        # Perform global average pooling to reduce shape from (3,768,4,4,4) to (3,768)
        # (If the batch size is 3).
        # batch_size = x_in.shape[0]
        # out = torch.nn.AvgPool3d((4,4,4))(out).view(batch_size, 768)
        return out
    

class Combined_model(torch.nn.Module):
    def __init__(self, feature_extractor_weights="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-05-08:07:48/model_final.pt",
                 device="cuda"):
        super(Combined_model, self).__init__()

        # SETUP Feature Extractor
        self.feature_extractor = EmbedSwinUNETR()
        #weights_default_val = 
        if feature_extractor_weights != None:
            self.feature_extractor.load_state_dict(torch.load(feature_extractor_weights, map_location=device)["state_dict"])
        
        self.classifier = Classifier()

        # Settings
        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = self.feature_extractor(x)

        # Perform global average pooling to reduce shape from (3,768,4,4,4) to (3,768)
        # (If the batch size is 3).
        batch_size = x.shape[0]
        x = torch.nn.AvgPool3d((4,4,4))(x).view(batch_size, 768)

        x = self.classifier(x)
        return x
    
    def forward_test(self, batch_size=3):
        '''A method for testing the model on a random input.'''
        x_in = torch.randn(batch_size,4,128,128,128).to(self.device)
        
        return(self.forward(x_in))
    
    def freeze(self, args):
        # First freeze all parameters, so that old parameters that are just hanging on get frozen.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        print("\n\n==============================================")
        print("Froze ALL parameters in Feature Extractor.")

        # Cycle through all the 4 modules of the SwinViT
        module_count = 0
        for ch in self.feature_extractor.swinViT.children():
            if module_count in args.freeze_blocks: #e.g. [3,4]
                print(f"Unfreezing SwinViT module {module_count}")
                for param in ch.parameters():
                    param.requires_grad = True
            
            module_count += 1
    
        n_params = lambda x: format(sum(p.numel() for p in x.parameters() if p.requires_grad), ",").replace(",", ".") # Format like 10.000.000
        print(f"\nFeature extractor parameters: {n_params(self.feature_extractor)}")
        print(f"Classifier parameters: {n_params(self.classifier)}")
        print(f"Total parameters count: {n_params(self)} \N{Abacus} \N{Flexed Biceps}")
        print("==============================================\n\n")

    # def freeze(self, blocks:int=0): # See this helpful link: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    #     # First freeze all parameters, so that old parameters that are just hanging on get frozen.
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = False
    #     print("\n\n==============================================")
    #     print("Froze ALL parameters in Feature Extractor.")

    #     # Then unfreeze the specific blocks we want
    #     child_counter = 0
    #     for child in self.feature_extractor.children():
    #         #print(f"################### Child {child_counter} is {child}")

    #         if child_counter == 0: # This represents the 
    #             block_counter = 0
    #             for block in child.children():
    #                 if not block_counter < blocks:
    #                     print(f"Unfreezing block {block_counter} of Child {child_counter} in Feature Extractor.")
    #                     for param in block.parameters():
    #                         param.requires_grad = True

    #                 block_counter += 1
    #         child_counter += 1

    #     n_params = lambda x: format(sum(p.numel() for p in x.parameters() if p.requires_grad), ",").replace(",", ".") # Format like 10.000.000
    #     print(f"\nFeature extractor parameters: {n_params(self.feature_extractor)}")
    #     print(f"Classifier parameters: {n_params(self.classifier)}")
    #     print(f"Total parameters count: {n_params(self)} \N{Abacus} \N{Flexed Biceps}")
    #     print("==============================================\n\n")
        


#%%
# LOAD MODEL
sequences = "t2" # "t1gd" "t2" "t1gd_and_t2"
label_column = "class_label"
on_val_data = True

train_df_path = "/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df."
valid_df_path = "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/valid_df.csv"
data_paths = None #[train_df_path, valid_df_path]

if sequences == "t2":
    # T2W Loader
    loader, loss = get_loader(args, seed = 82734,
                              label_column = label_column, seq_types="T2W", dataset_paths=data_paths)

    # T2W Model
    model = Combined_model()
    model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-16:57:17 (T2 standardized training)/model.pt", map_location="cuda")["state_dict"])
    model.eval()
elif sequences == "t1gd":
    # T1W-GD Loader
    loader, loss = get_loader(args, seed = 82734,
                              label_column = label_column, seq_types="T1W-GD", dataset_paths=data_paths)

    # T1W-GD Model
    model = Combined_model()
    model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-20:03:43 (T1GD standardized training)/model.pt", map_location="cuda")["state_dict"])
    model.eval()
elif sequences == "t1gd_and_t2":
    # T1W-GD and T2W Loader
    loader, loss = get_loader(args, seed = 82734,
                              label_column = label_column, seq_types="T1W-GD_T2W", dataset_paths=data_paths)

    # T1W-GD and T2W Model
    model = Combined_model()
    model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-23:09:52 (T1GD and T2 standardized training)/model.pt", map_location="cuda")["state_dict"])
    model.eval()
elif sequences == "test":
    print("TEST TEST TEST")
    model = Combined_model()
    model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-16:57:17 (T2 standardized training)/model.pt", map_location="cuda")["state_dict"])



# CREATE DATA MATRIX
all_preds = []
all_targets = []

# Cycle over validation matrices
for id, batch_data in enumerate(loader[on_val_data]):
    data, target = batch_data["images"].to(args.device), batch_data["label"].to(args.device)
    #print(f"DATA SHAPE: {data.shape}")

    #data = data[0]
    #print(f"DATA SHAPE: {data.shape}")

    pred = model(data)

    all_preds += pred.argmax(1).tolist()
    all_targets += target.tolist()

matrix = np.array(get_conf_matrix(all_preds=all_preds, all_targets=all_targets))
name = "bsf_classifier_"+sequences+"_" + ("val" if on_val_data else "train")  + "_conf_matrix.png"

subtitle = f"(Accuracy: {round(sum(np.diag(matrix)) / sum(matrix.flatten()), 2)})"
title="BSF on T2W" # Change manually for each model
print(sequences)
create_conf_matrix_fig(matrix, save_fig_as="/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name, epoch=123, title=title, subtitle=subtitle)
# %%

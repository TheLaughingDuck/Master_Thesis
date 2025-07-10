'''
I want to have a script that generates the features for the BSF models on the training data.


'''

#%%
# SETUP PCA Plots
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
import pandas as pd

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
    batch_size = 1
    debug_mode = False
    device = "cuda"
    cl_device = "cuda"
    pp_device = "cpu"
    data_aug_prob = 0.3

args = Args()


def load_model(sequences, label_column):
    # LOAD MODEL
    #sequences = "t2" #"t1gd_and_t2"

    if sequences == "t2":
        # T2W Loader
        loader, loss = get_loader(args, seed = 82734,
                                label_column = label_column, seq_types="T2W", dataset_paths=["/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/train_df_extra_meta.csv", "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/valid_df_extra_meta.csv", "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/test_df_extra_meta.csv"])

        # T2W Model
        model = EmbedSwinUNETR()
        model.to("cuda")
        model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-27-13:20:53 (t2)/model_final.pt", map_location="cuda")["state_dict"])
        model.eval()
    elif sequences == "t1gd":
        # T1W-GD Loader
        loader, loss = get_loader(args, seed = 82734,
                                label_column = label_column, seq_types="T1W-GD", dataset_paths=["/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/train_df_extra_meta.csv", "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/valid_df_extra_meta.csv", "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/test_df_extra_meta.csv"])

        # T1W-GD Model
        model = EmbedSwinUNETR()
        model.to("cuda")
        model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd/2025-04-07-10:40:33/model_final.pt", map_location="cuda")["state_dict"])
        model.eval()
    elif sequences == "t1gd_and_t2":
        # T1W-GD and T2W Loader
        loader, loss = get_loader(args, seed = 82734,
                                label_column = label_column, seq_types="T1W-GD_T2W", dataset_paths=["/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/train_df_extra_meta.csv", "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/valid_df_extra_meta.csv", "/home/simjo484/master_thesis/Master_Thesis/visualization/miscellaneous/data/test_df_extra_meta.csv"])

        # T1W-GD and T2W Model
        model = EmbedSwinUNETR()
        model.to("cuda")
        model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt", map_location="cuda")["state_dict"])
        model.eval()
    
    return model, loader

#%%


# CREATE DATA MATRIX
features = []
diag_labels = []

sequences = "t1gd_and_t2" #"t1gd_and_t2"

model, diag_loader = load_model(sequences, "class_label")

for id, batch_data in enumerate(diag_loader[0]):
    data, target = batch_data["images"].to(args.device), batch_data["label"].to(args.device)
    #print(f"DATA SHAPE: {data.shape}")

    #data = data[0]
    #print(f"DATA SHAPE: {data.shape}")

    x = model(data)

    batch_size = args.batch_size
    #x = torch.flatten(x)
    x = torch.nn.AvgPool3d((4,4,4))(x).view(batch_size, 768)[0] # The [0] is to remove the 1 in the shape from batch size 1.

    features.append(x.detach().to("cpu"))
    diag_labels.append(target.to("cpu"))



#%%
# Load loc labels
del diag_loader
model, loc_loader = load_model(sequences, "loc_label")
loc_labels = []
for id, batch_data in enumerate(loc_loader[0]):
    data, target = batch_data["images"].to(args.device), batch_data["label"].to(args.device)

    loc_labels.append(target.to("cpu"))

# Format labels
diag_labels_df = pd.DataFrame(np.array(diag_labels))
diag_labels_df.rename(columns={0: "diag_label"}, inplace=True)
loc_labels_df = pd.DataFrame(np.array(loc_labels))
loc_labels_df.rename(columns={0: "loc_label"}, inplace=True)

# Format features
features_df = pd.DataFrame(np.array(features))
print(f"FEATURES HAVE DIMS: {features_df.shape}")
print(f"LABELS HAVE DIMS: {diag_labels_df.shape}")
print(f"LABELS HAVE DIMS: {loc_labels_df.shape}")

# Combine features and labels
features_df = pd.concat([features_df, diag_labels_df, loc_labels_df], axis=1)


#%%
# Save features and labels
sequence_specific_name = "bsf_features_"+sequences

features_df.to_csv("/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/Results_chapter/BSF_generated_features/"+sequence_specific_name+".csv")

# %%
# A CHECK
print("A CHECK!")
sum([i!=j for i,j in zip(features_df.iloc[:, 768].tolist(), features_df.iloc[:, 769].tolist())])
# %%

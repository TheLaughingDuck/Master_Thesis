#%%
from datetime import datetime
import os
import torch
from model import Classifier

# model = Classifier()

# #%%
# ts = torch.randn(3,1,240,140,155).to(torch.float).to("cuda")
# model(ts).shape



# # %%
# model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
# model


# # %%




#### PARSER ####
import argparse
parser = argparse.ArgumentParser(description="Training Arguments")
parser.add_argument("--reg_weight", type=float, default=0.0001)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--val_every", type=int, default=5)
parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--logdir", type=str, default="/local/data2/simjo484/Training_outputs/classifier_training/from_scratch/", help="The super directory where the dir for this training should be saved.") #Don't change it, or change it very carefully
parser.add_argument("--comment", type=str)
parser.add_argument("--debug_mode", action="store_true")
parser.add_argument("--workers", type=int, default=18, help="Number of CPU workers. Mainly for data loading.")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--T2W", action="store_true") # add options for other modalities later
args = parser.parse_args()


from data_loader import get_loader
loaders, loss_weights = get_loader(args)


for batch_id, batch_data in enumerate(loaders[0]):
    data, target = batch_data["image"].to(args.device), batch_data["label"].to(args.device)

    print(data.shape)



#%%



#%%
# SETUP
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

# Model
model = EmbedSwinUNETR()
model.to("cuda")
model.load_state_dict(torch.load("/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt", map_location="cuda")["state_dict"])
# %%

# %%

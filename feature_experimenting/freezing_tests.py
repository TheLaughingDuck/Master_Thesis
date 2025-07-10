#%%
# SETUP

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
#from trainer import run_training
#from utils.data_utils import get_loader

from matplotlib import pyplot as plt
import re

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

from utils.parse_arguments import custom_parser


#%%

model = Combined_model()
model.freeze()


# %%

for param in model.feature_extractor.parameters():
    param.requires_grad = False

module_count = 0
for ch in model.feature_extractor.swinViT.children():
    if module_count in [3]:
        print(f"Unfreezing module {module_count}")
        for param in ch.parameters():
            param.requires_grad = True
    
    module_count += 1
    # print(f"CHILD {ch._get_name()}")
    # print(ch.)


# %%
model.feature_extractor.swinViT._get_name()

for i in model.feature_extractor.named_children():
    print(i)
# %%

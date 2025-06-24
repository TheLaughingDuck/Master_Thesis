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


args = custom_parser()

args.batch_size=3

from itertools import islice

loader, loss_weights = get_loader(args)
#%%
for idx, data in islice(enumerate(loader[1]), 10):
    img, lab = data["images"], data["label"]

    #print(img.shape)
    #print(lab)
    show_image_v2([img[0,:,:,:,:], img[1,:,:,:,:], img[2,:,:,:,:]], maintitle=str(lab))



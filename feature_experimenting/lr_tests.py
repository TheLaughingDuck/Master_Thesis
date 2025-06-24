#%%
# SETUP


import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
#from utils.data_utils import get_loader

from matplotlib import pyplot as plt
import re

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

from utils.parse_arguments import custom_parser

from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, ChainedScheduler, SequentialLR

args = custom_parser()
model = Combined_model()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(parameters, lr=args.optim_lr, weight_decay=args.reg_weight)

#%%

# Assuming optimizer uses lr = 1. for all groups
# lr = 0.09     if epoch == 0
# lr = 0.081    if epoch == 1
# lr = 0.729    if epoch == 2
# lr = 0.6561   if epoch == 3
# lr = 0.59049  if epoch >= 4
scheduler1 = ConstantLR(factor=1, total_iters=20, optimizer=optimizer)
scheduler2 = ConstantLR(factor=0.5, total_iters=200, optimizer=optimizer)
scheduler = SequentialLR(optimizer=optimizer,
                         schedulers=[scheduler1, scheduler2],
                         milestones=[20])

epochs = [i for i in range(100)]
lrs = []
for epo in epochs:
    lrs.append(scheduler.get_last_lr())
    scheduler.step()

import matplotlib.pyplot as plt

plt.plot(epochs, lrs)
# %%

#%%
# SETUP
import torch
import os

from data_loader import get_loader
from trainer import run_training
from resnet_model import resnet3d_18

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


model = resnet3d_18(num_classes=3, in_channels=2)

loaders, loss_weights = get_loader(args)


# %%
# CREATE DATA MATRIX
features = []
labels = []

for id, batch_data in islice(enumerate(loaders[0]), 200):
    data, target = batch_data["image"].to(args.device), batch_data["label"].to(args.device)
    #print(f"DATA SHAPE: {data.shape}")

    #data = data[0]
    #print(f"DATA SHAPE: {data.shape}")

    #x = model(data)

    #batch_size = args.batch_size
    #x = torch.flatten(x)
    #x = torch.nn.AvgPool3d((4,4,4))(x).view(batch_size, 768)[0] # The [0] is to remove the 1 in the shape from batch size 1.

    x = torch.flatten(data)
    features.append(x.detach().to("cpu"))
    labels.append(target.to("cpu"))

features = np.array(features)
labels = np.array(labels)
print(f"FEATURES HAVE DIMS: {features.shape}")
print(f"LABELS HAVE DIMS: {labels.shape}")

# %%
# CREATE PCA PLOT
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
reduced_features = pca.fit_transform(features)
print(pca.explained_variance_ratio_)


plt.scatter(x=reduced_features[:,0], y=reduced_features[:,1], c=labels)
plt.xlabel("PC1")
plt.ylabel("PC2")

# scale_factor = 10*pca.explained_variance_
# plt.arrow(0,0,
#           scale_factor[0]*pca.components_[0,0],scale_factor[0]*pca.components_[0,1],
#           width=0.02) # PC1
# plt.arrow(0,0,
#           scale_factor[1]*pca.components_[1,0],scale_factor[1]*pca.components_[1,1],
#           width=0.02) # PC2
# plt.show()
# # %%

# ax = plt.gca()
# ax.set_aspect("equal")
# plt.arrow(0,0, np.sqrt(scale_factor[0])*pca.components_[0,0],np.sqrt(scale_factor[0])*pca.components_[0,1]) # PC1
# plt.arrow(0,0, np.sqrt(scale_factor[1])*pca.components_[1,0],np.sqrt(scale_factor[1])*pca.components_[1,1]) # PC2
# plt.show()
# pca.explained_variance_


# %%
print(sum(pca.explained_variance_ratio_[0:4]))
# %%
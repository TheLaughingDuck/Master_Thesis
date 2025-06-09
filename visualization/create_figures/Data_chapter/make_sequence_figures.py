'''
I want figures that show some example sequences of the data.

This is a script that creates those figures.
'''

#%%
# SETUP
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

SAVE_DIR = "/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/Data_chapter/figures/"

#%%
# LOAD DATA
IMAGES_PATH = "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/"
train_df = pd.read_csv("/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df.csv")
train_df.loc[0:2, ["T1W-GD", "T2W"]]

images_t1gd = [np.rot90(array_from_path(IMAGES_PATH+obj)) for obj in train_df.loc[0:2, ["T1W-GD"]].values.flatten()]
images_t2 = [np.rot90(array_from_path(IMAGES_PATH+obj)) for obj in train_df.loc[0:2, ["T2W"]].values.flatten()]
print(images[1].shape)

#%%
plt.imshow(images_t1gd[0][0:240, 0:240, 75], cmap="gray")
plt.axis("off")
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(SAVE_DIR+"t1gd_example_0.png", bbox_inches='tight')

plt.imshow(images_t1gd[1][0:240, 0:240, 75], cmap="gray")
plt.axis("off")
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(SAVE_DIR+"t1gd_example_1.png", bbox_inches='tight')
plt.show()


plt.imshow(images_t1gd[2][0:240, 0:240, 75], cmap="gray")
plt.axis("off")
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(SAVE_DIR+"t1gd_example_2.png", bbox_inches='tight')
plt.show()


#%%
# T2W Images
plt.imshow(images_t2[0][0:240, 0:240, 75], cmap="gray")
plt.axis("off")
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(SAVE_DIR+"t2_example_0.png", bbox_inches='tight')
plt.show()

plt.imshow(images_t2[1][0:240, 0:240, 75], cmap="gray")
plt.axis("off")
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(SAVE_DIR+"t2_example_1.png", bbox_inches='tight')
plt.show()


plt.imshow(images_t2[2][0:240, 0:240, 75], cmap="gray")
plt.axis("off")
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(SAVE_DIR+"t2_example_2.png", bbox_inches='tight')
plt.show()


#%%

# images = [IMAGES_PATH+train_df.loc[i, ["T1W-GD"]]["T1W-GD"] for i in range(3)]

# print(images)
# show_image_v2([[images[0]]], maintitle="", titles=[[""]])


# %%

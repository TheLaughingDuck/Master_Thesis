'''
This script as of today 2025-03-05 14:42 seems to run fine! Passed two epochs.
'''


# %%
# SETUP
##################################################################################################################################
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

device = "cpu" # Starting with cpu, so that the data loader will run on the cpu, and the training on gpu
print(f"Using {device} device")











# %%
# DEFINE CLASSIFIER
##################################################################################################################################
import torch.utils.checkpoint as cp

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768*4**3, 12),# (4*128**3, 12), #(768*4**3, 12), #(4*128**3, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.Softmax()
            #nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        #x = cp.checkpoint(self.flatten, x)
        logits = self.linear_relu_stack(x)
        #logits = cp.checkpoint(self.linear_relu_stack, x)
        return logits

model = Classifier().to(device)












# %%
# DEFINE FOUNDATION MODEL
##################################################################################################################################
class EmbedSwinUNETR(SwinUNETR):
    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        # Just return this embedding
        return(dec4)
        
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0, enc0)
        # logits = self.out(out)
        # return logits


BSF_embedder = EmbedSwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True
)
BSF_embedder.to(device)

BSF_embedder.load_state_dict(torch.load("/local/data2/simjo484/BrainSegFounder_custom_finetuning/downstream/BraTS/finetuning/runs/2025-03-05-08:07:48/model_final.pt",
                                        map_location=device)["state_dict"])
BSF_embedder.eval(); print("Set Foundation model to eval mode.\n")









# %%
# STRATIFY DATA
# This is a custom, temporary setup, only used for testing on single T2 images.
##################################################################################################################################

# Define input data transforms
transforms = Compose([
    LoadImaged(keys="images"),
    #EnsureChannelFirst(), # Did not cause problem or worse segmentation when removed
    NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
    Resized(keys="images", spatial_size=(128, 128, 128)),
    #EnsureType(),
    Rotate90d(keys="images", k=3, spatial_axes=(0,1)),
    #EnsureType(),
    ToTensord(keys="images", track_meta=False)#,

    #Lambda(lambda x: [{"images": BSF_embedder(x["images"].view(1,4,128,128,128)), "label": x["label"]}])
    #Lambda(lambda x: print("Before ToDeviced:", type(x["images"])) or x),
    #ToDeviced(keys="images", device=device), # For some reason this causes a RuntimeError
    #Lambda(lambda x: print("After ToDeviced:", type(x["images"])) or x)
])


# Load prepared observation data
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/final_observations_singles.pkl", "rb") as f:
    observations = pickle.load(f)
print(f"Observation data shape: {observations.shape}")


# Filter on only T2 sequences
observations = observations[observations["T2W"] != "---"]

# Make a list of all patient ids
patients = observations.drop_duplicates(subset=["subjetID"])["subjetID"].tolist()

# Create column that indicates which files are available for each patient
observations["file_pattern"] = (
    (~observations["T1W"].isin(["---"])).astype(int).astype(str) +
    (~observations["T1W-GD"].isin(["---"])).astype(int).astype(str) +
    (~observations["T2W"].isin(["---"])).astype(int).astype(str)
)

# Get unique patients & assign them a stratification label
patients = observations.groupby("subjetID")["diagnosis"].agg(lambda x: x.value_counts().idxmax()).reset_index()


# Create a splitter for the data, train proportion 70%
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=104)

# Show the number of observations per stratification key
unique(patients["diagnosis"])

# Perform split
for train_idx, test_idx in splitter.split(patients, patients["diagnosis"]):
    train_patients = patients.iloc[train_idx]["subjetID"]
    test_patients = patients.iloc[test_idx]["subjetID"]

# Split DataFrame
train_df = observations[observations["subjetID"].isin(train_patients)]
test_df = observations[observations["subjetID"].isin(test_patients)]


import itertools

# Assemble train data observation paths
train_data_paths = []
rootpath = "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/"
# for index, obs in observations.iterrows():
#     train_data_paths.append({"images": [rootpath+name for name in  obs[2:].tolist()], "label": 0})
for index, obs in itertools.islice(train_df.iterrows(), 10):
    train_data_paths.append({"images": [rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"]], "label": torch.tensor(obs["label"]).to(device).long()})

train_dataset = data.Dataset(data=train_data_paths, transform=transforms)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=5, shuffle=False, num_workers=12, pin_memory=True
)


# Assemble test/valid data observation paths
test_data_paths = []
for index, obs in itertools.islice(test_df.iterrows(), 10):
    test_data_paths.append({"images": [rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"]], "label": torch.tensor(obs["label"]).to(device).long()})
test_dataset = data.Dataset(data=test_data_paths, transform=transforms)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=5, shuffle=False, num_workers=12, pin_memory=True
)









# %%
# DEFINE TRAIN AND TEST LOOPS
##################################################################################################################################

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = "cpu" # Required for now. Without this we get a RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
#print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, obs in enumerate(dataloader):
        # Unpack input and label
        X = obs["images"]
        y = obs["label"]#.to(device).long()

        # Compute embedding
        X = BSF_embedder(X)
        #print(f"X embedded shape {X_embedded.shape}")

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X) #batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, obs in enumerate(dataloader):
            X = obs["images"]
            y = obs["label"]

            X = BSF_embedder(X)
        #for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

















# %%
# QUE TRAINING MONTAGE
##################################################################################################################################

# Set Hyperparams
learning_rate = 1e-4
#batch_size = 64
epochs = 5

# Set loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
# %%

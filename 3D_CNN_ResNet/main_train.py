# %%
import os
import sys
import re
import pickle
import datetime
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter


from torchsummary import summary

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Activations,
    LoadImaged,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotated,
    Rand3DElasticd,
    EnsureTyped,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    SpatialPadd,
    MapTransform,
    DivisiblePadd,
    Orientationd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    AsDiscrete,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandRotate90d,
    ToTensord,
    Lambdad,
    ScaleIntensityRanged,
)

# %% UTILITIES ################## TRAINING


def train(model, loader, criterion, optimizer, epoch, writer, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, desc=f"Epoch {epoch} Training") as t:
        for batch in t:
            inputs, targets = batch["images"], batch["label"]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, targets_max = torch.max(targets.data, 1)
            total += targets_max.size(0)
            correct += (predicted == targets_max).sum().item()

            t.set_postfix(
                loss=running_loss / len(loader),
                acc=correct / total,
            )

    # for batch in tqdm(loader, desc=f"Epoch {epoch} Training"):
    #     inputs, targets = batch["images"], batch["label"]
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = model(inputs)

    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()

    #     running_loss += loss.item()
    #     _, predicted = torch.max(outputs.data, 1)
    #     _, targets_max = torch.max(targets.data, 1)
    #     total += targets_max.size(0)
    #     correct += (predicted == targets_max).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    writer.add_scalar("loss", epoch_loss, epoch)
    writer.add_scalar("accuracy", epoch_acc, epoch)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, epoch, writer, device):
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, desc=f"Epoch {epoch} Validation") as t:
        for batch_idx, batch in enumerate(t):
            inputs, targets = batch["images"], batch["label"]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, targets_max = torch.max(targets.data, 1)
            total += targets_max.size(0)
            correct += (predicted == targets_max).sum().item()

            t.set_postfix(
                loss=running_loss / (batch_idx + 1),
                acc=correct / total,
            )
    # for batch in tqdm(loader, desc=f"Epoch {epoch} Validation"):
    #     inputs, targets = batch["images"], batch["label"]
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     outputs = model(inputs)
    #     loss = criterion(outputs, targets)

    #     running_loss += loss.item()
    #     _, predicted = torch.max(outputs.data, 1)
    #     _, targets_max = torch.max(targets.data, 1)
    #     total += targets_max.size(0)
    #     correct += (predicted == targets_max).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    writer.add_scalar("loss", epoch_loss, epoch)
    writer.add_scalar("accuracy", epoch_acc, epoch)
    return epoch_loss, epoch_acc


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    predictions = []

    for batch in test_loader:
        inputs, targets = batch["images"], batch["label"]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # apply sigmoid to outputs
        outputs_np = torch.softmax(outputs, dim=1).cpu().numpy()

        predictions.extend(outputs_np)

    return predictions


# %% UTILITIES ################## DATA
def get_train_transform(
    roi_size=[128, 128, 128], scaled_intensity_min=-4.6, scaled_intensity_max=4.6
):
    train_transform = Compose(
        [
            # load  images and stack them together
            LoadImaged(keys="images"),
            EnsureTyped(keys="images"),
            # add channel dim
            EnsureChannelFirstd(
                keys="images"
            ),  # NOTE: this must be present if not Resized does not work
            ScaleIntensityRanged(
                keys="images",
                a_min=scaled_intensity_min,
                a_max=scaled_intensity_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Resized(keys="images", spatial_size=roi_size, mode="area"),
            RandFlipd(keys="images", prob=0.5, spatial_axis=-1),
            RandFlipd(keys="images", prob=0.5, spatial_axis=-2),
            RandFlipd(keys="images", prob=0.5, spatial_axis=-3),
            # transform the label saved in "extra" to one-hot
            # transform the label saved in "extra" to one-hot
            Lambdad(
                keys="label",
                func=lambda x: torch.nn.functional.one_hot(
                    torch.tensor(x), num_classes=config["training"]["num_classes"]
                ).type(torch.float32),
            ),
        ]
    )

    # ###################### Simon version
    # train_transform = Compose(
    #     [
    #         LoadImaged(keys="images"),
    #         NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
    #         Resized(keys="images", spatial_size=(128, 128, 128)),
    #         RandRotate90d(
    #             keys="images", prob=1, max_k=3, spatial_axes=(0, 1)
    #         ),
    #         RandFlipd(keys="images", prob=1, spatial_axis=2),
    #         ToTensord(keys="images", track_meta=False),
    #     ]
    # )

    return train_transform


def get_valid_transform(
    roi_size=[128, 128, 128], scaled_intensity_min=-4.6, scaled_intensity_max=4.6
):
    val_transform = Compose(
        [
            # load  images and stack them together
            LoadImaged(keys="images"),
            EnsureTyped(keys="images"),
            # # add channel dim
            EnsureChannelFirstd(
                keys="images"
            ),  # NOTE: this must be present if not Resized does not work
            ScaleIntensityRanged(
                keys="images",
                a_min=scaled_intensity_min,
                a_max=scaled_intensity_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Resized(keys="images", spatial_size=roi_size, mode="area"),
            # transform the label saved in "extra" to one-hot
            # transform the label saved in "extra" to one-hot
            Lambdad(
                keys="label",
                func=lambda x: torch.nn.functional.one_hot(
                    torch.tensor(x), num_classes=config["training"]["num_classes"]
                ).type(torch.float32),
            ),
        ]
    )
    return val_transform


# %% CONFIGURATIONS (mimics the hydra config file)

config = {
    "work_dir": "/local/data2/simjo484/Training_outputs/3D_ResNet",
    "data_dir": "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files",
    "splits_dir": "/local/data1/simjo484/mt_data/all_data/MRI/simon",
    "dataset": {
        "modalities": ["T1W-GD"], #["T1W", "T1W-GD", "T2W"] # EDIT HERE
        "roi": [128, 128, 128],
    },
    "global_seed": 20251013,
    "training": {
        "num_classes": 3, # EDIT HERE
        "batch_size": 3,
        "learning_rate": 0.00001,#0.00001,
        "num_epoch": 200,
        "num_workers": 18,
        "dropout": 0.5,
        "weight_decay": 0.001,
        "early_stopping_patience": 50,
    },
    "testing":{
        "create_test_predictions": True#,
        #"model_name": "ResNet_2p1_t1gd_diag",
        #"logdir": "" # The directory where everything about this specific model run is saved. e.g. "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-150823 (resnet_mixed_conv) (t2) (diag)"
    },
    "models": {
        "model": "resnet_mixed_conv",  # resnet2p1, resnet_mixed_conv, 3dconv # EDIT HERE
        "stem_type": "None", # NOT USED  # resnet_mixed_convStem, conv3dStem, resnet2p1Stem
        "pretrain": True, # EDIT HERE
    },
}


#%%
# SET SPECIFIC configurations for generating test data predictions
if config["testing"]["create_test_predictions"] == True:
    pass

    # # ResNet 2p1 T1W-GD DIAGNOSE Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_2p1_t1gd_diag"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-105151 (resnet2p1) (t1gd) (diag)"
    # config["models"]["model"] = "resnet2p1"
    # config["dataset"]["modalities"] = ["T1W-GD"]
    # config["training"]["num_classes"] = 3

    # # ResNet 2p1 T2W DIAGNOSE Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_2p1_t2_diag"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-130837 (resnet2p1) (t2) (diag)"
    # config["models"]["model"] = "resnet2p1"
    # config["dataset"]["modalities"] = ["T2W"]
    # config["training"]["num_classes"] = 3

    # # ResNet 2p1 fused T1W-GD and T2W DIAGNOSE Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_2p1_t1gd_and_t2_diag"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-070626 (resnet2p1) (t1gd and t2) (diag)"
    # config["models"]["model"] = "resnet2p1"
    # config["dataset"]["modalities"] = ["T1W-GD", "T2W"]
    # config["training"]["num_classes"] = 3



    # # ResNet Mixed T1W-GD DIAGNOSE Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_mixed_t1gd_diag"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-082835 (resnet_mixed_conv) (t1gd) (diag)"
    # config["models"]["model"] = "resnet_mixed_conv"
    # config["dataset"]["modalities"] = ["T1W-GD"]
    # config["training"]["num_classes"] = 3

    # ResNet Mixed T2W DIAGNOSE Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_mixed_t2_diag"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-150823 (resnet_mixed_conv) (t2) (diag)"
    # config["models"]["model"] = "resnet_mixed_conv"
    # config["dataset"]["modalities"] = ["T2W"]
    # config["training"]["num_classes"] = 3

    # ResNet Mixed fused T1W-GD and T2W DIAGNOSE Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_mixed_t1gd_and_t2_diag"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-085122 (resnet_mixed_conv) (t1gd and t2) (diag)"
    # config["models"]["model"] = "resnet_mixed_conv"
    # config["dataset"]["modalities"] = ["T1W-GD", "T2W"]
    # config["training"]["num_classes"] = 3



    # ResNet 2p1 T1W-GD LOCATION Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_2p1_t1gd_loc"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-124822 (resnet2p1) (t1gd) (loc)"
    # config["models"]["model"] = "resnet2p1"
    # config["dataset"]["modalities"] = ["T1W-GD"]
    # config["training"]["num_classes"] = 2

    # ResNet 2p1 T2W LOCATION Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_2p1_t2_loc"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-141009 (resnet2p1) (t2) (loc)"
    # config["models"]["model"] = "resnet2p1"
    # config["dataset"]["modalities"] = ["T2W"]
    # config["training"]["num_classes"] = 2

    # # ResNet 2p1 fused T1W-GD and T2W LOCATION Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_2p1_t1gd_and_t2_loc"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (t1gd and t2) (loc)"
    # config["models"]["model"] = "resnet2p1"
    # config["dataset"]["modalities"] = ["T1W-GD", "T2W"]
    # config["training"]["num_classes"] = 2



    # # ResNet Mixed T1W-GD LOCATION Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_mixed_t1gd_loc"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-171150 (resnet_mixed_conv) (t1gd) (loc)"
    # config["models"]["model"] = "resnet_mixed_conv"
    # config["dataset"]["modalities"] = ["T1W-GD"]
    # config["training"]["num_classes"] = 2

    # # ResNet Mixed T2W LOCATION Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_mixed_t2_loc"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-161408 (resnet_mixed_conv) (t2) (loc)"
    # config["models"]["model"] = "resnet_mixed_conv"
    # config["dataset"]["modalities"] = ["T2W"]
    # config["training"]["num_classes"] = 2

    # # ResNet Mixed fused T1W-GD and T2W LOCATION Model
    # # HAS BEEN RUN
    # config["testing"]["model_name"] = "ResNet_mixed_t1gd_and_t2_loc"
    # config["testing"]["logdir"] = "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (t1gd and t2) (loc)"
    # config["models"]["model"] = "resnet_mixed_conv"
    # config["dataset"]["modalities"] = ["T1W-GD", "T2W"]
    # config["training"]["num_classes"] = 2

# %% DATASET - load file paths and labels

# image files
IMG_ROOT_DIR = config["data_dir"]
# check if it exists
if not os.path.exists(IMG_ROOT_DIR):
    raise ValueError("ROOT_DIR does not exist")

# labels
LABEL_ROOT_DIR = config["splits_dir"]
if config["training"]["num_classes"] == 2: label_column = "loc_label" # The name of the label column in the data
elif config["training"]["num_classes"] == 3: label_column = "class_label"

# load the dataframes
if config["training"]["num_classes"] == 2:
    train_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "train_df_loc.csv"))
    valid_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "valid_df_loc.csv"))
    test_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "test_df_loc.csv"))
elif config["training"]["num_classes"] == 3:
    train_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "train_df.csv"))
    valid_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "valid_df.csv"))
    test_df = pd.read_csv(os.path.join(LABEL_ROOT_DIR, "test_df.csv"))

# filter out those that do not have the modalities that we need
for modality in config["dataset"]["modalities"]:
    train_df = train_df[train_df[modality].notnull()]
    valid_df = valid_df[valid_df[modality].notnull()]
    test_df = test_df[test_df[modality].notnull()]
    
    # and different from "---"
    train_df = train_df[train_df[modality] != "---"]
    valid_df = valid_df[valid_df[modality] != "---"]
    test_df = test_df[test_df[modality] != "---"]

# check that the files exist
for modality in config["dataset"]["modalities"]:
    for idx, row in train_df.iterrows():
        if not os.path.isfile(os.path.join(IMG_ROOT_DIR, row[modality])):
            raise ValueError(
                f"File {os.path.join(IMG_ROOT_DIR, row[modality])} does not exist"
            )

    for idx, row in valid_df.iterrows():
        if not os.path.isfile(os.path.join(IMG_ROOT_DIR, row[modality])):
            raise ValueError(
                f"File {os.path.join(IMG_ROOT_DIR, row[modality])} does not exist"
            )

# build list of dictionaries with image and integer label
train_dataset = [
    {
        "images": [
            os.path.join(IMG_ROOT_DIR, row[modality])
            for modality in config["dataset"]["modalities"]
        ],
        "label": row[label_column],
    }
    for idx, row in train_df.iterrows()
]
valid_dataset = [
    {
        "images": [
            os.path.join(IMG_ROOT_DIR, row[modality])
            for modality in config["dataset"]["modalities"]
        ],
        "label": row[label_column],
    }
    for idx, row in valid_df.iterrows()
]
test_dataset = [
    {
        "images": [
            os.path.join(IMG_ROOT_DIR, row[modality])
            for modality in config["dataset"]["modalities"]
        ],
        "label": row[label_column],
    }
    for idx, row in test_df.iterrows()
]
# print the number of samples
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in valid dataset: {len(valid_dataset)}")
print(f"Number of samples in test dataset: {len(test_dataset)}")

# compute class weights based on the training dataset. Larger weights to the less represented classes
# get unique class labels
class_labels = train_df[label_column].unique()
# sort them
class_labels = np.sort(class_labels)
# get the number of subjects per class
class_counts = [len(train_df[train_df[label_column] == i]) for i in class_labels]
# compute the class weights
class_weights = np.array(
    [
        sum(class_counts) / (len(class_counts) * class_counts[i])
        for i in range(len(class_counts))
    ]
)
# normalize the weights
class_weights = class_weights / class_weights.sum()

# print the class weights (label : nbr subjects : class weight)
for i in range(len(class_labels)):
    print(
        f"Label {class_labels[i]}: {class_counts[i]:3d} subjects, weight {class_weights[i]:0.4f}"
    )

# convert to tensor
class_weights = torch.tensor(class_weights).float()

# %% DATASET - create the dataset and dataloaders

# set the seed
torch.manual_seed(config["global_seed"])
# create the dataset
train_dataset = Dataset(
    data=train_dataset,
    transform=get_train_transform(roi_size=config["dataset"]["roi"]),
)
valid_dataset = Dataset(
    data=valid_dataset,
    transform=get_valid_transform(roi_size=config["dataset"]["roi"]),
)
test_dataset = Dataset(
    data=test_dataset,
    transform=get_valid_transform(roi_size=config["dataset"]["roi"]),
)

# create the dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["training"]["num_workers"],
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["training"]["num_workers"],
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["training"]["num_workers"],
)

# %% CHECK DATALOADER

from monai.utils import set_determinism

# get next batch
batch = next(iter(train_dataloader))

# get image and label
image = batch["images"]
label = batch["label"]

# check shapes
print(image.shape)
print(label)

# plot image
fig, ax = plt.subplots(1, len(config["dataset"]["modalities"]), figsize=(20, 20))
if len(config["dataset"]["modalities"]) == 1:
    ax = [ax]
else:
    ax = ax.flatten()
# plot each modality
for midx, m in enumerate(config["dataset"]["modalities"]):
    ax[midx].imshow(image[0, midx, :, :, 70].numpy().squeeze(), cmap="gray")
    ax[midx].set_title(m)
plt.show()

# plot histogram
fig, ax = plt.subplots(1, len(config["dataset"]["modalities"]), figsize=(20, 20))
if len(config["dataset"]["modalities"]) == 1:
    ax = [ax]
else:
    ax = ax.flatten()
# plot each modality
for midx, m in enumerate(config["dataset"]["modalities"]):
    ax[midx].hist(image[0, midx, :, :, :].numpy().squeeze().flatten(), bins=100)
    ax[midx].set_title(m)
    ax[midx].set_yscale("log")

plt.show()

# %% DEFINE MODEL
sys.path.append(config["work_dir"])
import stems
import torchvision
import importlib

importlib.reload(stems)

# load model and fix the model stem
if config["models"]["model"] == "resnet2p1":
    model = torchvision.models.video.r2plus1d_18(
        pretrained=config["models"]["pretrain"]
    )
    model.stem = stems.R2Plus1dStem4MRI(channels=len(config["dataset"]["modalities"]))
elif config["models"]["model"] == "resnet_mixed_conv":
    model = torchvision.models.video.mc3_18(pretrained=config["models"]["pretrain"])
    model.stem = stems.modifybasicstem(channels=len(config["dataset"]["modalities"]))
elif config["models"]["model"] == "3dconv":
    model = torchvision.models.video.r3d_18(pretrained=config["models"]["pretrain"])
    model.stem = stems.modifybasicstem(channels=len(config["dataset"]["modalities"]))
else:
    raise ValueError(f"Model {config['models']['model']} not supported")

# change the number of classes and add regularization
model.fc = nn.Sequential(
    nn.Dropout(config["training"]["dropout"]),
    nn.Linear(model.fc.in_features, config["training"]["num_classes"]),
)


# print model summary
summary(
    model,
    input_size=(
        len(config["dataset"]["modalities"]),
        config["dataset"]["roi"][0],
        config["dataset"]["roi"][1],
        config["dataset"]["roi"][2],
    ),
    device="cpu",
)

# put model on device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

model.to(device)

# %% DEFINE LOSS AND OPTIMIZER

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
)

# %%
# LAST SETTINGS BEFORE TRAINING
if not config["testing"]["create_test_predictions"]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    linear_probing_dir = Path(config["work_dir"], "linear_probing", timestamp)
    linear_probing_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(linear_probing_dir, "checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ###### Save config
    import json
    with open(linear_probing_dir.__str__()+"/config.json", 'w') as file:
        json.dump(config, file)
    ######

    writer_train = SummaryWriter(
        log_dir=Path(linear_probing_dir, "logs", "train", timestamp)
    )
    writer_val = SummaryWriter(
        log_dir=Path(linear_probing_dir, "logs", "validate", timestamp)
    )

# %% TRAINING LOOP
# Early Stopping and Checkpoint Initialization
if not config["testing"]["create_test_predictions"]:
    best_val_loss = 1000
    patience_counter = 0

    for epoch in range(config["training"]["num_epoch"]):
        print(f"Epoch {epoch + 1}/{config['training']['num_epoch']}\r", end="")
        start_time = time.time()
        train(model, train_dataloader, criterion, optimizer, epoch, writer_train, device)

        val_loss, val_acc = validate(
            model, valid_dataloader, criterion, epoch, writer_val, device
        )

        print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")

        torch.save(model.state_dict(), Path(checkpoints_dir, f"checkpoint_{epoch}.pth"))

        # Checkpoint and Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint
            torch.save(
                model.state_dict(),
                Path(checkpoints_dir, "best_checkpoint.pth"),
            )
            print("Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping")
                break

    writer_train.close()
    writer_val.close()

# %% TEST
if not config["testing"]["create_test_predictions"]:
    # Load Best Model for Test Evaluation
    model.load_state_dict(torch.load(Path(checkpoints_dir, "best_checkpoint.pth")))

    predictions = test(model, valid_dataloader, device)

    linear_probing_results = Path(linear_probing_dir, "linear_probing_results.csv")
    print(f"Saving predictions to {linear_probing_results}")
    pd.DataFrame(predictions).to_csv(linear_probing_results, index=False)



# %%
# CREATE TEST SET CLASS PREDICTIONS

PREDICTION_PATH = "/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/statistical_tests/class_predictions"

if config["testing"]["create_test_predictions"]:
    print(f"USING MODEL: {config["testing"]["model_name"]}")
    # Load Best Model for Test Evaluation
    model.load_state_dict(torch.load(Path(config["testing"]["logdir"], "checkpoints/best_checkpoint.pth")))

    predictions = test(model, test_dataloader, device)

    linear_probing_results = Path(PREDICTION_PATH, config["testing"]["model_name"]+".csv")
    print(f"Saving test data predictions to {linear_probing_results}")
    pd.DataFrame(predictions).to_csv(linear_probing_results, index=False)
# %%

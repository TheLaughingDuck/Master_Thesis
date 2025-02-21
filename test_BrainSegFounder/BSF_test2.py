# %%
# Import dependencies
#####################
import torch
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt

from monai.networks.nets import SwinUNETR
from monai import data
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,
    Resized,
    ToTensord,
    Compose
)



# %%
# Set up model architecture
###########################
# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu" # Required for now. Without this we get a RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

# Load architecture
model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    use_checkpoint=True,
)
model.to(device)

# Load weights
checkpoint = torch.load("/local/data2/simjo484/BrainSegFounder_models/finetuned_model_fold_0.pt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Create transforms for input images
transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    #ScaleIntensityd(keys=["image"]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image"], track_meta=False, device=device)
])

# Format sequence paths
patient_paths = ["/local/data2/simjo484/BraTS_Iulians_testdata/00759/BraTS2021_00759_"]
image_filenames = [[path + seq_type + ".nii.gz" for seq_type in ["t1", "t1ce", "t2", "flair"] for path in patient_paths]]
image_filenames = [{"image": name} for name in image_filenames]

val_ds = data.Dataset(data=image_filenames, transform=transforms)
val_loader = data.DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=12, pin_memory=True
)



# %% Yield image and run model on one patient
###################################################################
x = next(iter(val_loader))["image"]
x = torch.rot90(x, k=3, dims=(2,3))

with torch.no_grad():
    output = model(x[:,:4,:,:,:])
    output = torch.softmax(output, dim=1)
    predicted_segmentation = torch.argmax(output, dim=1).cpu().numpy()

slice_index = 60
threshold_output = torch.where(output > 0.99, torch.tensor(1, device=output.device), torch.tensor(0, device=output.device))
plt.imshow(x[0, 1, :, :, slice_index], cmap="gray");plt.title("BraTS patient 00759 input channel T1GD");plt.show()
plt.imshow(threshold_output[0, 0, :, :, slice_index], cmap="gray");plt.title("BraTS patient 00759 output channel 0, threshold 0.99");plt.show()

#plt.hist(output[0,0,:,:].detach().numpy().ravel()); plt.show()



# %%

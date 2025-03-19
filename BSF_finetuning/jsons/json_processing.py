'''
Script that loads the "brats21_folds.json" file, and creates and saves a new .json file,
with appropriately adjusted filenames. Here is one example element from the "training" part:

{
    "fold": 0,
    "image": [
        "TrainingData/BraTS2021_01146/BraTS2021_01146_flair.nii.gz",
        "TrainingData/BraTS2021_01146/BraTS2021_01146_t1ce.nii.gz",
        "TrainingData/BraTS2021_01146/BraTS2021_01146_t1.nii.gz",
        "TrainingData/BraTS2021_01146/BraTS2021_01146_t2.nii.gz"
    ],
    "label": "TrainingData/BraTS2021_01146/BraTS2021_01146_seg.nii.gz"
}

If we want to train BSF on only T2W images, we make all four filenames be the T2W-filename,
so that the same T2W file is fed into the model during fine-tuning.
'''

#%%

import json

exit() # To stop accidentally running this script

# Load the provided JSON file
file_path = 'brats21_folds.json'  # Replace with your file path

with open(file_path, 'r') as file:
    data = json.load(file)

# Change image paths
for entry in data['training']:
    entry['image'] = [entry["image"][1], entry["image"][1], entry["image"][2], entry["image"][2]] #entry['image'][2:]

# Save the modified data back to a new JSON file
modified_file_path = 'brats21_folds_T1-GD_and_T1_modality.json'  # Replace with your desired file path
with open(modified_file_path, 'w') as file:
    json.dump(data, file)

#
# %%

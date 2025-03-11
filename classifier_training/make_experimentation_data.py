'''
This script creates two pickle files of T2 image paths. It is only designed for experimenting with the
model architecture etc
'''

import pickle
import torch


pp_device = "cpu" # Set processing device to cpu

exit() # Only run this script once!

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


# Perform split
for train_idx, test_idx in splitter.split(patients, patients["diagnosis"]):
    train_patients = patients.iloc[train_idx]["subjetID"]
    test_patients = patients.iloc[test_idx]["subjetID"]

# Split DataFrame
train_df = observations[observations["subjetID"].isin(train_patients)]
test_df = observations[observations["subjetID"].isin(test_patients)]



import itertools
rootpath = "/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/"

# Assemble train data observation paths
train_data_paths = []
for index, obs in train_df.iterrows():
    train_data_paths.append({"images": [rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"]], "label": torch.tensor(obs["label"]).to(pp_device).long()})


# Assemble validation data observation paths
valid_data_paths = []
for index, obs in test_df.iterrows():
    valid_data_paths.append({"images": [rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"], rootpath+obs["T2W"]], "label": torch.tensor(obs["label"]).to(pp_device).long()})


save_dir = "/home/simjo484/master_thesis/Master_Thesis/classifier_training"
with open(save_dir+"t2_training_paths.pkl", "wb") as f:
    pickle.dump(train_data_paths, f)

with open(save_dir+"t2_valid_paths.pkl", "wb") as f:
    pickle.dump(valid_data_paths, f)
# %%

'''
This script takes a processed meta data file, where each row represents an individual sequence/volume.
It then produces a new data file, where each row represents one observation,
with columns subjetID, diagnose, and filenames for each sequence.

The result is stored as a pickle.
'''

# %%
# SETUP
import torch

import pandas as pd
import pickle
import os
from utils import *

# Load prepared meta data
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/meta.pkl", "rb") as f:
    meta = pickle.load(f)
print(f"Meta data shape: {meta.shape}")

# Filter out volumes for which we have not found the filename (Is already done in process_meta.py)
#meta = meta[meta["file_found"] == True] # The rows should, and appear to be, unique. It contained 4154 rows last I checked, and there are 4633 files in the Final preprocessed files. Manageable.
print(f"Loaded volumes shape: {meta.shape}")


# Add a counter index for duplicate id, session, sequence
meta["type_counter"] = meta.groupby(["subjetID", "session_name", "seq_type", "diagnosis"]).cumcount()+1

observations = meta.pivot(index=["subjetID", "session_name", "diagnosis", "type_counter"], 
                          values="found_filename",
                          columns="seq_type")
observations = observations.reset_index()
observations = observations.fillna("unknown")
print(f"Observations shape after pivoting: {observations.shape}")


# %%
# PRODUCE SOME INFORMATION
print(f"Number of T1W volumes: {observations[observations["T1W"] != "unknown"].shape[0]}")
print(f"Number of T1W-GD volumes: {observations[observations["T1W-GD"] != "unknown"].shape[0]}")
print(f"Number of T2W volumes: {observations[observations["T2W"] != "unknown"].shape[0]}")
print("")
print(f"Number of patients with T1W, T1W-GD and T2W: {observations[(observations["T1W"] != "unknown") & (observations["T1W-GD"] != "unknown") & (observations["T2W"] != "unknown")].shape}")
print(f"Number of patients with T1W and T1W-GD: {observations[(observations["T1W"] != "unknown") & (observations["T1W-GD"] != "unknown")].shape[0]}")
print(f"Number of patients with T1W and T2W: {observations[(observations["T1W"] != "unknown") & (observations["T2W"] != "unknown")].shape[0]}")
print(f"Number of patients with T1W-GD and T2W: {observations[(observations["T1W-GD"] != "unknown") & (observations["T2W"] != "unknown")].shape[0]}")

print("\nNumber of patients for each diagnose:")
print(unique(observations.drop_duplicates(subset=["subjetID", "diagnosis"])["diagnosis"]))
print("\n")

# %%
# Format and save the observation data locally as "observations.pkl".
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/observations.pkl", "wb") as f:
    pickle.dump(observations, f)
print("Saved observation data to observations.pkl (supposedly)")




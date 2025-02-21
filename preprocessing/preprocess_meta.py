'''
This script loads patient meta data from an excel spreadsheet, and performs filtering, renaming etc.
The result is stored locally for further processing.
'''


# %%
# Import Dependencies and setup
###################################################
import pandas as pd
import pickle

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis") # set working directory, useful for running .py files as notebooks.
from utils import *

# %%
# Load raw excel data
data_raw = pd.read_excel("/local/data1/simjo484/mt_data/all_data/MRI/MRI_summary_extended_simon.xlsx")


# %%
# Preprocess meta data
###################################################
# Filter on pre_op
data = data_raw[data_raw["session_status"] == "pre_op"]


# Rename diagnoses
rename = {"Low-grade glioma/astrocytoma (WHO grade I/II)": "Low-Grade Glioma",
          "Medulloblastoma": "Medulloblastoma",
          "High-grade glioma/astrocytoma (WHO grade III/IV)": "High-Grade Glioma",
          "Ganglioglioma": "Ganglioglioma",
          "Ependymoma": "Ependymoma",
          "Atypical Teratoid Rhabdoid Tumor (ATRT)": "ATRT",
          "Brainstem glioma- Diffuse intrinsic pontine glioma": "DIPG",
          "Craniopharyngioma": "Craniopharyngioma",
          "Rhabdomyosarcoma": "Rhabdomyosarcoma",
          "Supratentorial or Spinal Cord PNET": "SP-PNET",
          "Neurofibroma/Plexiform": "Neurofibroma",
          "Other": "Other",
          "Neurocytoma": "Neurocytoma",
          "Subependymal Giant Cell Astrocytoma (SEGA)": "SEGA",
          "Malignant peripheral nerve sheath tumor (MPNST)": "MPNST"}

data.loc[:,"diagnosis"] = [rename[diag] for diag in data["diagnosis"]]


# Filter out diagnoses
data = data[data["diagnosis"] != "Other"]

# Combine sequence types
new_sequences = []
for seq_type in data["image_type"]:
    new_type = seq_type

    # Mark these for removal
    if seq_type in ["UNKNOWN", "T1W_FSPGR_GD", "T1W_FSPGR", "SWAN", "SWI", "ASL", "MAG", "PHE", "ANGIO", "Vs3D", "FD", "EXP"]: new_type = "remove"
    
    # These are T1W
    elif seq_type in ["T1W_SE", "T1W"]: new_type = "T1W"

    # These are T1W-GD
    elif seq_type in ["T1W_SE_GD", "T1W_SE_GD", "T1W_GD", "T1W-GD"]: new_type = "T1W-GD"

    # These are T1W FLAIR
    elif seq_type in ["T1W_FL"]: new_type = "T1W_FLAIR"

    # These are T1W FLAIR GAD
    elif seq_type in ["T1W_FL_GD"]: new_type = "T1W_FLAIR_GD"

    # These are MPR (separate from T1)
    elif seq_type in ["T1W_MPRAGE"]: new_type = "MPR"

    # These are T1W-MPRAGE_GD
    elif seq_type in ["T1W_MPRAGE_GD"]: new_type = "T1W_MPRAGE_GD"
    
    # These are T2W
    elif seq_type in ["FSE", "tse", "T2W"]: new_type = "T2W"

    # These are FLAIR
    elif seq_type in ["T2W_TRIM", "FLAIR", "T2W_FLAIR"]: new_type = "FLAIR"

    ## These are T2W_FLAIR
    #elif seq_type in []: new_type = "T2W_FLAIR"

    new_sequences.append(new_type)
data.loc[:, "image_type"] = new_sequences

# Filter out unknown image types
data = data[data["image_type"] != "remove"]

# Filter out based on manually annotated Notes
data = data[data["Notes_simon"] != "remove"]


# Do *not* filter out diagnoses yet.



# %%
# Format and save the meta data locally as "meta.pkl".
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/meta.pkl", "wb") as f:
    pickle.dump(data, f)

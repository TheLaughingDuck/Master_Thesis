'''
This script loads patient meta data from an excel spreadsheet, and performs filtering, renaming etc.
The result is stored locally for further processing. There are a few stepwise saves. The final output
is pickle file of a pandas dataframe with the observations (with filenames to the volumes).
'''
# %%
# SETUP META PROCESSING
###################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import pickle

import os
from utils import *

# Load raw excel data
data_raw = pd.read_excel("/local/data1/simjo484/mt_data/all_data/MRI/MRI_summary_extended_simon.xlsx")



# %%
# META PROCESSING SETTINGS

# Sometimes there are multiple files for the same patient, same session, and same sequence type (like T2W for example).
# If True, drop all but one
drop_duplicates = True


# %%
# PROCESS META DATA
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


# Filter out some diagnoses
data = data[data["diagnosis"] != "Other"]


# Combine sequence types
new_sequences = []
for seq_type in data["image_type"]:
    new_type = seq_type

    # Mark these for removal
    if seq_type in ["UNKNOWN", "T1W_FSPGR_GD", "T1W_FSPGR", "SWAN", "SWI", "ASL", "MAG", "PHE", "ANGIO", "Vs3D", "FD", "EXP"]: new_type = "remove"
    
    # These are T1W
    elif seq_type in ["T1W_SE", "T1W", "T1W_MPRAGE", "MPR"]: new_type = "T1W"

    # These are T1W-GD
    elif seq_type in ["T1W_SE_GD", "T1W_SE_GD", "T1W_GD", "T1W-GD", "T1W_MPRAGE_GD"]: new_type = "T1W-GD"

    # These are T1W FLAIR
    elif seq_type in ["T1W_FL"]: new_type = "T1W_FLAIR"

    # These are T1W FLAIR GAD
    elif seq_type in ["T1W_FL_GD"]: new_type = "T1W_FLAIR_GD"

    # These are MPR (separate from T1)
    #elif seq_type in ["T1W_MPRAGE"]: new_type = "MPR"

    # These are T1W-MPRAGE_GD
    #elif seq_type in ["T1W_MPRAGE_GD"]: new_type = "T1W_MPRAGE_GD"
    
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


# Filter away some image types (sequence types), and then create dummies for the remaining sequences
seq_group = ["T1W", "T1W-GD", "T2W"]
meta_unique_seqs = data[data["image_type"].isin(seq_group)]
#meta_unique_seqs["seq_type"] = meta_unique_seqs["image_type"]
meta_unique_seqs = meta_unique_seqs.assign(seq_type=meta_unique_seqs["image_type"])
meta_seq_dummies = pd.get_dummies(meta_unique_seqs, columns=["image_type"], prefix="Seq", prefix_sep="_")

# Create column with a shortened file_name (suitable for concatting below)
meta_seq_dummies.loc[:, "short_file_name"] = meta_seq_dummies["file_name"].str.replace(r'(.*?/FILES/)', "", regex=True).str.replace(r'\.json', "", regex=True)

# Create a column indicating whether the file corresponding to this record has been located.
meta_seq_dummies.loc[:, "file_found"] = pd.Series([False for i in range(meta_seq_dummies.shape[0])])
meta_seq_dummies.loc[:, "found_filename"] = pd.Series([False for i in range(meta_seq_dummies.shape[0])])

print(f"Meta data shape: {meta_seq_dummies.shape}")



# %%
# MATCH FILENAMES
# Dictionary with the number of files that were, and were *not* matched/found among the files
matches = {seq: 0 for seq in seq_group}
non_matches = {seq: 0 for seq in seq_group}

# Get file names (Final processed)
file_names = os.listdir("/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/")

import difflib

printout = False

remaining_file_names = file_names.copy()

for index, obs in meta_seq_dummies.iterrows():

    id, session, name, seq_type = obs[["subjetID", "session_name", "short_file_name", "seq_type"]]

    id = f"PP_C{id}___"
    session = session+"___"
    name = name+".nii.gz"
    # Constructed file name
    con_file_name = id+session+name

    if con_file_name not in remaining_file_names:
        # Record the non-match
        non_matches[seq_type] += 1

        if printout:
            print("-------------------------------------------------------")
            print("Constructed filename ("+seq_type+")")
            print(con_file_name)

            # Find candidates with heuristic
            candidates = remaining_file_names.copy()
            a = lambda x: (id in x) and (session in x)
            candidates = [x for x in candidates if a(x)]

            if len(candidates) != 0:
                print("\nHeuristic candidates:")
                for cand in candidates: print(cand)
                print('')
            
            # Find candidates with difflib
            print("\nDifflib candidates:")
            closematches = difflib.get_close_matches(con_file_name, candidates)
            for cand in closematches:
                print(cand)
            print('')
    else:
        # Make a note that the file for this record has been found
        meta_seq_dummies.loc[index, "file_found"] = True
        meta_seq_dummies.loc[index, "found_filename"] = con_file_name

        remaining_file_names.remove(con_file_name)
        matches[seq_type] += 1


print("")
for key in non_matches:
    print(key+" non-matches: "+str(non_matches[key]))

print("")
for key in matches:
    print(key+" matches: "+str(matches[key]))

print("\nTotal: "+str(sum(non_matches.values()) + sum(matches.values())))


# Filter on files that were found
meta_seq_dummies = meta_seq_dummies[meta_seq_dummies["file_found"] == True]

# %%
# SAVE META DATA
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/meta.pkl", "wb") as f:
    pickle.dump(meta_seq_dummies, f)
print("Saved processed meta data to meta.pkl (supposedly)")
meta = meta_seq_dummies

# %%
# PIVOT TO GET DATAFRAME WITH OBSERVATIONS

# Drop duplicates
# See top of this file (META PROCESSING SETTINGS)
if drop_duplicates:
    observations = meta.drop_duplicates(subset=["subjetID", "session_name", "seq_type"])
    observations = observations.pivot(index=["subjetID", "session_name", "diagnosis"], 
                            values="found_filename",
                            columns="seq_type")
    observations = observations.reset_index()
    observations = observations.fillna("---")
    print(f"Observations shape after pivoting (and dropping duplicates): {observations.shape}")
else:
    meta["type_counter"] = meta.groupby(["subjetID", "session_name", "seq_type", "diagnosis"]).cumcount()+1

    observations = meta.pivot(index=["subjetID", "session_name", "diagnosis", "type_counter"], 
                            values="found_filename",
                            columns="seq_type")
    observations = observations.reset_index()
    observations = observations.fillna("---")
    print(f"Observations shape after pivoting (and *not* dropping duplicates): {observations.shape}")



# %%
# SAVE OBSERVATION DATA
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/observations.pkl", "wb") as f:
    pickle.dump(observations, f)
print("Saved observation data to observations.pkl (supposedly)")



# %%
# CREATE BINARY COLUMNS FOR SEQ TYPE PAIRS
observations["T1_and_T1GD"] = (observations["T1W"] != "---") & (observations["T1W-GD"] != "---")
observations["T1GD_and_T2"] = (observations["T1W-GD"] != "---") & (observations["T2W"] != "---")
observations["T1_and_T2"] = (observations["T1W"] != "---") & (observations["T2W"] != "---")
observations["all_three"] = (observations["T1W"] != "---") & (observations["T1W-GD"] != "---") & (observations["T2W"] != "---")


# %%
# ASSIGN LABELS
observations["label"] = [None for i in range(observations.shape[0])]
for i in range(observations.shape[0]):
    diagnosis = observations.loc[i, "diagnosis"]

    if diagnosis in ['Low-Grade Glioma', 'Ganglioglioma']: label = 0
    elif diagnosis in ['Ependymoma']: label = 1
    elif diagnosis in ['Medulloblastoma', 'ATRT']: label = 2
    elif diagnosis in ['High-Grade Glioma']: label = 3
    else: label = "remove"

    observations.loc[i, "label"] = label

# Filter out unused diagnoses
observations = observations[observations["label"] != "remove"]



# %%
# SAVE OBSERVATION DATA (with single files)
# We do this in order to be able to test out training on only one sequence type.
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/final_observations_singles.pkl", "wb") as f:
    pickle.dump(observations, f)
print("Saved observation data (including single files) to final_observations.pkl (supposedly)")


# %%
# SPLIT OBSERVATIONS INTO TRAIN AND TEST DATA
# Make a list of all patient ids
patients = observations.drop_duplicates(subset=["subjetID"])["subjetID"].tolist()

# Create column that indicates which files are available for each patient
observations["file_pattern"] = (
    (~observations["T1W"].isin(["---"])).astype(int).astype(str) +
    (~observations["T1W-GD"].isin(["---"])).astype(int).astype(str) +
    (~observations["T2W"].isin(["---"])).astype(int).astype(str)
)


# Filter out observations that only feature one file
observations = observations[~observations["file_pattern"].isin(["100", "010", "001"])]

# Attach diagnosis to the stratification key
observations["stratify_key"] = observations["diagnosis"] + "_" + observations["file_pattern"]

# Filter out "Ependymoma_110" and "Low-Grade Glioma_110", because they only feature one observation each
observations = observations[~observations["stratify_key"].isin(["Ependymoma_110", "Low-Grade Glioma_110"])]

# Get unique patients & assign them a stratification label
patients = observations.groupby("subjetID")["stratify_key"].agg(lambda x: x.value_counts().idxmax()).reset_index()

# Create a splitter for the data, train proportion 70%
splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)

# Show the number of observations per stratification key
unique(patients["stratify_key"])

# Perform split
for train_idx, test_idx in splitter.split(patients, patients["stratify_key"]):
    train_patients = patients.iloc[train_idx]["subjetID"]
    test_patients = patients.iloc[test_idx]["subjetID"]

# Split DataFrame
train_df = observations[observations["subjetID"].isin(train_patients)]
test_df = observations[observations["subjetID"].isin(test_patients)]



# %%
# PRODUCE TRAIN OBSERVATION SUMMARY
observation_summary(train_df, title="Train data")

# Save train df
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df.pkl", "wb") as f:
    pickle.dump(train_df, f)
print("Saved train data (supposedly)")



# %%
# PRODUCE TEST OBSERVATION SUMMARY
observation_summary(test_df, title="Test data")

# Save test df
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.pkl", "wb") as f:
    pickle.dump(test_df, f)
print("Saved test data (supposedly)")



# %%

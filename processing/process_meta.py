'''
This script loads patient meta data from an excel spreadsheet, and performs filtering, renaming etc.
The result is stored locally for further processing. There are a few stepwise saves. The final output
is pickle file of a pandas dataframe with the observations (with filenames to the volumes).
'''
# %%
# SETUP META PROCESSING
###################################################
import pandas as pd
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
# PRODUCE SOME INFORMATION
print(f"Total number of rows (observations): {observations.shape[0]}")
print(f"Total number of unique (patient, session): {observations.drop_duplicates(subset=["subjetID", "session_name"]).shape[0]}")
print(f"Total number of unique patients: {observations.drop_duplicates(subset=["subjetID"]).shape[0]}")
print("")

print(f"Number of T1W volumes: {observations[observations["T1W"] != "---"].shape[0]}")
print(f"Number of T1W-GD volumes: {observations[observations["T1W-GD"] != "---"].shape[0]}")
print(f"Number of T2W volumes: {observations[observations["T2W"] != "---"].shape[0]}")
print("")

print(f"Number of patients with T1W, T1W-GD and T2W: {observations[(observations["T1W"] != "---") & (observations["T1W-GD"] != "---") & (observations["T2W"] != "---")].shape}")
print(f"Number of patients with T1W and T1W-GD: {observations[(observations["T1W"] != "---") & (observations["T1W-GD"] != "---")].shape[0]}")
print(f"Number of patients with T1W and T2W: {observations[(observations["T1W"] != "---") & (observations["T2W"] != "---")].shape[0]}")
print(f"Number of patients with T1W-GD and T2W: {observations[(observations["T1W-GD"] != "---") & (observations["T2W"] != "---")].shape[0]}")
print("")


print("\nNumber of observations with 2, or 3 out of (T1, T1GD and T2) for each diagnose:\n")
print(observations.groupby("diagnosis")[["T1_and_T1GD", "T1GD_and_T2", "T1_and_T2", "all_three"]].sum().reset_index())
print("\n")



# %%
# SAVE FINAL OBSERVATION DATA
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/final_observations.pkl", "wb") as f:
    pickle.dump(observations, f)
print("Saved observation data to final_observations.pkl (supposedly)")
# %%

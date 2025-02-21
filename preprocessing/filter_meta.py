'''This script takes the preprocessed data, and does some filtering and provides a summary'''



# %%
# Import Dependencies and setup
###################################################
import pandas as pd
import pickle

from utils import *

# %%
# Load prepared meta data
###################################################
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/meta.pkl", "rb") as f:
    meta = pickle.load(f)



# %%
# Select relevant columns
###################################################
meta = meta[["subjetID", "survival", "session_name", "diagnosis", "session_status", "Notes", "Notes_simon", "tumor_descriptor", "age_at_diagnosis", "age_at_sample_acquisition", "tumor_location", "image_type", "magnification", "scanner", "file_name", "session_name.1"]]



# %%
# Perform groupings
###################################################

# A dataframe where each row represents a unique sequence that was produced during some session for a patient
meta_unique_seqs = meta.drop_duplicates(subset=["subjetID", "session_name", "image_type"])[["subjetID", "session_name", "image_type", "diagnosis", "file_name"]]

# Filter away some image types (sequence types), and then create dummies for the remaining sequences
seq_group = ["T1W", "T1W-GD", "T2W", "FLAIR"]
meta_unique_seqs = meta_unique_seqs[meta_unique_seqs["image_type"].isin(seq_group)]
meta_seq_dummies = pd.get_dummies(meta_unique_seqs, columns=["image_type"], prefix="Seq", prefix_sep="_")

# Remove parts of the file_name
meta_seq_dummies["file_name"] = meta_seq_dummies["file_name"].str.replace(r'(.*?/FILES/)', "", regex=True).str.replace(r'\.json', "", regex=True)

meta_seq_dummies

# %%

# A df with dummies indicating which sequences are available for each patient
df_dummies = meta_seq_dummies.groupby("subjetID", as_index=False).agg({
    "diagnosis": "first",
    **{col: 'max' for col in meta_seq_dummies.columns if col.startswith('Seq_')} # dictionary unpacking
})

df_dummies


# %%
b = df_dummies[df_dummies[["Seq_" + seq for seq in seq_group]].all(axis=1)]
print(b["subjetID"][0:20])

# %%
id = 123
meta_unique_seqs[meta_unique_seqs["subjetID"] == id][["subjetID", "session_name", "image_type", "file_name"]]

# %%


#df_dummies[(df_dummies["Seq_T1W"] == True) and (df_dummies["Seq_T1W_GD"] == True) and (df_dummies["Seq_T2W"] == True) and (df_dummies["Seq_FLAIR"] == True)]
group = ["T1W", "T1W-GD", "T2W", "FLAIR"]

b = df_dummies[df_dummies[["Seq_" + seq for seq in group]].all(axis=1)]
print(b["subjetID"][0:20])
# %%

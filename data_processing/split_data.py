'''
This script performs some kind of observation filtering, and then split the observations into
train, validation, and test sets, with stratification, and groupwise on patients to avoid data leakage.
'''


exit() #blocker to make sure this script is not accidentally run


# %%
# SETUP
import pickle
import os

os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

# Load observations
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/final_observations_singles.pkl", "rb") as f:
    observations = pickle.load(f)



#%%
# FILTER OBSERVATIONS

# We have kind of enough observations with either both T1GD and T2,    or  T1 and T2.
# So for now, let's just filter on obs with T1GD and T2
observations = observations[observations["T1GD_and_T2"].isin([True])]
print(f"Number of observations after filtering: {observations.shape[0]}")

# Effectively renaming the label column, because label is used below to refer to the stratification key
observations["class_label"] = observations["label"].copy()

# Add a row indexer column, called slide_id, as per get_repetition_split_v2
observations["slide_id"] = [i for i in range(observations.shape[0])]

# Add a case_id column that is just a renaming of the subjet_ID column, as per get_repetition_split_v2
observations["case_id"] = observations["subjetID"].copy()

# Create a label column, which is the stratification key
observations["file_pattern"] = (
    (~observations["T1W"].isin(["---"])).astype(int).astype(str) +
    (~observations["T1W-GD"].isin(["---"])).astype(int).astype(str) +
    (~observations["T2W"].isin(["---"])).astype(int).astype(str)
)
# # Attach diagnosis to the stratification/label key
observations["label"] = observations["diagnosis"] + "_" + observations["file_pattern"]

# Perform train/val/test split. out is a df with a column "fold1" indicating the sets "train"/"validation"/"test".
cfg = {"class_stratification": True, "test_fraction": 0.2, "validation_fraction": 0.16, "number_of_folds": 1}
out = get_repetition_split_v2(cfg=cfg, df=observations, print_summary=False)

# Create separate dataframes
train_df = out[out["fold_1"] == "train"].copy()
valid_df = out[out["fold_1"] == "validation"].copy()
test_df = out[out["fold_1"] == "test"].copy()



#%%
# CHECK DATA LEAKAGE
check_data_leakage(train_df=train_df,
                   test_df=test_df,
                   valid_df=valid_df)



# %%
# CHECK OBSERVATION SUMMARY
observation_summary(train_df, "Train data")
observation_summary(valid_df, "Validation data")
observation_summary(test_df, "Test data")




#%%
# SAVE DATA SPLITS

# Load old data that turned out to have patient leakage
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df.pkl", "wb") as f:
    pickle.dump(train_df, f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/valid_df.pkl", "wb") as f:
    pickle.dump(valid_df, f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.pkl", "wb") as f:
    pickle.dump(test_df, f)

# %%

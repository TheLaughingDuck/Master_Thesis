'''
This script produces nice little tables visualizing the data for presentations and the report.
'''

#%%
# SETUP
import pickle
import tabulate
import pandas as pd

# Load data
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df.pkl", "rb") as f:
    train_df = pickle.load(f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/valid_df.pkl", "rb") as f:
    valid_df = pickle.load(f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.pkl", "rb") as f:
    test_df = pickle.load(f)


#%%
# Concat train and val
train_val_df = pd.concat([train_df, valid_df])
print(f"Train obs: {train_df.shape[0]}, Validation obs: {valid_df.shape[0]}, combined: {train_val_df.shape[0]}") # sanity check

# Map the classes to appropriate names
class_names = {0: "Low/High-grade Glioma", 1: "Ependymoma", 2: "Medulloblastoma"}
train_val_df["diagnosis"] = [class_names[i] for i in train_val_df["class_label"]]
test_df["diagnosis"] = [class_names[i] for i in test_df["class_label"]]


#%%
# MAKE TRAIN+VAL DATA TABLE
table = tabulate.tabulate(train_val_df.groupby("diagnosis")[["T1GD_and_T2", "T1_and_T2"]].sum().reset_index(),
                          tablefmt="latex",
                          showindex=False,
                          headers=["Class", "T1GD and T2", "T1 and T2"])
print(table)


# %%
# MAKE TEST DATA TABLE
table = tabulate.tabulate(test_df.groupby("diagnosis")[["T1GD_and_T2", "T1_and_T2"]].sum().reset_index(),
                          tablefmt="latex",
                          showindex=False,
                          headers=["Class", "T1GD and T2", "T1 and T2"])
print(table)
# %%

'''
In the report, I want to have tables that show the number of observations
per diagnose and in the train, val and test datasets.

This script creates those tables.
'''

#%%
# SETUP
import torch
import pandas as pd
import numpy as np
import pickle
from tabulate import tabulate

tablefmt = "latex"
#%%

# LOAD DATA
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df.pkl", "rb") as f:
    train_df = pickle.load(f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/valid_df.pkl", "rb") as f:
    valid_df = pickle.load(f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.pkl", "rb") as f:
    test_df = pickle.load(f)

# Combine the tables and make a copy
df_combined_diag = pd.concat([train_df, valid_df, test_df])
diag_data = df_combined_diag.copy()

# Convert suitable for the table below
converter = {0: "Glioma", 1: "Ependymoma", 2: "Medulloblastoma"}
diag_data["class_label"] = [converter[label] for label in diag_data["class_label"]]
diag_data = diag_data[["subjetID", "class_label", "fold_1"]].groupby(by=["class_label", "fold_1"]).count().to_dict()["subjetID"]

#%%
# CREATE FINAL DIAGNOSE TABLE

#print(tabulate({"A": [1,2,3], "B": [5,6,7]}, tablefmt="grid"))

tab = tabulate([["Glioma", diag_data[("Glioma", "train")], diag_data[("Glioma", "validation")], diag_data[("Glioma", "test")]],
                ["Ependymoma", diag_data[("Ependymoma", "train")], diag_data[("Ependymoma", "validation")], diag_data[("Ependymoma", "test")]],
                ["Medulloblastoma", diag_data[("Medulloblastoma", "train")], diag_data[("Medulloblastoma", "validation")], diag_data[("Medulloblastoma", "test")]]],
                
                headers=["Diagnose", "Training", "Validation", "Test"], tablefmt=tablefmt); print(tab)


# %%
# CREATE FINAL LOCATION TABLE

# Load data
train_df_loc = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/train_df_loc.csv")
valid_df_loc = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/valid_df_loc.csv")
test_df_loc = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/location_classification/data/test_df_loc.csv")


# Combine the tables and make a copy
df_combined_loc = pd.concat([train_df_loc, valid_df_loc, test_df_loc])
loc_data = df_combined_loc.copy()

# Convert suitable for the table below
converter = {0: "Supra", 1: "Infra"}
loc_data["loc_label"] = [converter[label] for label in loc_data["loc_label"]]
loc_data = loc_data[["subjetID", "loc_label", "fold_1"]].groupby(by=["loc_label", "fold_1"]).count().to_dict()["subjetID"]

# CREATE FINAL DIAGNOSE TABLE

tab = tabulate([["Infra", loc_data[("Infra", "train")], loc_data[("Infra", "validation")], loc_data[("Infra", "test")]],
                ["Supra", loc_data[("Supra", "train")], loc_data[("Supra", "validation")], loc_data[("Supra", "test")]]],
                
                headers=["Location", "Training", "Validation", "Test"], tablefmt=tablefmt); print(tab)


# %%

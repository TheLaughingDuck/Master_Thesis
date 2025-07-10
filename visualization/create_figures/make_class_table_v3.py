'''
I want a table that shows the number of patients and sessions
for each diagnosis and location combination.

This script does that. Quite well.
'''


#%%
#SETUP


import pandas as pd
import numpy as np

from tabulate import tabulate


train_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/data/train_df_loc_with_mixed.csv")
valid_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/data/valid_df_loc_with_mixed.csv")
test_df = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/data/test_df_loc_with_mixed.csv")

#%%

data = pd.concat([train_df, valid_df, test_df], axis=0)

converter = {"High-Grade Glioma": "Glioma", "Low-Grade Glioma": "Glioma", "Ependymoma":"Ependymoma", "Medulloblastoma": "Medulloblastoma"}
data["diagnosis"] = data["diagnosis"].replace(converter)

#%%
data.groupby(by=["diagnosis"]).agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    All_three=("all_three", "count")
    
).sort_values(by="Unique_patients", ascending=False) .reset_index()

#%%
data.groupby(by="tumor_location").agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    All_three=("all_three", "count")
    
).sort_values(by="Unique_patients", ascending=False) .reset_index()

# %%
data_infra_supra = data[data["tumor_location"].isin(["Supra", "Infra"])].copy()

data_infra_supra = data_infra_supra.groupby(by=["diagnosis", "tumor_location"]).agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    All_three=("all_three", "count")
    
).unstack(fill_value=0).stack().sort_values(by="diagnosis", ascending=False).reset_index()

print(tabulate(data_infra_supra, headers=["Diagnosis", "Location", "Patients", "Observation"], tablefmt="latex", showindex=False))

# %%

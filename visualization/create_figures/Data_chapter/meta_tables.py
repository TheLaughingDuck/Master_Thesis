'''
I wanted a histogram of the patient ages in the report.

This script creates that figure.
'''

#%%
# SETUP
import torch
import pandas as pd
import numpy as np
import pickle


# LOAD DATA
with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/train_df.pkl", "rb") as f:
    train_df = pickle.load(f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/valid_df.pkl", "rb") as f:
    valid_df = pickle.load(f)

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.pkl", "rb") as f:
    test_df = pickle.load(f)

# Load raw excel data
data_raw = pd.read_excel("/local/data1/simjo484/mt_data/all_data/MRI/MRI_summary_extended_simon.xlsx")

data = pd.concat([train_df, valid_df, test_df])

data["age_first_scan"] = [None for i in range(data.shape[0])]

subj_survival = data_raw.groupby("subjetID", as_index=False)["survival"].min()
patient_to_age_converter = {int(id): age for (idx, (id, age)) in subj_survival.iterrows()}

data["age_first_scan"] = [patient_to_age_converter[i] for i in data["subjetID"]]

#%%
import matplotlib.pyplot as plt


#subj_survival = data.groupby("subjetID", as_index=False)["survival"].min()


plt.hist(data["age_first_scan"]/365)
plt.suptitle("Patient age at first MRI scan", fontsize=20)
plt.xlabel("Age (in years)", fontsize=16)
plt.ylabel("Number of patients", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/patient_age_first_mri.png")

#%%
#pd.qcut(subj_survival["survival"], q=[0, 0.95])
pd.qcut(subj_survival["survival"]/365, q=[0, 0.95, 1])
#pd.qcut([1,2,3,4,5], q=[0, 0.5, 1])
# %%

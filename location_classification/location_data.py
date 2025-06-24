'''
This script loads the meta data (train, val, test as well as original excel file), and then
creates new .csv files that also contain tumor location.

Purpose of this script was to figure out the tumor locations. Turns out there are many many locations.
'''

#%%
# SETUP
import pandas as pd
from itertools import islice
import pickle
import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import unique

# LOAD DATA
data_raw = pd.read_excel("/local/data1/simjo484/mt_data/all_data/MRI/MRI_summary_extended_simon.xlsx")

meta_root = "/local/data1/simjo484/mt_data/all_data/MRI/simon/"
with open(meta_root+"train_df.pkl", "rb") as f:
    train_df = pickle.load(f)

with open(meta_root+"valid_df.pkl", "rb") as f:
    valid_df = pickle.load(f)

with open(meta_root+"test_df.pkl", "rb") as f:
    test_df = pickle.load(f)


#%%
# CREATE FUNCTION THAT ANNOTATES the data with location (infra, supra, mixed, remove)

supra_locations = [
    "Frontal Lobe",
    "Optic Pathway,Suprasellar/Hypothalamic/Pituitary",
    "Thalamus",
    "Temporal Lobe",
    "Frontal Lobe,Parietal Lobe",
    "Thalamus,Ventricles",
    "Occipital Lobe",
    "Occipital Lobe,Temporal Lobe",
    "Hippocampus",
    "Suprasellar/Hypothalamic/Pituitary",
    "Optic Pathway,Suprasellar/Hypothalamic/Pituitary,Thalamus",
    "Optic Pathway,Other locations NOS,Suprasellar/Hypothalamic/Pituitary,Thalamus",
    "Suprasellar/Hypothalamic/Pituitary,Thalamus",
    "Parietal Lobe",
    "Basal Ganglia,Thalamus",
    "Optic Pathway,Suprasellar/Hypothalamic/Pituitary,Ventricles",
    "Basal Ganglia,Other locations NOS,Temporal Lobe,Thalamus",
    "Parietal Lobe,Temporal Lobe",
    "Other locations NOS,Pineal Gland,Thalamus",
    "Temporal Lobe,Thalamus",
    "Other locations NOS,Suprasellar/Hypothalamic/Pituitary,Ventricles",
    "Basal Ganglia,Suprasellar/Hypothalamic/Pituitary,Thalamus",
    "Frontal Lobe,Temporal Lobe",
    "Frontal Lobe,Parietal Lobe,Temporal Lobe"]

infra_locations = [
    "Cerebellum/Posterior Fossa",
    "Cerebellum/Posterior Fossa,Meninges/Dura,Spinal Cord- Cervical,Spinal Cord- Thoracic,Ventricles",
    "Cerebellum/Posterior Fossa,Optic Pathway,Suprasellar/Hypothalamic/Pituitary,Thalamus",
    "Brain Stem-Medulla,Brain Stem- Midbrain/Tectum,Cerebellum/Posterior Fossa",
    "Cerebellum/Posterior Fossa,Meninges/Dura",
    "Brain Stem-Medulla,Cerebellum/Posterior Fossa,Ventricles",
    "Cerebellum/Posterior Fossa,Optic Pathway",
    "Cerebellum/Posterior Fossa,Ventricles",
    "Brain Stem- Midbrain/Tectum,Cerebellum/Posterior Fossa,Thalamus",
    "Cerebellum/Posterior Fossa,Other locations NOS",
    "Brain Stem- Pons,Cerebellum/Posterior Fossa",
    "Cerebellum/Posterior Fossa,Meninges/Dura,Optic Pathway,Other locations NOS,Suprasellar/Hypothalamic/Pituitary,Ventricles",
    "Cerebellum/Posterior Fossa,Meninges/Dura,Spinal Cord- Cervical,Spinal Cord- Lumbar/Thecal Sac,Spinal Cord- Thoracic",
    "Brain Stem- Midbrain/Tectum",
    "Brain Stem-Medulla,Brain Stem- Pons",
    "Brain Stem- Midbrain/Tectum,Thalamus",
    "Brain Stem- Pons"]

mixed_locations = [
    "Cerebellum/Posterior Fossa, Frontal Lobe",
    "Cerebellum/Posterior Fossa,Frontal Lobe",
    "Basal Ganglia,Cerebellum/Posterior Fossa,Occipital Lobe,Other locations NOS,Parietal Lobe,Temporal Lobe,Thalamus",
    "Brain Stem- Midbrain/Tectum,Temporal Lobe,Thalamus",
    "Meninges/Dura,Spinal Cord- Lumbar/Thecal Sac,Suprasellar/Hypothalamic/Pituitary",
    "Basal Ganglia,Frontal Lobe,Meninges/Dura,Other locations NOS,Spinal Cord- Lumbar/Thecal Sac,Suprasellar/Hypothalamic/Pituitary",
    "Basal Ganglia,Brain Stem- Midbrain/Tectum,Thalamus,Ventricles",
    "Parietal Lobe,Spinal Cord- Lumbar/Thecal Sac,Temporal Lobe,Thalamus",
    "Spinal Cord- Cervical,Spinal Cord- Thoracic,Temporal Lobe",
    "Brain Stem- Midbrain/Tectum,Occipital Lobe,Temporal Lobe,Thalamus"
]

def attach_tumour_loc_class(df, data_raw):
    '''
    Takes a df (either the train, valid, or test dataframe), and attaches tumour location to it, using the raw data.
    This function also now filters on only Supra and Infra (because the Mixed were so few), and makes a location label.
    '''
    
    data_raw_location = data_raw[["subjetID", "tumor_location"]].drop_duplicates(subset="subjetID", keep="first")
    df = pd.merge(df, data_raw_location, how="left", left_on="subjetID", right_on="subjetID")

    for i in range(df.shape[0]):
        # Rename Supra
        if df.loc[i, "tumor_location"] in supra_locations:
            df.loc[i, "tumor_location"] = "Supra"

        # Rename Infra
        elif df.loc[i, "tumor_location"] in infra_locations:
            df.loc[i, "tumor_location"] = "Infra"

        # Rename Mixed
        elif df.loc[i, "tumor_location"] in mixed_locations:
            df.loc[i, "tumor_location"] = "Mixed"

        else:
            df.loc[i, "tumor_location"] = "Remove"
    
    # Filter on Supra and Infra
    df = df[df["tumor_location"].isin(["Supra", "Infra", "Mixed", "Remove"])].copy()


    # Create location label
    converter = {"Supra": 0, "Infra": 1, "Mixed": 2, "Remove": 3}
    df["loc_label"] = [converter[i] for i in df["tumor_location"]]#df["tumor_location"]

    return df

#%%
train_df_loc = attach_tumour_loc_class(train_df, data_raw)
valid_df_loc = attach_tumour_loc_class(valid_df, data_raw)
test_df_loc = attach_tumour_loc_class(test_df, data_raw)

#os.chdir("/home/simjo484/master_thesis/Master_Thesis/tentorial_classification")
os.chdir("/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/")
train_df_loc.to_csv("data/train_df_loc_with_mixed.csv")
valid_df_loc.to_csv("data/valid_df_loc_with_mixed.csv")
test_df_loc.to_csv("data/test_df_loc_with_mixed.csv")







# # %%
# # Find all uniqe locations
# # locs = unique(data_raw["tumor_location"])["Values"]
# # for i in locs:
# #     print(i)
# #%%


# # s = 0
# # for location in locs:
# #     if "t" in location:
# #         print(location)
# #         s += 1
# # print(f"Total: {s}")

# # # %%
# # unique(data_raw["tumor_location"])[0:10]


# # %%
# # PRINT the tumor location for each patient in the training and validation data.
# df_combined = pd.concat([train_df, valid_df, test_df])
# subjects = df_combined["subjetID"].tolist()
# #train_subjects = train_df["subjetID"].tolist()

# unique_locations = []

# for subj in subjects:
#     df = data_raw[data_raw["subjetID"] == subj]
#     locs = df.drop_duplicates(subset=["tumor_location"])["tumor_location"].tolist()

#     # Cycle through the locations for one patient (often there are multiple)
#     for l in locs:
#         if l not in unique_locations: unique_locations.append(l)

# print(unique_locations)

# os.chdir("/home/simjo484/master_thesis/Master_Thesis/tentorial_classification")
# with open("locations.txt", "w") as f:
#     for l in unique_locations:
#         f.write(l)
#         f.write("\n")





# #%%
# print(data_raw.drop_duplicates(subset=["subjetID"]).shape)
# print(data_raw.drop_duplicates(subset=["subjetID", "tumor_location"]).shape)
# # %%

# %%

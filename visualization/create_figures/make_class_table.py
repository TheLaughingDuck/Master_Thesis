'''
I want a table with the counts of patients and sessions,
for diagnose and for location.

This script does that.
'''
#%%
# SETUP
import pickle
import pandas

with open("/local/data1/simjo484/mt_data/all_data/MRI/simon/meta.pkl", "rb") as f:
    data = pickle.load(f)


# PIVOT TO GET DATAFRAME WITH OBSERVATIONS

# Drop duplicates
drop_duplicates = True
# See top of this file (META PROCESSING SETTINGS)
if drop_duplicates:
    observations = data.drop_duplicates(subset=["subjetID", "session_name", "seq_type"])
    observations = observations.pivot(index=["subjetID", "session_name", "diagnosis", "tumor_location", "gender"], 
                            values="found_filename",
                            columns="seq_type")
    observations = observations.reset_index()
    observations = observations.fillna("---")
    print(f"Observations shape after pivoting (and dropping duplicates): {observations.shape}")
else:
    data["type_counter"] = data.groupby(["subjetID", "session_name", "seq_type", "diagnosis"]).cumcount()+1

    observations = data.pivot(index=["subjetID", "session_name", "diagnosis", "type_counter"], 
                            values="found_filename",
                            columns="seq_type")
    observations = observations.reset_index()
    observations = observations.fillna("---")
    print(f"Observations shape after pivoting (and *not* dropping duplicates): {observations.shape}")

data = observations.copy()

# %%
# CREATE BINARY COLUMNS FOR SEQ TYPE PAIRS
data["T1_and_T1GD"] = (data["T1W"] != "---") & (data['T1W-GD'] != "---")
data["T1GD_and_T2"] = (data['T1W-GD'] != "---") & (data["T2W"] != "---")
data["T1_and_T2"] = (data["T1W"] != "---") & (data["T2W"] != "---")
data["all_three"] = (data["T1W"] != "---") & (data['T1W-GD'] != "---") & (data["T2W"] != "---")

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

#%%
data = data.reset_index()

for i in range(data.shape[0]):
    # Rename Supra
    if data.loc[i, "tumor_location"] in supra_locations:
        data.loc[i, "tumor_location"] = "Supra"

    # Rename Infra
    elif data.loc[i, "tumor_location"] in infra_locations:
        data.loc[i, "tumor_location"] = "Infra"

    # Rename Mixed
    elif data.loc[i, "tumor_location"] in mixed_locations:
        data.loc[i, "tumor_location"] = "Mixed"

    else:
        data.loc[i, "tumor_location"] = "Remove"


# Filter out mixed and others
data = data[data["tumor_location"].isin(["Supra", "Infra"])].copy()

# Process gender
converter = {"Male": 0, "Female": 1}
data["gender"] = [converter[i] for i in data["gender"]]

data = data[data["T1GD_and_T2"] == True]

data

# %%
data.groupby(by=["diagnosis"]).agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    Prop_Female=("gender", "mean"),
    All_three=("all_three", "count")
    
).sort_values(by="Unique_patients", ascending=False) .reset_index()

# %%
data.groupby(by="tumor_location").agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    Prop_Female=("gender", "mean"),
    All_three=("all_three", "count")
    
).sort_values(by="Unique_patients", ascending=False) .reset_index()

# %%

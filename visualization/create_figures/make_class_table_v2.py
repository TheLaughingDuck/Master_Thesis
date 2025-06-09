
#%%
# SETUP
import pandas as pd

observations = pd.read_csv("/home/simjo484/master_thesis/Master_Thesis/visualization/observations.csv")


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
observations = observations.reset_index()

for i in range(observations.shape[0]):
    # Rename Supra
    if observations.loc[i, "tumor_location"] in supra_locations:
        observations.loc[i, "tumor_location"] = "Supra"

    # Rename Infra
    elif observations.loc[i, "tumor_location"] in infra_locations:
        observations.loc[i, "tumor_location"] = "Infra"

    # Rename Mixed
    elif observations.loc[i, "tumor_location"] in mixed_locations:
        observations.loc[i, "tumor_location"] = "Mixed"

    else:
        observations.loc[i, "tumor_location"] = "Remove"


# Filter out mixed and others
observations = observations[observations["tumor_location"].isin(["Supra", "Infra"])].copy()

# Process gender
converter = {"Male": 0, "Female": 1}
observations["gender"] = [converter[i] for i in observations["gender"]]




observations
# %%
observations.groupby(by=["diagnosis"]).agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    Prop_Female=("gender", "mean"),
    All_three=("all_three", "count")
    
).sort_values(by="Unique_patients", ascending=False) .reset_index()

# %%
observations.groupby(by="tumor_location").agg(
    Unique_patients=("subjetID", "nunique"),
    #Unique_sessions=("session_name", "nunique"),
    Prop_Female=("gender", "mean"),
    All_three=("all_three", "count")
    
).sort_values(by="Unique_patients", ascending=False) .reset_index()
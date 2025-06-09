# %%
# Import Dependencies and setup
###################################################
import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis") # set working directory, useful for running .py files as notebooks.

import utils
import pickle

# Open the dataset with extracted meta data
with open("/local/data1/simjo484/mt_data/all_data/MRI/meta.pkl", "rb") as f:
    meta = pickle.load(f)

# Select the useful columns
meta = meta[["subjetID", "survival", "session_name", "session_status", "diagnosis", "Notes", "Notes_simon", "tumor_descriptor", "tumor_location", "image_type", "magnification", "scanner", "file_name"]]


# %%
# How many unique patients are there?
###################################################
print("There are",
      meta.drop_duplicates(subset=["subjetID"]).shape[0],
      "patients.") # 679 patients



# %%
# How many unique patients, per diagnose?
###################################################
diagnoses = utils.unique(meta["diagnosis"])["Values"].tolist() # Get the unique diagnoses
unique_patients = meta.drop_duplicates(subset=["subjetID"])

# Print information for each diagnose
for diag in diagnoses:    
    numb_patients = unique_patients[unique_patients["diagnosis"] == diag].shape[0]
    print(diag, ":", numb_patients, "patients.")



# %%
# How many tuples of T1W, T2W and ADC for each diagnose?
# There may be multiple tuples for one patient
###################################################

for diag in diagnoses:
    this = meta[meta["diagnosis"] == diag]

    #print(this.shape)

#meta[meta["diagnosis"] == "Low-Grade Glioma"]
#utils.unique(meta["survival"])



# %%
# What are the dimensions of the images?
###################################################
# Get file names
#file_names = os.listdir("/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/")
core_filename = meta["file_name"].str.replace(r'(.*?/FILES/)', "", regex=True).str.replace(r'\.json', "", regex=True)
str_id = [str(i) for i in meta["subjetID"]]
gz_filenames = ["PP_C" + a + "___" + b + "___" + c + ".nii.gz" for (a,b,c) in zip(str_id, meta["session_name"], core_filename)]

import nibabel as nib

shapes = []
for name in gz_filenames[0:10]:
    try:
        img = nib.load("/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/" + name).get_fdata()
        shapes.append(str(img.shape))
        #print("Found:", name)
    except:
        pass
        #print("Could not find:", name)

utils.unique(shapes)



# %%
# If we only use T1W, T2W and ADC, how many patients have all three?
#utils.unique(meta["image_type"])

meta[[s in ["T1W", "T2W", "ADC"] for s in meta["image_type"]]].drop_duplicates(subset=["subjetID", "session_name", "image_type"])


# %%


meta.iloc[0]

# %%
# OLD CODE
###################################################
#utils.show_image(z=70, full_path="/local/data1/simjo484/mt_data/all_data/MRI/pre_processed/Final preprocessed files/PP_C73431___6365d_B_brain_10h47m___t2_ci3d_tra_iso-ciss.nii.gz")

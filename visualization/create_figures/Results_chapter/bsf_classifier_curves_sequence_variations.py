'''
In the report, I want some figures that show the training metrics (loss)
for some of the standardized training runs of the BSF classifier fine-tuning.

This script makes those figures.
'''

#%%
# SETUP
import torch
import matplotlib.pyplot as plt
import json


# Set up parameters
file_paths = ["/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-16:57:17 (T2 standardized training)/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-20:03:43 (T1GD standardized training)/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (with 768-10-3) architecture/2025-04-09-23:09:52 (T1GD and T2 standardized training)/TrainingTracker_metrics.json"]
titles = ["BSF on T2W", "BSF on T1W-GD", "BSF on fused T1W-GD and T2W"]
fig_names = ["T2_standardized_training_classifier_768_10_3.png", "T1GD_standardized_training_classifier_768_10_3.png", "T1GD_T2_standardized_training_classifier_768_10_3.png"]

# Create loss curve for each separate training
for path, tit, name in zip(file_paths, titles, fig_names):
    with open(path, 'r') as file:
        data = json.load(file)
    
    print(data)

    # Construct loss curves
    plt.plot(data["avg_train_loss"]["step"], data["avg_train_loss"]["value"], label="Training")
    plt.plot(data["avg_valid_loss"]["step"], data["avg_valid_loss"]["value"], label="Validation")
    plt.ylabel("Cross Entropy Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.legend()
    plt.suptitle(tit, fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name)
    plt.show()


# %%

'''
I ran a "1d grid search" of the number blocks to be frozen in the feature extractor (fused t1gd and t2 images).
This script creates training and validation loss curve figures for these trainings.

To be clear, this was on the diagnose classification task.
'''

#%%
# SETUP
import torch
import matplotlib.pyplot as plt
import json


# Set up parameters
file_paths = ["/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/2025-04-10-15:01:04 (T1GD and T2 standardized training (freeze 0))/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/2025-04-10-17:57:01 (T1GD and T2 standardized training (freeze 1))/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/2025-04-10-20:57:54 (T1GD and T2 standardized training (freeze 2))/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/2025-04-11-00:05:21 (T1GD and T2 standardized training (freeze 3))/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/2025-04-11-03:49:14 (T1GD and T2 standardized training (freeze 4))/TrainingTracker_metrics.json",
              "/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/2025-04-11-07:33:32 (T1GD and T2 standardized training (freeze 5))/TrainingTracker_metrics.json"]
titles = ["BSF with 0 frozen blocks", "BSF with 1 frozen block", "BSF with 2 frozen blocks", "BSF with 3 frozen blocks", "BSF with 4 frozen blocks", "BSF with 5 frozen blocks"]
fig_names = ["standardized_training_freeze_0.png", "standardized_training_freeze_1.png", "standardized_training_freeze_2.png", "standardized_training_freeze_3.png", "standardized_training_freeze_4.png", "standardized_training_freeze_5.png"]

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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(prop={'size': 16})
    plt.tight_layout()
    plt.suptitle(tit, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name)
    plt.show()


# %%

#%%
# SETUP
import torch
import matplotlib.pyplot as plt
import json

fig_path = "/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"

def make_loss_plot(data_file_path, title, fig_name):
    with open(data_file_path, 'r') as file:
        data = json.load(file)

    plt.plot(data["train_loss"]["step"], data["train_loss"]["value"], label="Training")
    plt.plot(data["valid_loss"]["step"], data["valid_loss"]["value"], label="Validation")
    plt.ylabel("Dice loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.suptitle(title, fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    plt.savefig(fig_path+fig_name)
    plt.close()

#%%
# BSF T1W-GD
make_loss_plot(data_file_path = "/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd/2025-04-07-10:40:33/TrainingTracker_metrics.json",
                title="BSF segmentation fine-tuning on T1W-GD",
                fig_name="bsf_t1gd.png")

# BSF T2W
make_loss_plot(data_file_path = "/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-27-13:20:53 (t2)/TrainingTracker_metrics.json",
                title="BSF segmentation fine-tuning on T2W",
                fig_name="bsf_t2.png")

# BSF T1W-GD and T2W
make_loss_plot(data_file_path = "/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/TrainingTracker_metrics.json",
                title="BSF segmentation fine-tuning on T1W-GD and T2W",
                fig_name="bsf_t1gd_t2.png")



# #%%
# # BSF T2
# file_path = "/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-27-13:20:53 (t2)/TrainingTracker_metrics.json"
# with open(file_path, 'r') as file:
#     data = json.load(file)

# print(data)

# plt.plot(data["train_loss"]["step"], data["train_loss"]["value"], label="Training")
# plt.plot(data["valid_loss"]["step"], data["valid_loss"]["value"], label="Validation")
# plt.ylabel("Dice loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.savefig(fig_path+"bsf_t2.png")



# #%%
# # BSF T1GD and T2
# file_path = "/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/TrainingTracker_metrics.json"
# with open(file_path, 'r') as file:
#     data = json.load(file)

# print(data)

# plt.plot(data["train_loss"]["step"], data["train_loss"]["value"], label="Training")
# plt.plot(data["valid_loss"]["step"], data["valid_loss"]["value"], label="Validation")
# plt.ylabel("Dice loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.savefig(fig_path+"bsf_t1gd_t2.png")

# #%%
# # BSF T1GD
# file_path = "/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd/2025-04-07-10:40:33/TrainingTracker_metrics.json"
# with open(file_path, 'r') as file:
#     data = json.load(file)

# print(data)

# plt.plot(data["train_loss"]["step"], data["train_loss"]["value"], label="Training")
# plt.plot(data["valid_loss"]["step"], data["valid_loss"]["value"], label="Validation")
# plt.ylabel("Dice loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.savefig(fig_path+"bsf_t1gd.png")

# %%

'''
I ran some training of Iulians script. This script will create figures with the training and validation loss.
'''


#%%
# SETUP
from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd



# CREATE LOSS CURVES
def make_loss_curves(logfiles, titles, file_names):
    '''
    
    '''
    all_data = {}

    # Cycle through each individual training run
    for id, paths, tit, name in zip(range(3), logfiles, titles, file_names):

        data = {"training": {}, "validation": {}}

        # Load first train, and then val event data
        for type_of_loss, event_file_path in zip(["training", "validation"], paths):
            
            # Cycle through values in one eventfile
            for summary in summary_iterator(event_file_path):
                for value in summary.summary.value:
                    if value.tag not in data[type_of_loss]:
                        data[type_of_loss][value.tag] = {'step': [], 'value': []}
                    data[type_of_loss][value.tag]['step'].append(float(summary.step))
                    data[type_of_loss][value.tag]['value'].append(float(value.simple_value))
        
        all_data[id] = data
    
    # Create figures
    for id, tit, name in zip(all_data.keys(), titles, file_names):
        # Create one loss curve
        plt.plot(all_data[id]["training"]["loss"]["step"], all_data[id]["training"]["loss"]["value"], color="blue", label="Training")
        plt.plot(all_data[id]["validation"]["loss"]["step"], all_data[id]["validation"]["loss"]["value"], color="orange", label="Validation")
        plt.legend(prop={'size': 16})
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Cross entropy loss", fontsize=16)
        plt.suptitle(tit, fontsize=20)
        plt.grid()
        plt.tight_layout()
        plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_loss"+".png")
        plt.show()

        # Create one accuracy curve
        # plt.plot(all_data[id]["training"]["accuracy"]["step"], all_data[id]["training"]["accuracy"]["value"], color="blue", label="Training")
        # plt.plot(all_data[id]["validation"]["accuracy"]["step"], all_data[id]["validation"]["accuracy"]["value"], color="orange", label="Validation")
        # plt.legend()
        # plt.xticks(fontsize=14)
        # plt.xticks(fontsize=14)
        # plt.xlabel("Epoch", fontsize=16)
        # plt.ylabel("Accuracy", fontsize=16)
        # plt.suptitle(tit+" (acc)", fontsize=20)
        # plt.tight_layout()
        # plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_acc"+".png")
        # plt.show()



import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import get_conf_matrix, create_conf_matrix_fig


# CREATE CONFUSION MATRICES
def make_matrices(pred_paths, titles, file_names, n_classes):
    '''
    Creates a series of confusion matrices.

    Takes a list of paths to linear probing result csv files, a list of figure titles, and a list of filenames to save the figures as.
    '''

    # Load the targets
    if n_classes == 2: # Location classification
        valid_df = pd.read_csv("/local/data1/simjo484/mt_data/all_data/MRI/simon/valid_df_loc.csv")
        targs = valid_df["loc_label"].tolist()
    elif n_classes == 3: # Diagnose classification
        valid_df = pd.read_pickle("/local/data1/simjo484/mt_data/all_data/MRI/simon/valid_df.pkl")
        targs = valid_df["class_label"].tolist()
    else:
        raise ValueError("Unsupported number of classes.")
    
    for path, tit, name in zip(pred_paths, titles, file_names):
        pred_logits = pd.read_csv(path)
        preds = [int(i) for i in pred_logits.idxmax(axis=1).tolist()]
        m = get_conf_matrix(preds, targs, n_classes=n_classes)
        m = np.array(m)
        
        
        subtitle = f"(Accuracy: {round(sum(np.diag(m)) / sum(m.flatten()), 2)})"

        name = "/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_val_conf_matrix"+".png"
        create_conf_matrix_fig(m, n_classes=n_classes, title=tit, save_fig_as=name, subtitle=subtitle)


# CREATE LOSS CURVES

# logfiles = [
#     ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (loc)/logs/train/20250411-114138/events.out.tfevents.1744364498.kawasaki.ad.liu.se.1147480.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (loc)/logs/validate/20250411-114138/events.out.tfevents.1744364498.kawasaki.ad.liu.se.1147480.1"],
#     ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-152650 (3dconv) (loc)/logs/train/20250409-152650/events.out.tfevents.1744205210.kawasaki.ad.liu.se.610681.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-152650 (3dconv) (loc)/logs/validate/20250409-152650/events.out.tfevents.1744205210.kawasaki.ad.liu.se.610681.1"],
#     ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (loc)/logs/train/20250409-133111/events.out.tfevents.1744198271.kawasaki.ad.liu.se.441010.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (loc)/logs/validate/20250409-133111/events.out.tfevents.1744198271.kawasaki.ad.liu.se.441010.1"]
#     ]
# titles = ["resnet2p1", "3dconv", "resnet_mixed_conv"]
# file_names = ["resnet2p1", "3dconv", "resnet_mixed_conv"]



# # Cycle through each individual training run
# for id, paths, tit, name in zip(range(3), logfiles, titles, file_names):

#     data = {"training": {}, "validation": {}}

#     # Load first train, and then val event data
#     for type_of_loss, event_file_path in zip(["training", "validation"], paths):
        
#         # Cycle through values in one eventfile
#         for summary in summary_iterator(event_file_path):
#             for value in summary.summary.value:
#                 if value.tag not in data[type_of_loss]:
#                     data[type_of_loss][value.tag] = {'step': [], 'value': []}
#                 data[type_of_loss][value.tag]['step'].append(float(summary.step))
#                 data[type_of_loss][value.tag]['value'].append(float(value.simple_value))
    
#     all_data[id] = data


# # Create figures
# for id, tit, name in zip(all_data.keys(), titles, file_names):
#     # Create one loss curve
#     plt.plot(all_data[id]["training"]["loss"]["step"], all_data[id]["training"]["loss"]["value"], color="blue", label="Training")
#     plt.plot(all_data[id]["validation"]["loss"]["step"], all_data[id]["validation"]["loss"]["value"], color="orange", label="Validation")
#     plt.legend()
#     plt.suptitle(tit+" (loss)")
#     plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_loss"+".png")
#     plt.show()

#     # Create one accuracy curve
#     plt.plot(all_data[id]["training"]["accuracy"]["step"], all_data[id]["training"]["accuracy"]["value"], color="blue", label="Training")
#     plt.plot(all_data[id]["validation"]["accuracy"]["step"], all_data[id]["validation"]["accuracy"]["value"], color="orange", label="Validation")
#     plt.legend()
#     plt.suptitle(tit+" (acc)")
#     plt.savefig("/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_acc"+".png")
#     plt.show()

#%%

### MAKE ResNet 2p1 (location) LOSS CURVES AND CONFUSION MATRICES (for t1gd, t2 and both)
make_loss_curves(logfiles=
                 [["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-124822 (resnet2p1) (t1gd) (loc)/logs/train/20250414-124822/events.out.tfevents.1744627702.kawasaki.ad.liu.se.2018017.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-124822 (resnet2p1) (t1gd) (loc)/logs/validate/20250414-124822/events.out.tfevents.1744627702.kawasaki.ad.liu.se.2018017.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-141009 (resnet2p1) (t2) (loc)/logs/train/20250414-141009/events.out.tfevents.1744632609.kawasaki.ad.liu.se.2087818.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-141009 (resnet2p1) (t2) (loc)/logs/validate/20250414-141009/events.out.tfevents.1744632609.kawasaki.ad.liu.se.2087818.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (t1gd and t2) (loc)/logs/train/20250411-114138/events.out.tfevents.1744364498.kawasaki.ad.liu.se.1147480.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (t1gd and t2) (loc)/logs/validate/20250411-114138/events.out.tfevents.1744364498.kawasaki.ad.liu.se.1147480.1"]],
                 
                 titles=
                 ["ResNet (2+1)D on T1W-GD", "ResNet (2+1)D on T2W", "ResNet (2+1)D on fused T1W-GD and T2W"],
                 
                 file_names=
                 ["resnet2p1_loc_t1gd", "resnet2p1_loc_t2", "resnet2p1_loc_t1gd_and_t2"])

make_matrices(pred_paths=
              ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (t1gd and t2) (loc)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-124822 (resnet2p1) (t1gd) (loc)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-141009 (resnet2p1) (t2) (loc)/linear_probing_results.csv"],
              
              titles=
              ["ResNet (2+1)D on fused T1W-GD and T2W", "ResNet (2+1)D on T1W-GD", "ResNet (2+1)D on T2W"],
              
              file_names=
              ["resnet2p1_loc_t1gd_and_t2", "resnet2p1_loc_t1gd", "resnet2p1_loc_t2"],
              
              n_classes=2)

#%%
### MAKE ResNet Mixed Convolution (location) LOSS CURVES AND CONFUSION MATRICES (for t1gd, t2 and both)
make_loss_curves(logfiles=
                 [["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-171150 (resnet_mixed_conv) (t1gd) (loc)/logs/train/20250414-171150/events.out.tfevents.1744643510.kawasaki.ad.liu.se.2386526.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-171150 (resnet_mixed_conv) (t1gd) (loc)/logs/validate/20250414-171150/events.out.tfevents.1744643510.kawasaki.ad.liu.se.2386526.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-161408 (resnet_mixed_conv) (t2) (loc)/logs/train/20250414-161408/events.out.tfevents.1744640048.kawasaki.ad.liu.se.2258569.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-161408 (resnet_mixed_conv) (t2) (loc)/logs/validate/20250414-161408/events.out.tfevents.1744640048.kawasaki.ad.liu.se.2258569.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (t1gd and t2) (loc)/logs/train/20250409-133111/events.out.tfevents.1744198271.kawasaki.ad.liu.se.441010.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (t1gd and t2) (loc)/logs/validate/20250409-133111/events.out.tfevents.1744198271.kawasaki.ad.liu.se.441010.1"]],
                 
                 titles=
                 ["ResNet Mixed on T1W-GD", "ResNet Mixed on T2W", "ResNet Mixed on fused T1W-GD and T2W"],
                 
                 file_names=
                 ["resnet_mixed_conv_loc_t1gd", "resnet_mixed_conv_loc_t2", "resnet_mixed_conv_loc_t1gd_and_t2"])

make_matrices(pred_paths=
              ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-171150 (resnet_mixed_conv) (t1gd) (loc)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-161408 (resnet_mixed_conv) (t2) (loc)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (t1gd and t2) (loc)/linear_probing_results.csv"],
              
              titles=
              ["ResNet Mixed on T1W-GD", "ResNet Mixed on T2W", "ResNet Mixed on fused T1W-GD and T2W"],
              
              file_names=
              ["resnet_mixed_conv_loc_t1gd", "resnet_mixed_conv_loc_t2", "resnet_mixed_conv_loc_t1gd_and_t2"],
              
              n_classes=2)


#%%
### MAKE ResNet 2p1 (diagnose) LOSS CURVES AND CONFUSION MATRICES (t1gd, t2, and both)
make_loss_curves(logfiles=
                 [["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-105151 (resnet2p1) (t1gd) (diag)/logs/train/20250416-105151/events.out.tfevents.1744793511.kawasaki.ad.liu.se.3510220.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-105151 (resnet2p1) (t1gd) (diag)/logs/validate/20250416-105151/events.out.tfevents.1744793511.kawasaki.ad.liu.se.3510220.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-130837 (resnet2p1) (t2) (diag)/logs/train/20250416-130837/events.out.tfevents.1744801717.kawasaki.ad.liu.se.3682482.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-130837 (resnet2p1) (t2) (diag)/logs/validate/20250416-130837/events.out.tfevents.1744801717.kawasaki.ad.liu.se.3682482.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-070626 (resnet2p1) (t1gd and t2) (diag)/logs/train/20250414-070626/events.out.tfevents.1744607186.kawasaki.ad.liu.se.1514516.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-070626 (resnet2p1) (t1gd and t2) (diag)/logs/validate/20250414-070626/events.out.tfevents.1744607186.kawasaki.ad.liu.se.1514516.1"]],
                 
                 titles=
                 ["ResNet (2+1)D on T1W-GD", "ResNet (2+1)D on T2W", "ResNet (2+1)D on fused T1W-GD and T2W"],
                 
                 file_names=
                 ["resnet2p1_diag_t1gd", "resnet2p1_diag_t2", "resnet2p1_diag_t1gd_and_t2"])

make_matrices(pred_paths=
              ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-105151 (resnet2p1) (t1gd) (diag)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-130837 (resnet2p1) (t2) (diag)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-070626 (resnet2p1) (t1gd and t2) (diag)/linear_probing_results.csv"],
              
              titles=
              ["ResNet (2+1)D on T1W-GD", "ResNet (2+1)D on T2W", "ResNet (2+1)D on fused T1W-GD and T2W"],
              
              file_names=
              ["resnet2p1_diag_t1gd", "resnet2p1_diag_t2", "resnet2p1_diag_t1gd_and_t2"],
              
              n_classes=3)



#%%
### MAKE ResNet Mixed Convolution (diagnose) LOSS CURVES AND CONFUSION MATRICES (t1gd, t2, and both)
make_loss_curves(logfiles=
                 [["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-082835 (resnet_mixed_conv) (t1gd) (diag)/logs/train/20250416-082835/events.out.tfevents.1744784915.kawasaki.ad.liu.se.3361028.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-082835 (resnet_mixed_conv) (t1gd) (diag)/logs/validate/20250416-082835/events.out.tfevents.1744784915.kawasaki.ad.liu.se.3361028.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-150823 (resnet_mixed_conv) (t2) (diag)/logs/train/20250416-150823/events.out.tfevents.1744808903.kawasaki.ad.liu.se.3828476.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-150823 (resnet_mixed_conv) (t2) (diag)/logs/validate/20250416-150823/events.out.tfevents.1744808903.kawasaki.ad.liu.se.3828476.1"],
                  ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-085122 (resnet_mixed_conv) (t1gd and t2) (diag)/logs/train/20250414-085122/events.out.tfevents.1744613482.kawasaki.ad.liu.se.1632539.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-085122 (resnet_mixed_conv) (t1gd and t2) (diag)/logs/validate/20250414-085122/events.out.tfevents.1744613482.kawasaki.ad.liu.se.1632539.1"]],
                 
                 titles=
                 ["ResNet Mixed on T1W-GD", "ResNet Mixed on T2W", "ResNet Mixed on fused T1W-GD and T2W"],
                 
                 file_names=
                 ["resnet_mixed_conv_diag_t1gd", "resnet_mixed_conv_diag_t2", "resnet_mixed_conv_diag_t1gd_and_t2"])

make_matrices(pred_paths=
              ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-082835 (resnet_mixed_conv) (t1gd) (diag)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-150823 (resnet_mixed_conv) (t2) (diag)/linear_probing_results.csv",
               "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-085122 (resnet_mixed_conv) (t1gd and t2) (diag)/linear_probing_results.csv"],
              
              titles=
              ["ResNet Mixed on T1W-GD", "ResNet Mixed on T2W", "ResNet Mixed on fused T1W-GD and T2W"],
              
              file_names=
              ["resnet_mixed_conv_diag_t1gd", "resnet_mixed_conv_diag_t2", "resnet_mixed_conv_diag_t1gd_and_t2"],
              
              n_classes=3)


# # DID FUSION WORK? For 3dconv?
# make_matrices(["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-152650 (3dconv) (t1gd and t2) (loc)/linear_probing_results.csv",
#                "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-121504 (3dconv) (t1gd) (loc)/linear_probing_results.csv",
#                "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-184315 (3dconv) (t2) (loc)/linear_probing_results.csv"],
#               ["3dconv_loc_t1gd_and_t2", "3dconv_loc_t1gd", "3dconv_loc_t2"],
#               ["3dconv_loc_t1gd_and_t2", "3dconv_loc_t1gd", "3dconv_loc_t2"],
#               n_classes=2)



#%%
### MAKE LOSS CURVES AND CONFUSION MATRICES for the randomly initialized models
# make_loss_curves(logfiles=
#                  [["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250415-191635 (resnet2p1) (t1gd and t2) (diag) (randomly initialized)/logs/train/20250415-191635/events.out.tfevents.1744737395.kawasaki.ad.liu.se.2908161.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250415-191635 (resnet2p1) (t1gd and t2) (diag) (randomly initialized)/logs/validate/20250415-191635/events.out.tfevents.1744737395.kawasaki.ad.liu.se.2908161.1"],
#                   ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-061735 (resnet_mixed_conv) (t1gd and t2) (diag) (randomly initialized)/logs/train/20250416-061735/events.out.tfevents.1744777055.kawasaki.ad.liu.se.3189361.0", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-061735 (resnet_mixed_conv) (t1gd and t2) (diag) (randomly initialized)/logs/validate/20250416-061735/events.out.tfevents.1744777055.kawasaki.ad.liu.se.3189361.1"]],
                  
#                  titles=
#                  ["Resnet (2+1)D on fused T1W-GD and T2W (random init.)", "Resnet Mixed on fused T1W-GD and T2W (random init.)"],
                  
#                  file_names=
#                  ["resnet2p1_diag_t1gd_and_t2_not_pretrained", "resnet_mixed_conv_diag_t1gd_and_t2_not_pretrained"])

# make_matrices(pred_paths=
#               ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250415-191635 (resnet2p1) (t1gd and t2) (diag) (randomly initialized)/linear_probing_results.csv", "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250416-061735 (resnet_mixed_conv) (t1gd and t2) (diag) (randomly initialized)/linear_probing_results.csv"],
              
#               titles=
#               ["Resnet (2+1)D on fused T1W-GD and T2W (random init.)", "Resnet Mixed on fused T1W-GD and T2W (random init.)"],
              
#               file_names=
#               ["resnet2p1_diag_t1gd_and_t2_not_pretrained", "resnet_mixed_conv_diag_t1gd_and_t2_not_pretrained"],
              
#               n_classes=3)


#%%
#### A list of paths to the final validation predictions by various training runs
# Location trainings
# val_pred_paths = ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250411-114138 (resnet2p1) (loc)/linear_probing_results.csv",
#                   "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-152650 (3dconv) (loc)/linear_probing_results.csv",
#                   "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250409-133111 (resnet_mixed_conv) (loc)/linear_probing_results.csv"]
# titles = ["resnet2p1_loc_t1gd_and_t2", "3dconv_loc_t1gd_and_t2", "resnet_mixed_conv_loc_t1gd_and_t2"]
# file_names = ["resnet2p1_loc_t1gd_and_t2", "3dconv_loc_t1gd_and_t2", "resnet_mixed_conv_loc_t1gd_and_t2"]



# CREATE DIAGNOSE CONFUSION MATRICES

#### A list of paths to the final validation predictions by various training runs

# Diagnose trainings
# val_pred_paths = ["/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-070626 (resnet2p1) (diag)/linear_probing_results.csv",
#                    "/local/data2/simjo484/Training_outputs/3D_ResNet/linear_probing/20250414-085122 (resnet_mixed_conv) (diag)/linear_probing_results.csv"]
# titles = ["resnet2p1_diag_t1gd_and_t2", "resnet_mixed_conv_diag_t1gd_and_t2"] #"3dconv_diag_t1gd_and_t2", "resnet_mixed_conv_diag_t1gd_and_t2"]
# file_names = ["resnet2p1_diag_t1gd_and_t2", "resnet_mixed_conv_diag_t1gd_and_t2"]


#%%

# for path, tit, name in zip(val_pred_paths, titles, file_names):
#     pred_logits = pd.read_csv(path)
#     preds = [int(i) for i in pred_logits.idxmax(axis=1).tolist()]
#     m = get_conf_matrix(preds, targs, n_classes=2)
#     print(m)

#     name = "/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_val_conf_matrix"+".png"
#     create_conf_matrix_fig(m, n_classes=2, title=tit, save_fig_as=name)


# for path, tit, name in zip(val_pred_paths, titles, file_names):
#     pred_logits = pd.read_csv(path)
#     preds = [int(i) for i in pred_logits.idxmax(axis=1).tolist()]
#     m = get_conf_matrix(preds, targs, n_classes=3)

#     name = "/home/simjo484/master_thesis/Master_Thesis/visualization/figures/"+name+"_val_conf_matrix"+".png"
#     create_conf_matrix_fig(m, n_classes=3, title=tit, save_fig_as=name)


'''
Make some ROC figures.

Usage: 
'''

#%%
# SETUP
from roc_curves_Iulian import plotROC
from tabulate import tabulate
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from torcheval.metrics.functional import multiclass_accuracy, multiclass_confusion_matrix, multiclass_precision, multiclass_recall

PREDICTION_PATH = "/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/statistical_tests/class_predictions"

tablefmt = "latex" #"fancy_grid"
#%%
# DIAGNOSIS METRICS



# Load test labels
test_df = pd.read_pickle("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.pkl")
diag_labels = test_df["class_label"].tolist()
diag_labels_onehot = pd.get_dummies(diag_labels).to_numpy()

# Natural language name, and file name for the prediction files
model_descriptors = [["BSF", "T1W-GD", "BSF_t1gd_diag"],
                     ["BSF", "T2W", "BSF_t2_diag"],
                     ["BSF", "T1W-GD and T2W", "BSF_t1gd_and_t2_diag"],
                     ["ResNet (2+1)D", "T1W-GD", "ResNet_2p1_t1gd_diag"],
                     ["ResNet (2+1)D", "T2W", "ResNet_2p1_t2_diag"],
                     ["ResNet (2+1)D", "T1W-GD and T2W", "ResNet_2p1_t1gd_and_t2_diag"],
                     ["ResNet Mixed", "T1W-GD", "ResNet_mixed_t1gd_diag"],
                     ["ResNet Mixed", "T2W", "ResNet_mixed_t2_diag"],
                     ["ResNet Mixed", "T1W-GD and T2W", "ResNet_mixed_t1gd_and_t2_diag"]]

metrics = []


for name, modalities, file_name in model_descriptors:

    predictions = pd.read_csv(Path(PREDICTION_PATH, file_name+".csv")).to_numpy().argmax(1)

    total_acc = multiclass_accuracy(input=torch.tensor(predictions), target=torch.tensor(diag_labels), average="micro", num_classes=3) #Verified, this is the global accuracy
    balanced_acc = multiclass_recall(input=torch.tensor(predictions), target=torch.tensor(diag_labels), average=None, num_classes=3)
    balanced_acc = sum(balanced_acc.tolist())/3
    #print(balanced_acc)

    class_prec = multiclass_precision(input=torch.tensor(predictions), target=torch.tensor(diag_labels), average=None, num_classes=3).tolist()
    class_reca = multiclass_recall(input=torch.tensor(predictions), target=torch.tensor(diag_labels), average=None, num_classes=3).tolist()
    
    # GET AUC
    predictions_onehot = pd.read_csv(Path(PREDICTION_PATH, file_name+".csv")).to_numpy()
    model_auc = plotROC(GT=diag_labels_onehot, PRED=predictions_onehot, classes=["Glioma", "Epen", "Med"],
                        #savePath="/visualization/create_figures/statistical_tests/roc_curves",
                        saveName="roc_curve_diag_"+file_name,
                        savePath="/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/statistical_tests/roc_curves",
                        title=name+" on "+modalities,
                        draw=False)[2]["macro"]

    for diagnosis, modality, prec, rec, acc, auc, bal_acc in zip(["Glioma", "Ependymoma", "Medulloblastoma"], ["", modalities, ""], class_prec, class_reca, ["", round(total_acc.tolist(),2), ""], ["", round(model_auc,2), ""], ["", round(balanced_acc,2), ""]):
        metrics.append([diagnosis, modality, round(prec,2), round(rec,2), acc, bal_acc, auc])#, #class_prec[0], class_prec[1], class_prec[2], class_reca[0], class_reca[1], class_reca[2]])

    #print(f"total accuracy for {name} {modalities}: {total_acc}")

# CREATE BSF METRIC TABLE
print(tabulate(metrics,
               headers=["Diagnosis", "Modalities", "Precision", "Recall", "Accuracy", "Bal. Accuracy", "Bal. AUC"],
               tablefmt=tablefmt))

# # CREATE ResNet (2+1)D METRIC TABLE
# print(tabulate(metrics[9:18],
#                headers=["Diagnose", "Modalities", "Precision", "Recall", "Accuracy", "AUC"],
#                tablefmt="latex"))

# # CREATE ResNet Mixed METRIC TABLE
# print(tabulate(metrics[18:],
#                headers=["Diagnose", "Modalities", "Precision", "Recall", "Accuracy", "AUC"],
#                tablefmt="latex"))


# %%
# LOCATION METRICS

# Load test labels
test_df = pd.read_csv("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df_loc.csv")
loc_labels = test_df["loc_label"].tolist()
loc_labels_onehot = pd.get_dummies(loc_labels).to_numpy()

# Natural language name, and file name for the prediction files
model_descriptors = [#["BSF", "T1W-GD", "BSF_t1gd_loc"],
                     #["BSF", "T2W", "BSF_t2_loc"],
                     #["BSF", "T1W-GD and T2W", "BSF_t1gd_and_t2_loc"],
                     ["ResNet (2+1)D", "T1W-GD", "ResNet_2p1_t1gd_loc"],
                     ["ResNet (2+1)D", "T2W", "ResNet_2p1_t2_loc"],
                     ["ResNet (2+1)D", "T1W-GD and T2W", "ResNet_2p1_t1gd_and_t2_loc"],
                     ["ResNet Mixed", "T1W-GD", "ResNet_mixed_t1gd_loc"],
                     ["ResNet Mixed", "T2W", "ResNet_mixed_t2_loc"],
                     ["ResNet Mixed", "T1W-GD and T2W", "ResNet_mixed_t1gd_and_t2_loc"]]

metrics = []


for name, modalities, file_name in model_descriptors:

    predictions = pd.read_csv(Path(PREDICTION_PATH, file_name+".csv")).to_numpy().argmax(1)

    total_acc = multiclass_accuracy(input=torch.tensor(predictions), target=torch.tensor(loc_labels), average="micro", num_classes=2)
    balanced_acc = multiclass_recall(input=torch.tensor(predictions), target=torch.tensor(loc_labels), average=None, num_classes=2)
    balanced_acc = sum(balanced_acc.tolist())/2

    class_prec = multiclass_precision(input=torch.tensor(predictions), target=torch.tensor(loc_labels), average=None, num_classes=2).tolist()
    class_reca = multiclass_recall(input=torch.tensor(predictions), target=torch.tensor(loc_labels), average=None, num_classes=2).tolist()
    
    # GET AUC
    predictions_onehot = pd.read_csv(Path(PREDICTION_PATH, file_name+".csv")).to_numpy()
    model_auc = plotROC(GT=loc_labels_onehot, PRED=predictions_onehot, classes=["Supra", "Infra"],
                        saveName="roc_curve_loc_"+file_name,
                        savePath="/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/statistical_tests/roc_curves",
                        title=name+" on "+modalities,
                        draw=False)[2]["macro"]

    for location, prec, rec, acc, auc, bal_acc in zip(["Supra", "Infra"], class_prec, class_reca, [round(total_acc.tolist(),2), "", ""], [round(model_auc,2), "", ""], [round(balanced_acc,2), "", ""]):
        metrics.append([location, modalities, round(prec,2), round(rec,2), acc, bal_acc, auc])#, #class_prec[0], class_prec[1], class_prec[2], class_reca[0], class_reca[1], class_reca[2]])

    #print(f"total accuracy for {name} {modalities}: {total_acc}")

# CREATE METRIC TABLE
# Then manually cut it apart in the latex table
print(tabulate(metrics,
               headers=["Location", "Modalities", "Precision", "Recall", "Accuracy", "Bal. Accuracy", "Bal. AUC"],
               tablefmt=tablefmt))

# %%

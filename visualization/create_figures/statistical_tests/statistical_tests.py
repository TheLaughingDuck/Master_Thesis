'''
In the report, I need to report the differences between models, and perform statistical testing.

This script does that.
'''

#%%
# SETUP
import pandas as pd
import numpy as np
import torch

from roc_curves_Iulian import plotROC
from scipy.stats import wilcoxon
from tabulate import tabulate

import matplotlib.pyplot as plt

from pathlib import Path

PREDICTION_PATH = "/home/simjo484/master_thesis/Master_Thesis/visualization/create_figures/statistical_tests/class_predictions"


# LOAD LABELS

# Load diag test labels
test_df = pd.read_csv("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df.csv")
diag_labels = test_df["class_label"].tolist()
print(f"Observations in each diagnosis class: {[sum([lab == i for lab in diag_labels]) for i in range(3)]}")
diag_labels = pd.get_dummies(diag_labels).to_numpy() # Get one-hot encoding


# Load loc test labels
test_df = pd.read_csv("/local/data1/simjo484/mt_data/all_data/MRI/simon/test_df_loc.csv")
loc_labels = test_df["loc_label"].tolist()
print(f"Observations in each location class: {[sum([lab == i for lab in loc_labels]) for i in range(2)]}")
loc_labels = pd.get_dummies(loc_labels).to_numpy() # Get one-hot encoding

#%%
# THE PREVIOUS DESIGN
## two-sided tests, and with no Bonferroni correction

# A list with an element for each individual test.
## These elements consist of *Test name*, *model 1 name* and *model 2 name*.
## The model names are used to find the correct prediction files.
test_descriptors = [# Diagnosis task
                    ["BSF", "T1W-GD vs fused", "BSF_t1gd_diag", "BSF_t1gd_and_t2_diag", "diagnose"],
                    ["BSF", "T2W vs fused", "BSF_t2_diag", "BSF_t1gd_and_t2_diag", "diagnose"],
                    ["ResNet (2+1)D", "T1W-GD vs fused", "ResNet_2p1_t1gd_diag", "ResNet_2p1_t1gd_and_t2_diag", "diagnose"],
                    ["ResNet (2+1)D", "T2W vs fused", "ResNet_2p1_t2_diag", "ResNet_2p1_t1gd_and_t2_diag", "diagnose"],
                    ["ResNet Mixed", "T1W-GD vs fused", "ResNet_mixed_t1gd_diag", "ResNet_mixed_t1gd_and_t2_diag", "diagnose"],
                    ["ResNet Mixed", "T2W vs fused", "ResNet_mixed_t2_diag", "ResNet_mixed_t1gd_and_t2_diag", "diagnose"],
                    
                    # Location
                    ["ResNet (2+1)D", "T1W-GD vs fused", "ResNet_2p1_t1gd_loc", "ResNet_2p1_t1gd_and_t2_loc", "location"],
                    ["ResNet (2+1)D", "T2W vs fused", "ResNet_2p1_t2_loc", "ResNet_2p1_t1gd_and_t2_loc", "location"],
                    ["ResNet Mixed", "T1W-GD vs fused", "ResNet_mixed_t1gd_loc", "ResNet_mixed_t1gd_and_t2_loc", "location"],
                    ["ResNet Mixed",  "T2W vs fused", "ResNet_mixed_t2_loc", "ResNet_mixed_t1gd_and_t2_loc", "location"]]
test_results = []

stats = []
diff = []

for name, seqs, model1, model2, task in test_descriptors:
    # Task settings
    if task == "diagnose":
        classes = ["Glioma", "Epen", "Med"]
        labels = diag_labels
    elif task == "location":
        classes = ["Supra", "Infra"]
        labels = loc_labels
    
    # Load predictions
    preds_model1 = pd.read_csv(Path(PREDICTION_PATH, model1+".csv")).to_numpy()
    preds_model2 = pd.read_csv(Path(PREDICTION_PATH, model2+".csv")).to_numpy()
    
    # Get the predicted probability of all classes but the true one.
    ## A sort of loss
    model1_values = [1-row[ind] for row, ind in zip(preds_model1, labels)]
    model2_values = [1-row[ind] for row, ind in zip(preds_model2, labels)]
    
    ## Avg "loss", or probability of all classes but the true class
    #avg_model1 = np.median(model1_values)
    #avg_model2 = np.median(model2_values)

    #print(model1_values)
    #print(model2_values)
    #print("\n")

    # Perform test
    statistic, p_val = wilcoxon(model1_values, model2_values, zero_method="zsplit") # e.g. (0.0, 0.25)
    stats.append(statistic)
    diff.append([i-j for i, j in zip(model1_values, model2_values)])

    # Determine test result
    if p_val < 0.05: test_res = "Reject H_0"
    else: test_res = "Fail to reject H_0"
    test_results.append([name, seqs, p_val, test_res])#, np.min(model1_values), np.median(model1_values), np.max(model1_values), np.min(model2_values), np.median(model2_values), np.max(model2_values)])

print(tabulate(test_results, headers=["Name", "Sequences", "p value", "Test Result"],
               tablefmt="latex"))

plt.hist(diff[1], bins=20)
plt.show()




#%%
# Function for running a set of tests

def run_tests(settings, task, Bonferroni_correction=True):
    '''
    Function for running one set of tests.
    settings: list containing the settings for the test, i.e. the following elements:
        - name: Name of the single-sequence model (e.g. ResNet (2+1)D)
        - seqs: Sequences used by the single-sequence, vs used by the fused model (e.g. T1W-GD vs fused)
        - model1: Stored name of the first model (e.g "ResNet_2p1_t1gd_diag")
        - model2: Stored name of the second model (e.g "ResNet_2p1_t1gd_and_t2_diag")
        - task: Task to be performed (diagnose or location)
    
    Bonferroni_correction: Whether to apply Bonferroni correction or not. True by default.

    '''

    print(f"\n\n\nRunning tests for {task} task\n")

    # Determine the number of tests, for Bonferroni correction
    n_tests = len(settings)
    if Bonferroni_correction:
        alpha = 0.05 / n_tests
        print(f"Using Bonferroni correction, test-wise alpha: {alpha}\n")
    else:
        alpha = 0.05
        print(f"Not using Bonferroni correction, test-wise alpha: {alpha}\n")

    # Task settings
    if task == "diagnose":
        #classes = ["Glioma", "Epen", "Med"]
        labels = diag_labels
    elif task == "location":
        #classes = ["Supra", "Infra"]
        labels = loc_labels
    

    test_results = []
    stats = []
    diff = []

    # Cycle through the different tests
    for name, seqs, model1, model2 in settings:
        # Load predictions
        preds_model1 = pd.read_csv(Path(PREDICTION_PATH, model1+".csv")).to_numpy()
        preds_model2 = pd.read_csv(Path(PREDICTION_PATH, model2+".csv")).to_numpy()
        
        # Get the predicted probability of the true class.
        ## A sort of loss
        model1_values = [row[ind] for row, ind in zip(preds_model1, labels)] # take 1-row[ind] to get the probability of *not* the true class
        model2_values = [row[ind] for row, ind in zip(preds_model2, labels)]
        
        # Perform test
        statistic, p_val = wilcoxon(model1_values, model2_values, zero_method="pratt", alternative="less")
        stats.append(statistic)
        diff.append([i-j for i, j in zip(model1_values, model2_values)])

        # Determine test result
        if p_val < alpha: test_res = "Reject $H_0$"
        else: test_res = "Fail to reject $H_0$"
        
        test_results.append([name, seqs, p_val, alpha, test_res])
    
    print(tabulate(test_results, headers=["Name", "Sequences", "p value", "$alpha$", "Test Result"],
        tablefmt="latex"))
    


#%%

# DIAGNOSIS TASK TESTS
run_tests([
    ["BSF", "T1W-GD vs fused", "BSF_t1gd_diag", "BSF_t1gd_and_t2_diag"],
    ["BSF", "T2W vs fused", "BSF_t2_diag", "BSF_t1gd_and_t2_diag"],
    ["ResNet (2+1)D", "T1W-GD vs fused", "ResNet_2p1_t1gd_diag", "ResNet_2p1_t1gd_and_t2_diag"],
    ["ResNet (2+1)D", "T2W vs fused", "ResNet_2p1_t2_diag", "ResNet_2p1_t1gd_and_t2_diag"],
    ["ResNet Mixed", "T1W-GD vs fused", "ResNet_mixed_t1gd_diag", "ResNet_mixed_t1gd_and_t2_diag"],
    ["ResNet Mixed", "T2W vs fused", "ResNet_mixed_t2_diag", "ResNet_mixed_t1gd_and_t2_diag"]],
    
    task="diagnose")


# LOCATION TASK TESTS
run_tests([
    ["ResNet (2+1)D", "T1W-GD vs fused", "ResNet_2p1_t1gd_loc", "ResNet_2p1_t1gd_and_t2_loc"],
    ["ResNet (2+1)D", "T2W vs fused", "ResNet_2p1_t2_loc", "ResNet_2p1_t1gd_and_t2_loc"],
    ["ResNet Mixed", "T1W-GD vs fused", "ResNet_mixed_t1gd_loc", "ResNet_mixed_t1gd_and_t2_loc"],
    ["ResNet Mixed",  "T2W vs fused", "ResNet_mixed_t2_loc", "ResNet_mixed_t1gd_and_t2_loc"]],
    
    task="location")
# %%


# %%
# SHOW EVENT LOG
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import re
import os
from utils import *
import numpy as np

def plot_train_val_from_eventfile(event_file=None, latest = True, print_values=False, maintitle=None):
    if latest and event_file == None:
        path = "/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/"
        path += sorted(os.listdir(path))[-1] + "/" # Get latest run folder
        path += sorted(os.listdir(path), key=lambda x: len(x), reverse=True)[0] # Get events file path (longest file name)
        print(path)
        
        event_file = path

    # Extract start date+time, e.g. "2025-03-10-07:40:57"
    start_date = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}", event_file).group(0)

    if latest and maintitle == None:
        maintitle = "Last run ("+start_date+")"
    
    data = {}

    # Iterate over events in the event file
    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag not in data:
                data[value.tag] = {'step': [], 'value': []}
            data[value.tag]['step'].append(float(summary.step))
            data[value.tag]['value'].append(float(value.simple_value))

    fig, axs = plt.subplots(nrows=2)
    
    # Get loss weights
    class_prop = [i/165 for i in [132, 14, 19]]
    naive_precision = [i for i in class_prop] # Precisions are equal to the class proportions
    naive_recall = [1/3 for i in class_prop] 

    # Train and val loss
    axs[0].set_title("Avg Training and validation loss")
    axs[0].plot(data["avg_train_loss"]["step"], data["avg_train_loss"]["value"], color="blue", label="Training")
    axs[0].plot(data["avg_val_loss"]["step"], data["avg_val_loss"]["value"], color="orange", label="Validation")
    axs[0].legend()

    # Validation Accuracy
    n_steps = len(data["acc"]["step"])
    axs[1].set_title("Validation accuracy (global, unweighted)")
    axs[1].plot(data["acc"]["step"], data["acc"]["value"], color="blue")
    axs[1].plot(data["acc"]["step"], [1/3 for i in range(n_steps)], color="gray", linestyle="--")
    
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    fig.suptitle(maintitle, fontsize=16)
    fig.show()
    
    # Precision and Recall plots
    fig, axs = plt.subplots(nrows=2)
    # Precision
    axs[0].set_title("Precision")
    axs[0].plot(data["prec_class_0"]["step"], data["prec_class_0"]["value"], color="blue", label="Class 0")
    axs[0].plot(data["prec_class_0"]["step"], [naive_precision[0] for i in range(n_steps)], color="blue", linestyle="--")
    
    axs[0].plot(data["prec_class_1"]["step"], data["prec_class_1"]["value"], color="orange", label="Class 1")
    axs[0].plot(data["prec_class_1"]["step"], [naive_precision[1] for i in range(n_steps)], color="orange", linestyle="--")

    axs[0].plot(data["prec_class_2"]["step"], data["prec_class_2"]["value"], color="green", label="Class 2")
    axs[0].plot(data["prec_class_2"]["step"], [naive_precision[2] for i in range(n_steps)], color="green", linestyle="--")
    axs[0].legend()

    # Recall
    axs[1].set_title("Recall")
    axs[1].plot(data["rec_class_0"]["step"], data["rec_class_0"]["value"], color="blue", label="Class 0")
    #axs[1].plot(data["rec_class_0"]["step"], [naive_recall[0] for i in range(n_steps)], color="blue", linestyle="--")
    
    axs[1].plot(data["rec_class_1"]["step"], data["rec_class_1"]["value"], color="orange", label="Class 1")
    #axs[1].plot(data["rec_class_1"]["step"], [naive_recall[1] for i in range(n_steps)], color="orange", linestyle="--")

    axs[1].plot(data["rec_class_2"]["step"], data["rec_class_2"]["value"], color="green", label="Class 2")
    #axs[1].plot(data["rec_class_2"]["step"], [naive_recall[2] for i in range(n_steps)], color="green", linestyle="--")
    
    axs[1].plot(data["rec_class_2"]["step"], [1/3 for i in range(n_steps)], color="gray", linestyle="--")
    axs[1].legend()

    # Figure settings
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    fig.suptitle(maintitle, fontsize=16)
    fig.show()

    if print_values:
        for summary in summary_iterator(path):
            print(summary.summary.value)
    

    #### CHECK IF THERE IS A CONFUSION MATRIX ####
    dir_path = os.path.dirname(event_file)
    target_file_path = os.path.join(dir_path, "conf_matrices.png")

    if os.path.isfile(target_file_path):
        # If there is an image, load and show it.
        img = plt.imread(target_file_path)
        fig, axs = plt.subplots(nrows=1)
        axs.imshow(img)
        axs.axis('off')
        fig.show()

    


# plot_train_val_from_eventfile(event_file="/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/2025-03-17-13:25:01 (Run for 300 epochs)/events.out.tfevents.1742214307.kawasaki.ad.liu.se",
#                               maintitle="Classifier with finetuned BSF")

# plot_train_val_from_eventfile(event_file="/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/2025-03-18-08:22:28 (With 4-channel out-of-box BSF on T2 only)/events.out.tfevents.1742282554.kawasaki.ad.liu.se",
#                               maintitle="Classifier with original BSF")

plot_train_val_from_eventfile()
# %%

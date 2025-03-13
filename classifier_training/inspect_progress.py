# %%
# SHOW EVENT LOG
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import re
import os

def plot_train_val_from_eventfile(event_file=None, latest = True, print_values=False):
    if latest:
        path = "/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/"
        path += sorted(os.listdir(path))[-1] + "/" # Get latest run folder
        path += sorted(os.listdir(path))[0] # Get events file path
        print(path)
        
        event_file = path

    # Extract start date+time, e.g. "2025-03-10-07:40:57"
    start_date = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}", event_file).group(0)
    
    data = {}

    # Iterate over events in the event file
    for summary in summary_iterator(event_file):
        for value in summary.summary.value:
            if value.tag not in data:
                data[value.tag] = {'step': [], 'value': []}
            data[value.tag]['step'].append(float(summary.step))
            data[value.tag]['value'].append(float(value.simple_value))

    fig, axs = plt.subplots(nrows=2)

    # Train and val loss
    axs[0].set_title("Avg Training and validation loss")
    axs[0].plot(data["avg_train_loss"]["step"], data["avg_train_loss"]["value"], color="blue", label="Training")
    axs[0].plot(data["avg_val_loss"]["step"], data["avg_val_loss"]["value"], color="orange", label="Validation")
    axs[0].legend()

    # Validation Accuracy
    axs[1].set_title("Validation accuracy (global, unweighted)")
    axs[1].plot(data["acc"]["step"], data["acc"]["value"], color="blue")
    
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    fig.suptitle("Metrics ("+start_date+")", fontsize=16)
    fig.show()
    
    # Precision and Recall plots
    fig, axs = plt.subplots(nrows=2)
    # Precision
    axs[0].set_title("Precision")
    axs[0].plot(data["prec_class_0"]["step"], data["prec_class_0"]["value"], color="blue", label="Class 0")
    axs[0].plot(data["prec_class_1"]["step"], data["prec_class_1"]["value"], color="orange", label="Class 1")
    axs[0].plot(data["prec_class_2"]["step"], data["prec_class_2"]["value"], color="green", label="Class 2")
    axs[0].legend()

    # Recall
    axs[1].set_title("Recall")
    axs[1].plot(data["rec_class_0"]["step"], data["rec_class_0"]["value"], color="blue", label="Class 0")
    axs[1].plot(data["rec_class_1"]["step"], data["rec_class_1"]["value"], color="orange", label="Class 1")
    axs[1].plot(data["rec_class_2"]["step"], data["rec_class_2"]["value"], color="green", label="Class 2")
    axs[1].legend()

    # Figure settings
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    fig.suptitle("Metrics ("+start_date+")", fontsize=16)
    fig.show()

    if print_values:
        for summary in summary_iterator(path):
            print(summary.summary.value)


    # Confusion matrix
    # fig, axs = plt.subplots(nrows=1)
    # mat = [[1,2,3], [3, 4,5], [5,6,7]]
    # axs.matshow(mat)
    # plt.ylabel("True")
    # plt.xlabel("Pred")
    # fig.suptitle("Confusion matrix ("+start_date+")")
    # fig.show()


plot_train_val_from_eventfile(print_values=False)

# %%

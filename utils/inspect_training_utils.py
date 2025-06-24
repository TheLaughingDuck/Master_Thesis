'''
Helper functions for creating training curves etc.
'''
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import re
import os

import tensorflow as tf

#%%

def create_loss_curve_fig(): # Might not be used?
    path = "/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/"
    path += sorted(os.listdir(path))[-1] + "/" # Get latest run folder
    logdir = path
    path += sorted(os.listdir(path), key=lambda x: len(x), reverse=True)[0] # Get events file path (longest file name)
    event_file = path

    print(f"event_file is {event_file}")
    data = {}

    # Iterate over events in the event file
    for summary in tf.data.TFRecordDataset(event_file):#summary_iterator(event_file):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAa")
        for value in summary.summary.value:
            if value.tag not in data:
                data[value.tag] = {'step': [], 'value': []}
            data[value.tag]['step'].append(float(summary.step))
            data[value.tag]['value'].append(float(value.simple_value))

    fig, axs = plt.subplots(nrows=1)

    # Train and val loss
    print(data)
    axs.plot(data["avg_train_loss"]["step"], data["avg_train_loss"]["value"], color="blue", label="Training")
    axs.plot(data["avg_val_loss"]["step"], data["avg_val_loss"]["value"], color="orange", label="Validation")
    axs.legend()
    start_date = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}", event_file).group(0)
    maintitle = "Last run ("+start_date+")"
    fig.suptitle(maintitle, fontsize=16)
    fig.savefig(logdir+"/loss_curves")
    fig.show()
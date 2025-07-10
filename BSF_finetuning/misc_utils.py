'''
This script is for miscellaneous utils.
'''

import time
from matplotlib import pyplot as plt
import json
import os

class TrainingTracker():
    def __init__(self, args):
        self.start_time = time.ctime()
        self.end_time = None
        self.args = args

        self.epoch_data = {}
            # {"avg_train_loss": {"step": [], "value": []},
            # "avg_valid_loss": {"step": [], "value": []},
            # "acc_glob_unweighted": {"step": [], "value": []},

            # "prec_class_0": {"step": [], "value": []},
            # "prec_class_1": {"step": [], "value": []},
            # "prec_class_2": {"step": [], "value": []},

            # "rec_class_0": {"step": [], "value": []},
            # "rec_class_1": {"step": [], "value": []},
            # "rec_class_2": {"step": [], "value": []},

            # "learning_rate": {"step": [], "value": []},
            # "epoch_time": {"step": [], "value": []},
            
            # # For testing
            # "A": {"step": [1,2,3], "value": [7,8,9]},
            # "B": {"step": [1,2,3], "value": [4,5,6]}}
    
    def update_epoch(self, data: dict = None):
        for key in data.keys():
            metric = data[key]

            if key not in self.epoch_data.keys():
                print(key)
                self.epoch_data[key] = {}
                self.epoch_data[key]["step"] = metric["step"] #epoch
                self.epoch_data[key]["value"] = metric["value"]
            else:
                self.epoch_data[key]["step"] += metric["step"] #epoch
                self.epoch_data[key]["value"] += metric["value"]
    

    def make_key_fig(self, keys: list = None, kwargs=None, title=""):
        '''
        Make a plot of one or more keys in self.epoch_data. Can also pass keyword arguments to specify color
        for each key.

        Example kwargs structure: {"key1": {"color": "blue"}, "key2": {"color": "orange"}}
        '''
        if type(keys) != list: keys = [keys]

        fig, axs = plt.subplots(nrows=1)
        for key in keys:
            # Process the kwargs
            if kwargs != None and key in kwargs.keys():
                color = kwargs[key]["color"] if "color" in kwargs[key].keys() else None
                label = kwargs[key]["label"] if "label" in kwargs[key].keys() else None
            else:
                color = None
                label = None
            axs.plot(self.epoch_data[key]["step"], self.epoch_data[key]["value"], color=color, label=label)
        axs.legend()
        axs.set_xlabel("Epochs")
        title += "(" + self.start_time + ")"
        fig.suptitle(title, fontsize=16)
        fig.savefig(self.args.logdir + "/" + keys[0] + "_fig")#self.args.logdir+"/key_fig")
    
    def to_json(self):
        filename = self.args.logdir+"/TrainingTracker_metrics.json"
        with open(filename, 'w') as file:
            json.dump(self.epoch_data, file)
        print("\nSaved epoch data to\n\t", filename)

#obj = TrainingTracker(300)

#obj.update_epoch({"A": {"step": [16,17,18], "value": [7,8,9]}})
#obj.plot_key(["A", "B"], {"A": {"color": "orange", "label": "Training"}, "B": {"color": "blue", "label": "Validation"}}, title="Avg Training and Validation loss")
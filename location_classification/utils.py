# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_recall

def get_metrics(all_preds:list, all_targets:list, num_classes:int, config, epoch:int, conf_matr_title:str):
    '''
    Function that takes two torch tensors; all_preds, and all_targets,
    and calculates various performance metrics. Returns a dict structure.
    '''
    # Create and save confusion matrix
    conf_matrix = get_conf_matrix(all_preds=all_preds.tolist(), all_targets=all_targets.tolist())
    create_conf_matrix_fig(conf_matrix, save_fig_as=config["logdir"]+"/validation_matrix", epoch=epoch, title=conf_matr_title)

    acc = multiclass_accuracy(all_preds, target=all_targets, num_classes=num_classes, average="micro")
    prec = multiclass_precision(all_preds, target=all_targets, num_classes=num_classes, average=None)
    rec = multiclass_recall(all_preds, target=all_targets, num_classes=num_classes, average=None)

    metrics = {"acc": acc, "prec": prec, "rec": rec}

    return metrics



def get_conf_matrix(all_preds, all_targets, n_classes=3):
    '''
    Takes two integer lists of all target classes, and all predictions by some classifier.

    Returns a confusion matrix, with true class on rows, and predicted class on the columns,
    as per https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    '''
    matrix = [[0 for i in range(n_classes)] for i in range(n_classes)]

    print(f"There are {len(all_preds)} predictions in all_preds.")

    for i in range(n_classes):
        for j in range(n_classes):
            for tar, pre in zip(all_targets, all_preds):
                if tar == i and pre == j:
                    matrix[i][j] += 1
    
    return(matrix)



import re

def create_conf_matrix_fig(conf_matrix, save_fig_as=None, epoch=None, title=""):
    '''
    Takes confusion matrices (on training and validation data),
    and creates a figure with them. Saves as a png.

    The true classes are on the rows, and the predicted values on the columns.
    '''
    fig, axs = plt.subplots(ncols=1)
    #fig.tight_layout(rect=(0,0,1,0.999))

    # axs[0].matshow(train_mat)
    # for (i, j), z in np.ndenumerate(train_mat):
    #     axs[0].text(j, i, '{}'.format(z), ha='center', va='center')
    # axs[0].set_yticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    # axs[0].set_xticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    # axs[0].set_xlabel("True value")
    # axs[0].set_ylabel("Prediction")
    # axs[0].set_title("Training data")

    axs.matshow(conf_matrix)
    for (i, j), z in np.ndenumerate(conf_matrix):
        axs.text(j, i, '{}'.format(z), ha='center', va='center')
    axs.set_yticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    axs.set_xticks(ticks=[0,1,2], labels=["Gli", "Epe", "Med"])
    axs.set_ylabel("True value")
    axs.set_xlabel("Prediction")
    #axs.set_title("Validation data")

    if save_fig_as != None:
        start_date = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}", save_fig_as).group(0)
        fig.suptitle(title+" (Epoch "+str(epoch)+")", fontsize=16)
        fig.savefig(save_fig_as)
    else:
        fig.suptitle("Confusion matrix", fontsize=16)
        fig.show()
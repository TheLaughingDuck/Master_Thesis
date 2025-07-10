'''
Created by Iulian Emil Tampu
'''

import os
import numpy as np
import itertools
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(GT, PRED):
    """
    GT and PRED are supposed to be categorical (or logits for the prediction)
    """
    GT = GT.argmax(axis=-1)
    PRED = PRED.argmax(axis=-1)
    nbr_classes = GT.max() + 1
    nbr_samples = GT.shape[0]
    cm = np.zeros((nbr_classes, nbr_classes))

    for s in range(nbr_samples):
        t_idx = GT[s]
        p_idx = PRED[s]
        cm[t_idx, p_idx] += 1
    return cm


def plotModelPerformance_v2(
    tr_loss, tr_acc, val_loss, val_acc, save_path, display=False, best_epoch=None
):
    """
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - save_path: path to where to save the model
    """

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "pink",
        "gray",
        "purple",
        "brown",
        "olive",
        "cyan",
        "teal",
    ]
    line_style = [":", "-.", "--", "-"]
    ax1.set_xlabel("Epochs", fontsize=15)
    ax1.set_ylabel("Loss", fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[2])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[3])
    plt.legend(["Training loss", "Validation loss"])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Accuracy and F1-score", fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])
    if best_epoch:
        l7 = ax2.axvline(x=best_epoch)

    # add legend
    if best_epoch:
        lns = l1 + l2 + l3 + l4 + l7
        labs = [
            "Training loss",
            "Validation loss",
            "Training accuracy",
            "Validation accuracy",
            "Best_model",
        ]
        ax1.legend(lns, labs, loc=7, fontsize=15)
    else:
        lns = l1 + l2 + l3 + l4
        labs = [
            "Training loss",
            "Validation loss",
            "Training accuracy",
            "Validation accuracy",
        ]
        ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title("Training loss, accuracy and Dice-score trends", fontsize=20)
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(os.path.join(save_path, "perfomance.pdf"), bbox_inches="tight", dpi=100)
    fig.savefig(os.path.join(save_path, "perfomance.png"), bbox_inches="tight", dpi=100)
    plt.close()

    if display is True:
        plt.show()
    else:
        plt.close()


def get_performance_metrics(true_logits, pred_softmax, average="macro"):
    from sklearn.metrics import (
        average_precision_score,
        recall_score,
        roc_curve,
        auc,
        roc_auc_score,
        f1_score,
        accuracy_score,
        matthews_corrcoef,
        confusion_matrix,
    )

    """
    Utility that returns the evaluation metrics as a disctionary.
    THe metrics that are returns are:
    - accuracy
    - f1-score
    - precision and recall
    - auc
    - MCC
    INPUT
    Ytest : np.array
        Array containing the ground truth for the test data
    Ptest_softmax : np.array
        Array containing the softmax output of the model for each
        test sample
    OUTPUT
    metrics_dict : dictionary
    """
    # there might be cases where one of the classes does not have any sample in the true_logits. If this is the case
    # roc_curve and subsequent auc calculation will complain. Thus check this, save the index of the class with no
    # positive evidence and add a None value in the AUC computation
    index_class_with_no_positive_evidence = [
        i for i in range(true_logits.shape[-1]) if true_logits[:, i].sum() == 0
    ]

    if len(index_class_with_no_positive_evidence):
        adjusted_true_logits = np.delete(
            true_logits, index_class_with_no_positive_evidence, 1
        )
        adjusted_pred_softmax = np.delete(
            pred_softmax, index_class_with_no_positive_evidence, 1
        )
    else:
        adjusted_true_logits = true_logits
        adjusted_pred_softmax = pred_softmax

    # compute confusion matrix
    cnf_matrix = get_confusion_matrix(
        true_logits,
        pred_softmax,
    )

    # get TP, TN, FP, FN
    FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    TP = (np.diag(cnf_matrix)).astype(float)
    TN = (cnf_matrix.sum() - (FP + FN + TP)).astype(float)

    # compute class metrics
    fpr, tpr, _ = roc_curve(
        adjusted_true_logits.argmax(axis=-1) + 1,
        adjusted_pred_softmax.argmax(axis=-1) + 1,
        pos_label=np.unique(adjusted_true_logits.argmax(axis=-1)).size,
    )
    precision = np.array(TN / (FP + TN))
    recall = np.array(TP / (TP + FN))
    accuracy = np.array((TP + TN) / (TP + TN + FP + FN))
    f1score = np.array(TP / (TP + 0.5 * (FP + FN)))

    # set to None the class that does not have any positive evidence
    precision[index_class_with_no_positive_evidence] = None
    recall[index_class_with_no_positive_evidence] = None
    accuracy[index_class_with_no_positive_evidence] = None
    f1score[index_class_with_no_positive_evidence] = None

    summary_dict = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1-score": f1score,
        "auc": auc(fpr, tpr),
    }

    # compute overall metrics
    summary_dict["overall_precision"] = average_precision_score(
        adjusted_true_logits, adjusted_pred_softmax, average=average
    )
    summary_dict["overall_recall"] = recall_score(
        np.argmax(adjusted_true_logits, axis=-1),
        np.argmax(adjusted_pred_softmax, axis=-1),
        average=average,
    )
    summary_dict["overall_accuracy"] = accuracy_score(
        np.argmax(adjusted_true_logits, axis=-1),
        np.argmax(adjusted_pred_softmax, axis=-1),
    )
    summary_dict["overall_f1-score"] = f1_score(
        np.argmax(adjusted_true_logits, axis=-1),
        np.argmax(adjusted_pred_softmax, axis=-1),
        average=average,
    )

    summary_dict["overall_auc"] = auc(fpr, tpr)

    summary_dict["matthews_correlation_coefficient"] = matthews_corrcoef(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
    )

    return summary_dict


def plotConfusionMatrix(
    GT,
    PRED,
    classes,
    cmap=plt.cm.Blues,
    savePath=None,
    saveName=None,
    draw=False,
):
    from sklearn.metrics import matthews_corrcoef

    """
    Funtion that plots the confision matrix given the ground truths and the predictions
    """
    # compute confusion matrix
    cm = get_confusion_matrix(GT, PRED)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation=None, cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=25,
        )

    # plt.tight_layout()
    plt.ylabel("True label", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    acc = 100 * (np.trace(cm) / np.sum(cm))
    mcc = matthews_corrcoef(GT.argmax(axis=-1), PRED.argmax(axis=-1))
    plt.title(f"Accuracy: {acc:03.2f}, MCC: {mcc:3.2f}", fontsize=20)
    fig.tight_layout()

    # save if needed
    if savePath is not None:
        # set up name
        if saveName is None:
            saveName = "ConfisionMatrix_ensemble_prediction"

        if os.path.isdir(savePath):
            fig.savefig(
                os.path.join(savePath, f"{saveName}.pdf"), bbox_inches="tight", dpi=100
            )
            fig.savefig(
                os.path.join(savePath, f"{saveName}.png"), bbox_inches="tight", dpi=100
            )
        else:
            raise ValueError(
                "Invalid save path: {}".format(os.path.join(savePath, f"{saveName}"))
            )

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return f"ACC: {acc:3.2f}, MCC: {mcc:3.2f}"


## PLOT ROC


def plotROC(GT, PRED, classes, savePath=None, saveName=None, draw=False):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    from scipy import interp
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    """
    Funtion that plots the ROC curve given the ground truth and the logits prediction

    INPUT
    GT : array
        True labels
    PRED : array
        Array of float the identifies the logits prediction
    classes : list
        List of string that identifies the labels of each class
    save path : string
        Identifies the path where to save the ROC plots
    save name : string
        Specifying the name of the file to be saved
    draw : bool
        True if to print the ROC curve

    OUTPUT
    fpr : dictionary that contains the false positive rate for every class and
           the overall micro and marco averages
    trp : dictionary that contains the true positive rate for every class and
           the overall micro and marco averages
    roc_auc : dictionary that contains the area under the curve for every class and
           the overall micro and marco averages

    Check this link for better understanding of micro and macro-averages
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    Here computing both the macro-average ROC and the micro-average ROC.
    Using code from https://scikit-learn.org/dev/auto_examples/model_selection/plot_roc.html with modification
    """
    # handle the case where there are no positive evidence for a class
    index_class_with_no_positive_evidence = [
        i for i in range(GT.shape[-1]) if GT[:, i].sum() == 0
    ]

    if len(index_class_with_no_positive_evidence):
        adjusted_GT = np.delete(GT, index_class_with_no_positive_evidence, 1)
        adjusted_PRED = np.delete(PRED, index_class_with_no_positive_evidence, 1)
    else:
        adjusted_GT = GT
        adjusted_PRED = PRED

    # define variables
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes = list(np.delete(classes, index_class_with_no_positive_evidence, 0))
    n_classes = len(classes)
    lw = 2  # line width

    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(adjusted_GT[:, i], adjusted_PRED[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        adjusted_GT.ravel(), adjusted_PRED.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # ¤¤¤¤¤¤¤¤¤¤ macro-average roc

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves and save
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(
        [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
            "teal",
        ]
    )
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {} (area = {:0.2f})"
            "".format(classes[i], roc_auc[i]),
        )

    ax.plot([0, 1], [0, 1], "k--", lw=lw)

    major_ticks = np.arange(0, 1, 0.1)
    minor_ticks = np.arange(0, 1, 0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.grid(color="b", linestyle="-.", linewidth=0.1, which="both")

    ax.set_xlabel("False Positive Rate", fontsize=25)
    ax.set_ylabel("True Positive Rate", fontsize=25)
    ax.set_title("Multi-class ROC (OneVsAll)", fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ work on the zummed-in image
    # colors = cycle(
    #     [
    #         "blue",
    #         "orange",
    #         "green",
    #         "red",
    #         "purple",
    #         "brown",
    #         "pink",
    #         "gray",
    #         "olive",
    #         "cyan",
    #         "teal",
    #     ]
    # )
    # axins = zoomed_inset_axes(
    #     ax, zoom=1, loc=7, bbox_to_anchor=(0, 0, 0.99, 0.9), bbox_transform=ax.transAxes
    # )
    # print(axins)

    # axins.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )

    # axins.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
    #     color="navy",
    #     linestyle=":",
    #     linewidth=4,
    # )

    # for i, color in zip(range(n_classes), colors):
    #     axins.plot(
    #         fpr[i],
    #         tpr[i],
    #         color=color,
    #         lw=lw,
    #         label="ROC curve of class {} (area = {:0.2f})"
    #         "".format(classes[i], roc_auc[i]),
    #     )

    #     # sub region of the original image
    #     x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
    #     axins.set_xlim(x1, x2)
    #     axins.set_ylim(y1, y2)
    #     axins.grid(color="b", linestyle="--", linewidth=0.1)

    #     axins.set_xticks(np.linspace(x1, x2, 4))
    #     axins.set_yticks(np.linspace(y1, y2, 4))

    # # draw a bbox of the region of the inset axes in the parent axes and
    # # connecting lines between the bbox and the inset axes area
    # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", ls="--")

    # save is needed
    if savePath is not None:
        # set up name
        if saveName is None:
            saveName = "Multiclass_ROC"

        if os.path.isdir(savePath):
            fig.savefig(
                os.path.join(savePath, f"{saveName}.pdf"), bbox_inches="tight", dpi=100
            )
            fig.savefig(
                os.path.join(savePath, f"{saveName}.png"), bbox_inches="tight", dpi=100
            )
        else:
            raise ValueError(
                "Invalid save path: {}".format(os.path.join(savePath, f"{saveName}"))
            )

    if draw is True:
        plt.draw()
    else:
        plt.close()

    # return fpr, tpr, roc_auc


## PLOR PR (precision and recall) curves


def plotPR(GT, PRED, classes, savePath=None, saveName=None, draw=False):
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        f1_score,
    )
    from sklearn.metrics import average_precision_score
    from itertools import cycle
    from scipy import interp
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    """
    Funtion that plots the PR (precision and recall) curve given the ground truth and the logits prediction

    INPUT
    - GT: true labels
    - PRED: array of float the identifies the logits prediction
    - classes: list of string that identifies the labels of each class
    - save path: sting that identifies the path where to save the ROC plots
    - save name: string the specifies the name of the file to be saved.
    - draw: bool if to print or not the ROC curve

    OUTPUT
    - precision: dictionary that contains the precision every class and micro average
    - recall: dictionary that contains the recall for every class and micro average
    - average_precision: float of the average precision
    - F1: dictionare containing the micro and marco average f1-score
    """
    # handle the case where there are no positive evidence for a class
    index_class_with_no_positive_evidence = [
        i for i in range(GT.shape[-1]) if GT[:, i].sum() == 0
    ]

    if len(index_class_with_no_positive_evidence):
        adjusted_GT = np.delete(GT, index_class_with_no_positive_evidence, 1)
        adjusted_PRED = np.delete(PRED, index_class_with_no_positive_evidence, 1)
    else:
        adjusted_GT = GT
        adjusted_PRED = PRED

    # define variables
    precision = dict()
    recall = dict()
    average_precision = dict()
    classes = list(np.delete(classes, index_class_with_no_positive_evidence, 0))
    n_classes = len(classes)
    lw = 2  # line width

    # ¤¤¤¤¤¤¤¤¤¤¤ f1_score
    F1 = {
        "micro": f1_score(
            np.argmax(adjusted_GT, axis=-1),
            np.argmax(adjusted_PRED, axis=-1),
            average="micro",
        ),
        "macro": f1_score(
            np.argmax(adjusted_GT, axis=-1),
            np.argmax(adjusted_PRED, axis=-1),
            average="macro",
        ),
    }
    # print('F1-score (micro and macro): {0:0.2f} and {0:0.2f}'.format(F1['micro'], F1['macro']))

    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            adjusted_GT[:, i], adjusted_PRED[:, i]
        )
        average_precision[i] = average_precision_score(
            adjusted_GT[:, i], adjusted_PRED[:, i]
        )

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        adjusted_GT.ravel(), adjusted_PRED.ravel()
    )
    average_precision["micro"] = average_precision_score(
        adjusted_GT, adjusted_PRED, average="micro"
    )
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #  .format(average_precision["micro"]))

    # Plot all PR curves and save

    # create iso-f1 curves and plot on top the PR curves for every class
    fig, ax = plt.subplots(figsize=(10, 10))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for idx, f_score in enumerate(f_scores):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        if idx == 0:
            (l,) = ax.plot(
                x[y >= 0], y[y >= 0], color="gray", alpha=0.2, label="iso-f1 curves"
            )
        else:
            (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    # labels.append('iso-f1 curves')
    (l,) = ax.plot(
        recall["micro"],
        precision["micro"],
        color="gold",
        lw=lw,
        label="micro-average Precision-recall (area = {0:0.2f})".format(
            average_precision["micro"]
        ),
    )
    lines.append(l)
    # labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

    colors = cycle(
        [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
            "teal",
        ]
    )
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            recall[i],
            precision[i],
            color=color,
            lw=lw,
            label="Precision-recall curve of class {:9s} (area = {:0.2f})"
            "".format(classes[i], average_precision[i]),
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("True positive rate - Recall [TP/(TP+FN)]", fontsize=20)
    ax.set_ylabel("Positive predicted value - Precision [TP/(TP+TN)]", fontsize=20)
    ax.set_title("Multi-class Precision-recall curve", fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # save is needed
    if savePath is not None:
        # set up name
        if saveName is None:
            saveName = "Multiclass_PR"

        if os.path.isdir(savePath):
            fig.savefig(
                os.path.join(savePath, f"{saveName}.pdf"), bbox_inches="tight", dpi=100
            )
            fig.savefig(
                os.path.join(savePath, f"{saveName}.png"), bbox_inches="tight", dpi=100
            )
        else:
            raise ValueError(
                "Invalid save path: {}".format(os.path.join(savePath, f"{saveName}"))
            )

    if draw is True:
        plt.draw()
    else:
        plt.close()

    # return precision, recall, average_precision, F1
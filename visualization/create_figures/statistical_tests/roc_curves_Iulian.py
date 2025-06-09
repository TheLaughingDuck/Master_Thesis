import numpy as np
import matplotlib.pyplot as plt
import os

def plotROC(GT, PRED, classes, savePath=None, saveName=None, draw=False, title=""):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    #from scipy import interp
    from numpy import interp
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
    # ax.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )

    # ax.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
    #     color="navy",
    #     linestyle=":",
    #     linewidth=4,
    # )

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

    # Simon
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax.set_xlabel("False Positive Rate", fontsize=25)
    ax.set_ylabel("True Positive Rate", fontsize=25)

    if title == "":
        ax.set_title("Multi-class ROC (OneVsAll)", fontsize=20)
    elif title == "None":
        pass
    else:
        ax.set_title(title, fontsize=40)
    plt.legend(loc="lower right", fontsize=22)

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
            # fig.savefig(
            #     os.path.join(savePath, f"{saveName}.pdf"), bbox_inches="tight", dpi=100
            # )
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

    return fpr, tpr, roc_auc
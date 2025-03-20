'''
Script that trains a classifier on some combination of T1, T2 and T1-GD sequences.

This script was adapted from scripts used to train the BrainSegFounder models.
This script was copied and modified in March of 2025. See
https://github.com/lab-smile/BrainSegFounder
for the original source code, that falls under the LICENSE that is also available in this dir.
'''

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from trainer import run_training
#from utils.data_utils import get_loader

from matplotlib import pyplot as plt
import re

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

#parser = argparse.ArgumentParser(description="Classifier pipeline")

# # Arguments that should be modified
# parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
# parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
# parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
# parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
# parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
# parser.add_argument("--feature_extractor", default="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-05-08:07:48/model_final.pt", type=str, help="Path to the fine-tuned feature extractor model weights")
# parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
# parser.add_argument("--debug_mode", default="False", type=str, help="Set the pipeline into debug mode: only a few observations are used to achieve massive speedup.") # change to store_true=False
# parser.add_argument("--comment", default="", type=str, help="A short comment for the output file, to help distinguish previous runs. Example: \".../runs/2025-XX-XX-XX:XX:XX (Deeper classifier)\"")

# # Arguments to probably leave alone
# parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
# parser.add_argument("--batch_size", default=3, type=int, help="Number of observations per batch")
# parser.add_argument("--logdir", default=".", type=str, help="directory to save the tensorboard logs")
# parser.add_argument("--workers", default=18, type=int, help="number of workers")
# parser.add_argument("--pp_device", default="cpu", type=str, help="Preprocessing device")
# parser.add_argument("--cl_device", default="cuda", type=str, help="Classifier device")
# parser.add_argument("--save_checkpoint", default="True", help="save checkpoint during training")
# parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint") # just let it be for now. Have not implemented checkpointing
# #parser.add_argument("--test_mode", action="store_true", default=False, help="just leave be, it's for the data loader")

from utils.parse_arguments import custom_parser

def main():
    args = custom_parser()
    # args = parser.parse_args()
    # args.logdir = "/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/" + args.logdir
    # args.test_mode = False
    # if args.debug_mode == "False": args.debug_mode = False
    # elif args.debug_mode == "True": args.debug_mode = True
    # else: raise ValueError("--debug_mode argument is either \"True\" or \"False\"")

    # if args.save_checkpoint == "False": args.save_checkpoint = False
    # elif args.save_checkpoint == "True": args.save_checkpoint = True
    # else: raise ValueError("--save_checkpoint argument is either \"True\" or \"False\"")

    # # Mark debug runs so they are easy to find and delete
    # if args.debug_mode == True: args.logdir += " (debug mode)"

    # # Add an optional short comment to the logdir
    # if args.comment != "": args.logdir += " (" + args.comment + ")"

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True) # What does this do?

    # Get data loaders
    loader, loss_weights = get_loader(args)


    # Should probably enable these when I want to be able to checkpoint the classifier
    # pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    # pretrained_pth = os.path.join(pretrained_dir, model_name)

    # Define classifier
    classifier = Classifier().to(args.cl_device)
    print(f"\nClassifier uses {args.cl_device} device.")

    # Define Feature Extractor
    feature_extractor = EmbedSwinUNETR(
        #img_size=(128, 128, 128),
        #in_channels=4,
        #out_channels=3,
        #feature_size=48,
        #use_checkpoint=True # "use gradient checkpointing to save memory"
    )
    feature_extractor.to(args.cl_device)
    feature_extractor.load_state_dict(torch.load(args.feature_extractor, #"/local/data2/simjo484/BrainSegFounder_custom_finetuning/downstream/BraTS/finetuning/runs/2025-03-05-08:07:48/model_final.pt",
                                                 map_location=args.pp_device)["state_dict"])
    feature_extractor.eval(); print("Set Feature Extractor to eval mode. \N{Nerd Face}")
    print(f"Feature Extractor using {args.cl_device} device.")

    model = piped_classifier(feature_extractor, classifier)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters count: {pytorch_total_params} \N{Abacus} \N{Flexed Biceps}")

    print("Batch size is:", args.batch_size, ". Max epochs:", args.max_epochs)

    best_acc = 0
    start_epoch = 0


    # Enable later when we want to  use checkpointing
    #
    # if args.checkpoint is not None:
    #     checkpoint = torch.load(args.checkpoint, map_location="cpu")
    #     from collections import OrderedDict

    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint["state_dict"].items():
    #         new_state_dict[k.replace("backbone.", "")] = v
    #     model.load_state_dict(new_state_dict, strict=False)
    #     if "epoch" in checkpoint:
    #         start_epoch = checkpoint["epoch"]
    #     if "best_acc" in checkpoint:
    #         best_acc = checkpoint["best_acc"]
    #     print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    
    # 
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    
    loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=loss_weights) # This is apparently the NLLLoss! (Negative log likelihood. Range is [0 -> +inf)  )
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr)



    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_fn,
        #acc_func=dice_acc,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        feature_extractor=feature_extractor
    )

    ############# Create confusion matrices ###########
    loader, loss_weights = get_loader(args)
    
    all_preds = []
    all_targets = []
    for batch_id, batch_data in enumerate(loader[0]):
        data, target = batch_data["images"].to(args.cl_device), batch_data["label"].to(args.cl_device)
        # Extract features, calc predictions
        #data = feature_extractor(data)
        pred = model(data)

        # Save preds and targets
        all_preds += pred.argmax(1).tolist()
        all_targets += target.tolist()
    train_mat = get_conf_matrix(all_targets=all_targets, all_preds=all_preds)

    all_preds = []
    all_targets = []
    for batch_id, batch_data in enumerate(loader[1]):
        data, target = batch_data["images"].to(args.cl_device), batch_data["label"].to(args.cl_device)
        # Extract features, calc predictions
        #data = feature_extractor(data)
        pred = model(data)

        # Save preds and targets
        all_preds += pred.argmax(1).tolist()
        all_targets += target.tolist()
    valid_mat = get_conf_matrix(all_targets=all_targets, all_preds=all_preds)

    # Create and save confusion matrix figure
    create_conf_matrix_fig(train_mat=train_mat, valid_mat=valid_mat, save_fig_as=args.logdir+"/conf_matrices")

    return accuracy


if __name__ == "__main__":
    main()
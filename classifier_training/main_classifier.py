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

from utils.parse_arguments import custom_parser

def main():
    args = custom_parser()
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True) # What does this do?

    #### GET Data Loader (and loss weights)
    loader, loss_weights = get_loader(args)


    # Should probably enable these when I want to be able to checkpoint the classifier
    # pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    # pretrained_pth = os.path.join(pretrained_dir, model_name)

    # Define classifier
    classifier = Classifier().to(args.cl_device)
    print(f"\nClassifier uses {args.cl_device} device.")

    # Define Feature Extractor
    feature_extractor = EmbedSwinUNETR()
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

    
    #### DEFINE Optimizer
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
    
    loss_fn = nn.CrossEntropyLoss(reduction="mean", weight=loss_weights)


    #### DEFINE Learning Rate Scheduler
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
    

    #### RUN Training Loop
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

    return accuracy


if __name__ == "__main__":
    main()
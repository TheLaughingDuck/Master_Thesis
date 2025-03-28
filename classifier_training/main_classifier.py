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
    #### Parse the arguments
    args = custom_parser(terminal=True)
    
    #np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True) # What does this do?


    # Should probably enable these when I want to be able to checkpoint the classifier
    # pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    # pretrained_pth = os.path.join(pretrained_dir, model_name)


    ####################################
    ## V ##   MODEL DEFINITION   ## V ##
    ####################################

    model = Combined_model(feature_extractor_weights="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-27-13:20:53/model_final.pt") #args.feature_extractor)
    model.freeze()#blocks=args.freeze_blocks)

    ####################################
    ## ^ ##   MODEL DEFINITION   ## ^ ##
    ####################################

    #### ARGUMENT PRINTOUT
    print("\n\n##############################################\n")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters count: {format(pytorch_total_params, ",").replace(",", ".")} \N{Abacus} \N{Flexed Biceps}")
    print(f"Model uses {args.cl_device} device.")
    print("Batch size is:", args.batch_size, ". Max epochs:", args.max_epochs)
    print("\n##############################################\n\n")

    # Used for checkpointing
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



    ###########################################################
    ## V ##   DEFINE OPTIMIZER AND SCHEDULER AND LOSS   ## V ##
    ###########################################################
    
    #### OPTIMIZER
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))


    #### Learning Rate SCHEDULER
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
        print("\nUsing Warmup cosine learning rate")
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9)
        print("Using Reduce on Plateau scheduler")
        
    elif args.lrschedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
        print(f"\n\nUsing constant learning rate {args.optim_lr}.\n\n")
    else:
        scheduler = None
    
    
    #### DATA LOADER
    loader, loss_weights = get_loader(args) #Classes are Gli (0), Epe (1), Med (2).
    print(f"\nThe loss weights are: {loss_weights}\n")

    #### LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss(reduction="sum", weight=loss_weights)

    ###########################################################
    ## ^ ##   DEFINE OPTIMIZER AND SCHEDULER AND LOSS   ## ^ ##
    ###########################################################
    

    #### RUN Training Loop
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_fn,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch
    )

    return accuracy


if __name__ == "__main__":
    main()
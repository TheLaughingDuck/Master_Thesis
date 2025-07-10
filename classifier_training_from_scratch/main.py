'''
Script that trains a classifier.
'''

import torch
import torch.nn as nn
import torchvision

import argparse
import os
from datetime import datetime

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis/")


# Modules
from model import Classifier
from data_loader import get_loader
from trainer import run_training
from resnet_model import resnet3d_18



#### PARSER ####
parser = argparse.ArgumentParser(description="Training Arguments")
parser.add_argument("--reg_weight", type=float, default=0.0001)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lrschedule", type=str, default="constant")
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--val_every", type=int, default=1)
#parser.add_argument("--save_every", type=int, default=5)
parser.add_argument("--logdir", type=str, default="/local/data2/simjo484/Training_outputs/classifier_training/from_scratch/", help="The super directory where the dir for this training should be saved.") #Don't change it, or change it very carefully
parser.add_argument("--comment", type=str)
parser.add_argument("--debug_mode", action="store_true")
parser.add_argument("--workers", type=int, default=18, help="Number of CPU workers. Mainly for data loading.")
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--T2", action="store_true") # add options for other modalities later
parser.add_argument("--T1GD_and_T2", action="store_true")
parser.add_argument("--T1_and_T1GD_and_T2", action="store_true")



def main():
    #### PARSE ARGUMENTS
    args = parser.parse_args()
    
    # Process arguments
    args.logdir += datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if args.debug_mode == True: args.logdir += " (debug mode)" # Mark debug runs so they are easy to find and delete
    if (args.comment != "" and args.comment is not None): args.logdir += " (" + args.comment + ")"

    #### Determine sequence combination
    if sum([args.T2, args.T1GD_and_T2, args.T1_and_T1GD_and_T2]) != 1:
        raise ValueError("Only one of the sequence specifiers may be used.")
    else:
        if args.T2 == True:
            args.in_channels = 1
        elif args.T1GD_and_T2 == True:
            args.in_channels = 2
        elif args.T1_and_T1GD_and_T2 == True:
            args.in_channels = 3 

    #### DEFINE MODEL
    #model = Classifier(args.in_channels)
    model = resnet3d_18(num_classes=3, in_channels=args.in_channels)

    # print("\n\n==========================================")
    # print(model)
    # print("==========================================\n\n")

    #### DEFINE OPTIMIZER
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.reg_weight)

    #### Learning Rate SCHEDULER
    if args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    elif args.lrschedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9)
        print("Using Reduce on Plateau scheduler")
        
    elif args.lrschedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
        print(f"\n\nUsing constant learning rate {args.lr}.\n\n")
    else:
        scheduler = None

    #### LOAD DATA
    loaders, loss_weights = get_loader(args)

    #### LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss(reduction="sum", weight=loss_weights)

    #### START TRAINING
    run_training(
        model=model,
        train_loader=loaders[0],
        valid_loader=loaders[1],
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        args=args)




if __name__ == "__main__":
    main()

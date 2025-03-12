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
from utils.data_utils import get_loader

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

parser = argparse.ArgumentParser(description="Classifier pipeline")

# Arguments definetly used by the Classifier
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--logdir", default=".", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--batch_size", default=1, type=int, help="Number of observations per batch")
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--pp_device", default="cpu", type=str, help="Preprocessing device")
parser.add_argument("--cl_device", default="cuda", type=str, help="Classifier device")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--debug_mode", default="False", type=str, help="Set the pipeline into debug mode: only a few observations are used to achieve massive speedup.")


def main():
    args = parser.parse_args()
    args.logdir = "/local/data2/simjo484/Training_outputs/classifier_training/t2/runs" + args.logdir
    args.test_mode = False
    if args.debug_mode == "False": args.debug_mode = False
    elif args.debug_mode == "True": args.debug_mode = True
    else: raise ValueError("--debug_mode argument is either \"True\" or \"False\"")

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True) # What does this do?

    # Get data loaders
    loader = get_loader(args)


    # Should probably enable these when I want to be able to checkpoint the classifier
    # pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    # pretrained_pth = os.path.join(pretrained_dir, model_name)

    # Define classifier
    model = Classifier().to(args.cl_device)
    #model = Detective_Classifier().to(args.cl_device)
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
    feature_extractor.load_state_dict(torch.load("/local/data2/simjo484/BrainSegFounder_custom_finetuning/downstream/BraTS/finetuning/runs/2025-03-05-08:07:48/model_final.pt",
                                        map_location=args.pp_device)["state_dict"])
    feature_extractor.eval(); print("Set Feature Extractor to eval mode.")
    print(f"Feature Extractor using {args.cl_device} device.")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count:", pytorch_total_params)

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
    # if args.optim_name == "adam":
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    # elif args.optim_name == "adamw":
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    # elif args.optim_name == "sgd":
    #     optimizer = torch.optim.SGD(
    #         model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
    #     )
    # else:
    #     raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr)

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_fn,
        #acc_func=dice_acc,
        args=args,
        start_epoch=start_epoch,
        feature_extractor=feature_extractor
    )
    return accuracy


if __name__ == "__main__":
    main()
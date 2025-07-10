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
from model import Combined_model, Custom_CNN_Classifier
from data_loader import get_loader
from trainer import run_training

# Future
config = {
    "reg_weight": 0.0001,
    "lr": 0.000001, #1e-4 is too large.
    "lrschedule": "constant",
    "batch_size": 4,
    "max_epochs": 200,
    "val_every": 1,
    "logdir": "/local/data2/simjo484/Training_outputs/classifier_training/on_location/",
    "comment": "",
    "debug_mode": False,
    "workers": 18,
    "device": "cuda",
    "model": "custom_cnn", #bsf_plus_dense

    "tracking": { # Just some things to keep track of, like number of model parameters

    }
}



def main():    
    # Process arguments
    config["logdir"] += datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if config["debug_mode"] == True: config["logdir"] += " (debug mode)" # Mark debug runs so they are easy to find and delete
    if (config["comment"] != "" and config["comment"] is not None): config["logdir"] += " (" + config["comment"] + ")"

    #### DEFINE MODEL
    if config["model"] == "bsf_plus_dense":
        model = Combined_model(feature_extractor_weights="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt")
        
        # Also LOAD DATA
        loaders, loss_weights = get_loader(config, sequences = ["T1W-GD", "T1W-GD", "T2W", "T2W"])

    elif config["model"] == "custom_cnn":
        model = Custom_CNN_Classifier(in_channels=2, n_classes=2)

        # Also LOAD DATA
        loaders, loss_weights = get_loader(config, sequences = ["T1W-GD", "T2W"])

    config["tracking"]["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("\n\n==========================================")
    # print(model)
    # print("==========================================\n\n")

    #### DEFINE OPTIMIZER
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["reg_weight"])

    #### Learning Rate SCHEDULER
    if config["lrschedule"] == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    elif config["lrschedule"] == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9)
        print("Using Reduce on Plateau scheduler")
    elif config["lrschedule"] == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
        print(f"\n\nUsing constant learning rate {config["lr"]}.\n\n")
    else:
        scheduler = None

    # #### LOAD DATA
    # loaders, loss_weights = get_loader(config, sequences = ["T1W-GD", "T1W-GD", "T2W", "T2W"])

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
        config=config)




if __name__ == "__main__":
    main()

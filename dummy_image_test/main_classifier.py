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


import re

import os
os.chdir("/home/simjo484/master_thesis/Master_Thesis")
from utils import *

from utils.parse_arguments import custom_parser



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NoisyImageDataset(Dataset):
    def __init__(self, num_images=500, image_size=(1, 100, 100)):
        self.num_images = num_images
        self.image_size = image_size
        self.data = []
        self.labels = []

        for i in range(num_images):
            label = i % 2  # Alternate between class 0 and 1
            base_brightness = 0.2 if label == 0 else 0.8
            noise = np.random.normal(loc=0.0, scale=0.1, size=image_size)
            image = base_brightness + noise
            image = np.clip(image, 0.0, 1.0)  # Keep pixel values in range [0, 1]
            self.data.append(torch.tensor(image, dtype=torch.float32))
            self.labels.append(label)
        
        print(f"LABELELES: {self.labels}")
        self.labels = torch.tensor(self.labels)
        print(f"LABELELES: {self.labels}")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return {"image":self.data[idx], "label":self.labels[idx]}



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(100**2, 10),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(10, 2)#,
            #nn.ReLU(),

            # nn.Dropout(p=0.4),
            # nn.Linear(300, 3)#,

            #nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


# The config dict is here just acting as a way to save some information about the training, for example the number of trainable parameters.
config = {

}

def main():
    #### Parse the arguments
    args = custom_parser(terminal=True)

    args.batch_size = 1
    args.val_every = 1
    args.optim_lr = 1e-5
    
    #np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True) # What does this do?


    # Should probably enable these when I want to be able to checkpoint the classifier
    # pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    # pretrained_pth = os.path.join(pretrained_dir, model_name)


    ####################################
    ## V ##   MODEL DEFINITION   ## V ##
    ####################################

    #model = Combined_model(feature_extractor_weights="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-27-13:20:53/model_final.pt") #args.feature_extractor)
    
    model = Classifier()
    model.to("cpu")
    
    #model = Combined_model(feature_extractor_weights="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt") #args.feature_extractor)
    #model.freeze(args)#blocks=args.freeze_blocks)

    #config["FE_parameters"] = sum(p.numel() for p in model.feature_extractor.parameters() if p.requires_grad)
    #config["classifier_parameters"] = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
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
    
    
    train_loader = NoisyImageDataset()
    val_loader = NoisyImageDataset()

    #### LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    ###########################################################
    ## ^ ##   DEFINE OPTIMIZER AND SCHEDULER AND LOSS   ## ^ ##
    ###########################################################
    

    #### RUN Training Loop
    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_fn,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        config=config
    )

    return accuracy


if __name__ == "__main__":
    main()
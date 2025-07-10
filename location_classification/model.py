'''
Module for defining the classifier.
'''

import torch
import torch.nn as nn


from torch import nn
import numpy as np
from monai.networks.nets import SwinUNETR
import torch


import torch
from torch.nn import Threshold

import numpy as np
import nibabel as nib

from utils import *

import matplotlib.pyplot as plt

from monai.networks.nets import SwinUNETR
from monai import data
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    NormalizeIntensity,
    Resize,
    ToTensor,
    Compose,
    Rotate90,
    Lambda
)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(768, 10),
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




# Define Feature Extractinator
class EmbedSwinUNETR(SwinUNETR):
    '''
    Class that represents the 4-channel BSF architecture intended for BraTS.
    '''

    def __init__(self, **args):
        super(EmbedSwinUNETR, self).__init__(
            img_size=(128, 128, 128), # Essentially trivial
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
            **args)
        
        #self.load_state_dict(torch.load(weights, map_location=device))

    def forward(self, x_in):

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
    
        out = self.swinViT(x_in, self.normalize)[4]

        # Perform global average pooling to reduce shape from (3,768,4,4,4) to (3,768)
        # (If the batch size is 3).
        # batch_size = x_in.shape[0]
        # out = torch.nn.AvgPool3d((4,4,4))(out).view(batch_size, 768)
        return out



class Combined_model(torch.nn.Module):
    def __init__(self, feature_extractor_weights="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-05-08:07:48/model_final.pt",
                 device="cuda"):
        super(Combined_model, self).__init__()

        # SETUP Feature Extractor
        self.feature_extractor = EmbedSwinUNETR()
        #weights_default_val = 
        self.feature_extractor.load_state_dict(torch.load(feature_extractor_weights, map_location=device)["state_dict"])
        
        self.classifier = Classifier()

        # Settings
        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = self.feature_extractor(x)

        # Perform global average pooling to reduce shape from (3,768,4,4,4) to (3,768)
        # (If the batch size is 3).
        batch_size = x.shape[0]
        x = torch.nn.AvgPool3d((4,4,4))(x).view(batch_size, 768)

        x = self.classifier(x)
        return x
    
    def forward_test(self, batch_size=3):
        '''A method for testing the model on a random input.'''
        x_in = torch.randn(batch_size,4,128,128,128).to(self.device)
        
        return(self.forward(x_in))
    
    def freeze(self):
        # First freeze all parameters, so that old parameters that are just hanging on get frozen.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        print("\n\n==============================================")
        print("Froze ALL parameters in Feature Extractor.")

        # Cycle through all the 4 modules of the SwinViT
        module_count = 0
        for ch in self.feature_extractor.swinViT.children():
            if module_count in [3, 4]:
                print(f"Unfreezing SwinViT module {module_count}")
                for param in ch.parameters():
                    param.requires_grad = True
            
            module_count += 1
    
        n_params = lambda x: format(sum(p.numel() for p in x.parameters() if p.requires_grad), ",").replace(",", ".") # Format like 10.000.000
        print(f"\nFeature extractor parameters: {n_params(self.feature_extractor)}")
        print(f"Classifier parameters: {n_params(self.classifier)}")
        print(f"Total parameters count: {n_params(self)} \N{Abacus} \N{Flexed Biceps}")
        print("==============================================\n\n")

    # def freeze(self, blocks:int=0): # See this helpful link: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    #     # First freeze all parameters, so that old parameters that are just hanging on get frozen.
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = False
    #     print("\n\n==============================================")
    #     print("Froze ALL parameters in Feature Extractor.")

    #     # Then unfreeze the specific blocks we want
    #     child_counter = 0
    #     for child in self.feature_extractor.children():
    #         #print(f"################### Child {child_counter} is {child}")

    #         if child_counter == 0: # This represents the 
    #             block_counter = 0
    #             for block in child.children():
    #                 if not block_counter < blocks:
    #                     print(f"Unfreezing block {block_counter} of Child {child_counter} in Feature Extractor.")
    #                     for param in block.parameters():
    #                         param.requires_grad = True

    #                 block_counter += 1
    #         child_counter += 1

    #     n_params = lambda x: format(sum(p.numel() for p in x.parameters() if p.requires_grad), ",").replace(",", ".") # Format like 10.000.000
    #     print(f"\nFeature extractor parameters: {n_params(self.feature_extractor)}")
    #     print(f"Classifier parameters: {n_params(self.classifier)}")
    #     print(f"Total parameters count: {n_params(self)} \N{Abacus} \N{Flexed Biceps}")
    #     print("==============================================\n\n")


class Lone_Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2*128**3, 100),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(100, 10),
            nn.ReLU(),

            nn.Dropout(p=0.5),
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

















class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_channels)
        )
    
    def forward(self, x):
        return self.block(x)



class Custom_CNN_Classifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Custom_CNN_Classifier, self).__init__()

        self.ConvSequence = nn.Sequential(
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (10,10,10)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (10,10,10)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (10,10,10)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (8,8,8)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (5,5,5)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (5,5,5)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (5,5,5)),
            nn.Dropout(p=0.4),
            ConvBlock(2, 2, (5,5,5))
        )
        # self.net = nn.Sequential(a)

        # TESTING
        self.flatten = nn.Flatten()
        self.ConvBlock = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=2, kernel_size=(10,10,10)),
            nn.ReLU()
        )
        self.lin_rel_stack = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(93312, 10),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(10, n_classes)
        )

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameter count: {format(n_params, ",").replace(",", ".")} \N{Abacus}.\n")

        # Put the model on the gpu. This should be at the end of the class __init__
        self.device = "cuda"
        self.to(self.device)
        print(f"Model uses {self.device} device.")
    
    def forward(self, x):
        x = self.ConvSequence(x)
        #print(f"x shape: {x.shape}")
        x = self.flatten(x)
        logits = self.lin_rel_stack(x)
        return(logits)
        
        return self.net(x)
'''
This file provides setup like classes that are necessary for the training of a classifier.
'''

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
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(768*4**3, 700),
            nn.ReLU(),

            nn.Dropout(p=0.4),
            nn.Linear(700, 300),
            nn.ReLU(),

            nn.Dropout(p=0.4),
            nn.Linear(300, 3)#,

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
        super().__init__(
            img_size=(128, 128, 128), # Essentially trivial
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
            **args) 

    def forward(self, x_in):

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
    
        hidden_states_out = self.swinViT(x_in, self.normalize)

        return hidden_states_out[4]
        #enc0 = self.encoder1(x_in)
        #enc1 = self.encoder2(hidden_states_out[0])
        #enc2 = self.encoder3(hidden_states_out[1])
        #enc3 = self.encoder4(hidden_states_out[2])
        #dec4 = self.encoder10(hidden_states_out[4])

        # Just return this embedding
        #return(dec4)
        
        # dec3 = self.decoder5(dec4, hidden_states_out[3])
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0, enc0)
        # logits = self.out(out)
        # return logits


class piped_classifier(torch.nn.Module):
    def __init__(self, model1, model2):
        super(piped_classifier, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x

class Combined_model(torch.nn.Module):
    def __init__(self, feature_extractor_weights, device="cuda"):
        super(Combined_model, self).__init__()

        # SETUP Feature Extractor
        self.feature_extractor = EmbedSwinUNETR()
        self.feature_extractor.load_state_dict(torch.load(feature_extractor_weights, map_location=device)["state_dict"])
        
        self.classifier = Classifier()

        # Settings
        self.to(device)
    
    def freeze(self, blocks:int): # See this helpful link: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
        child_counter = 0
        for child in self.feature_extractor.children():
            #print(f"################### Child {child_counter} is (...)")
            if child_counter == 0:
                block_counter = 0
                for block in child.children():
                    if block_counter < blocks:
                        print(f"Freezing block {block_counter} of Child {child_counter} in Feature Extractor.")
                        for param in block.parameters():
                            param.requires_grad = False

                    block_counter += 1
            
            # End of main child loop
            child_counter += 1

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

#%%
from torcheval.metrics.functional import multiclass_accuracy, multiclass_confusion_matrix, multiclass_precision, multiclass_recall
from utils.visualization_utils import get_conf_matrix, create_conf_matrix_fig

def get_metrics(all_preds:list, all_targets:list, num_classes:int, args, epoch:int, conf_matr_title:str):
    '''
    Function that takes two torch tensors; all_preds, and all_targets,
    and calculates various performance metrics. Returns a dict structure.
    '''
    # Create and save confusion matrix
    conf_matrix = get_conf_matrix(all_preds=all_preds.tolist(), all_targets=all_targets.tolist())
    create_conf_matrix_fig(conf_matrix, save_fig_as=args.logdir+"/validation_matrix", epoch=epoch, title=conf_matr_title)

    acc = multiclass_accuracy(all_preds, target=all_targets, num_classes=num_classes, average="micro")
    prec = multiclass_precision(all_preds, target=all_targets, num_classes=num_classes, average=None)
    rec = multiclass_recall(all_preds, target=all_targets, num_classes=num_classes, average=None)

    metrics = {"acc": acc, "prec": prec, "rec": rec}

    return metrics



# Likely deprecated
def generate_data(paths: list = None, device: str = None):
    '''
    Takes a list structure full of filenames for the data. Returns a generator that can be used like:
    "
    for obs in data_generator(paths):
        # Model stuff
    "

    ### Parameters
        paths: A list, like
            [
                [['patient1_t1.nii.gz', 'patient1_t1ce.nii.gz', 'patient1_t2.nii.gz', 'patient1_flair.nii.gz']],
                [['patient2_t1.nii.gz', 'patient2_t1ce.nii.gz', 'patient2_t2.nii.gz', 'patient2_flair.nii.gz']],
                [['patient3_t1.nii.gz', 'patient3_t1ce.nii.gz', 'patient3_t2.nii.gz', 'patient3_flair.nii.gz']]
            ]
        
        device: A string, specifying the current device. For example "cpu", "cuda:0".
    '''

    transforms = Compose([
        LoadImage(image_only=True),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        Resize(spatial_size=(128, 128, 128)),
        #EnsureType(),
        Rotate90(k=3, spatial_axes=(0,1)),
        #EnsureType(),
        ToTensor(track_meta=False)#,
    ])

    dataset = data.Dataset(data=paths, transform=transforms)
    val_loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True
    )

    return val_loader
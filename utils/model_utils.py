'''
Helper utils for setting up the model in a training pipeline.
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
            nn.Linear(768, 10),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(10, 3)#,
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

class Feature_extractor(SwinUNETR): # Not used
    def __init__(self):
        super(Feature_extractor, self).__init__(
            img_size=(128, 128, 128), # Essentially trivial
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True)
    
    def forward(self, x):
        x = self.swinViT(x, self.normalize)[4]
        return(x)



class piped_classifier(torch.nn.Module): #Not used
    def __init__(self, model1, model2):
        super(piped_classifier, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x

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
            if module_count in [3]:
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
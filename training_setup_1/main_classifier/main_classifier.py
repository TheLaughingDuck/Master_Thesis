'''
This script is designed to be'''

import argparse
import os
from functools import partial

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.distributed as dist # type: ignore
import torch.multiprocessing as mp # type: ignore
import torch.nn.parallel # type: ignore
import torch.utils.data.distributed # type: ignore
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference # type: ignore
from monai.losses import DiceLoss # type: ignore
from monai.metrics import DiceMetric # type: ignore
from monai.networks.nets import SwinUNETR # type: ignore
from monai.transforms import Activations, AsDiscrete, Compose # type: ignore
from monai.utils.enums import MetricReduction # type: ignore

# Definetly used by the T2 Classifier
from utils.utils import Classifier, EmbedSwinUNETR, Detective_Classifier, Detective_EmbedSwinUNETR

parser = argparse.ArgumentParser(description="Classifier pipeline")

# Arguments definetly used by the Classifier
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--logdir", default=".", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--n_batches", default=1, type=int, help="number of batches")
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--pp_device", default="cpu", type=str, help="Preprocessing device")
parser.add_argument("--cl_device", default="cuda", type=str, help="Classifier device")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--debug_mode", default="False", type=str, help="Set the pipeline into debug mode: only a few observations are used to achieve massive speedup.")


# Arguments that maybe could be discarded
# parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
# parser.add_argument("--data_dir", default="/dataset/brats2021/", type=str, help="dataset directory")
# parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")

# parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
# parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
# parser.add_argument("--momentum", default=0.99, type=float, help="momentum")

# parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")

# #parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class")
# #parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
# parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
# parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
# parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
# parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
# parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
# parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
# parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")

# parser.add_argument(
#     "--pretrained_dir",
#     default="/local/data2/simjo484/BrainSegFounder_models/BraTS/ssl",#"""./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/",
#     type=str,
#     help="pretrained checkpoint directory",
# )
#parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


def main():
    args = parser.parse_args()
    args.logdir = "/local/data2/simjo484/Classifier_training/t2/runs/" + args.logdir
    args.test_mode = False
    if args.debug_mode == "False": args.debug_mode = False
    elif args.debug_mode == "True": args.debug_mode = True
    else: raise ValueError("--debug_mode argument is either \"True\" or \"False\"")

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True) # What does this do?

    # Set preprocessing and classifier devices
    # They are now arguments
    #pp_device = "cpu"
    #cl_device = "cuda"

    loader = get_loader(args)

    print("\nNumber of batches is:", args.n_batches, ". Max epochs:", args.max_epochs, "\n")

    # Should probably enable these when I want to be able to checkpoint the classifier
    # pretrained_dir = args.pretrained_dir
    # model_name = args.pretrained_model_name
    # pretrained_pth = os.path.join(pretrained_dir, model_name)

    # Define classifier
    model = Classifier().to(args.cl_device)
    #model = Detective_Classifier().to(args.cl_device)
    print(f"Classifier uses {args.cl_device} device.")

    # Define Feature Extractor
    feature_extractor = EmbedSwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True # "use gradient checkpointing to save memory"
    )
    feature_extractor.to(args.pp_device)
    feature_extractor.load_state_dict(torch.load("/local/data2/simjo484/BrainSegFounder_custom_finetuning/downstream/BraTS/finetuning/runs/2025-03-05-08:07:48/model_final.pt",
                                        map_location=args.pp_device)["state_dict"])
    feature_extractor.eval(); print("Set Feature Extractor to eval mode.")
    print(f"Feature Extractor using {args.pp_device} device")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count:", pytorch_total_params)

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

    

    # Maybe I want to use multiple optimisation procedures?
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
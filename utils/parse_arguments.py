'''
This script has a function that does all the
parsing of arguments for training the classifier
pipeline.
'''

#%%
# SETUP
import argparse
import os
from datetime import datetime


def custom_parser(terminal=False):
    '''
    A function that returns an argparse.Namespace object.

    It would be nice to develop this func further, to take kwargs, that can be added to the argparse.Namespace object,
    and that their default values match up with the ones below in the ArgumentParser part, but I don't have the time to fix that at the moment.
    
    terminal: whether the main script is run in a terminal, meaning there are arguments that need to be parsed,
            or if it is run in a jupyter notebook, meaning potential arguments need to be specified as keyword arguments instead.
    '''

    if terminal == False:
        class Args(argparse.Namespace):
            logdir = ""
            optim_lr = 1e-4
            reg_weight = 1e-4
            roi_x = 128
            roi_y = 128
            roi_z = 128
            distributed = False
            workers = 18
            data_dir='/local/data2/simjo484/BRATScommon/BRATS21/'
            json_list = "./jsons/brats21_folds.json"
            fold = 4
            test_mode = False
            batch_size = 2
            debug_mode = False
            cl_device = "cuda"
            pp_device = "cpu"
            data_aug_prob = 0.5
        
        args = Args()
        return(args)
    else:
        ######### Create parser
        parser = argparse.ArgumentParser(description="Classifier pipeline")

        ######### Creates arguments
        # Arguments that should be modified
        parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
        parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
        parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
        parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
        parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
        parser.add_argument("--feature_extractor", default="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/2025-03-05-08:07:48/model_final.pt", type=str, help="Path to the fine-tuned feature extractor model weights")
        parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
        #parser.add_argument("--debug_mode", default="False", type=str, help="Set the pipeline into debug mode: only a few observations are used to achieve massive speedup.") # change to store_true=False
        parser.add_argument("--comment", default="", type=str, help="A short comment for the output file, to help distinguish previous runs. Example: \".../runs/2025-XX-XX-XX:XX:XX (Deeper classifier)\"")
        parser.add_argument("--freeze_blocks", default=0, type=int, help="The number of blocks to freeze in the feature extractor, starting at the input.")
        parser.add_argument("--data_aug_prob", default=0.5, type=float, help="The probability used for the data augmentation of the training data.")
        
        parser.add_argument("--T2", action="store_true") # add options for other modalities later
        parser.add_argument("--T1GD_T2", action="store_true")
        parser.add_argument("--T1GD", action="store_true")
        parser.add_argument("--debug_mode", action="store_true")


        # Arguments to probably leave alone
        parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
        parser.add_argument("--batch_size", default=3, type=int, help="Number of observations per batch")
        parser.add_argument("--logdir", default="/local/data2/simjo484/Training_outputs/classifier_training/miscellaneous_runs/runs/", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--workers", default=18, type=int, help="number of workers")
        parser.add_argument("--pp_device", default="cpu", type=str, help="Preprocessing device")
        parser.add_argument("--cl_device", default="cuda", type=str, help="Classifier device")
        parser.add_argument("--save_checkpoint", default="True", help="save checkpoint during training")
        parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint") # just let it be for now. Have not implemented checkpointing
        #parser.add_argument("--test_mode", action="store_true", default=False, help="just leave be, it's for the data loader")
        
        args = parser.parse_args()
    
    ######### Process arguments
    #args.logdir = "/local/data2/simjo484/Training_outputs/classifier_training/t2/runs/" + args.logdir
    args.logdir += datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    args.test_mode = False
    # if args.debug_mode == "False": args.debug_mode = False
    # elif args.debug_mode == "True": args.debug_mode = True
    # else: raise ValueError("--debug_mode argument is either \"True\" or \"False\"")

    if args.save_checkpoint == "False": args.save_checkpoint = False
    elif args.save_checkpoint == "True": args.save_checkpoint = True
    else: raise ValueError("--save_checkpoint argument is either \"True\" or \"False\"")

    # If debug mode, set epochs very low also (to speedup)
    if args.debug_mode:
        args.max_epochs = 2
        args.val_every = 1
    
    # Manual interpretation of the freeze_blocks argument
    if args.freeze_blocks == 0: args.freeze_blocks = []
    if args.freeze_blocks == 1: args.freeze_blocks = [4]
    if args.freeze_blocks == 2: args.freeze_blocks = [3,4]
    if args.freeze_blocks == 3: args.freeze_blocks = [2,3,4]
    if args.freeze_blocks == 4: args.freeze_blocks = [1,2,3,4]
    if args.freeze_blocks == 5: args.freeze_blocks = [0,1,2,3,4]

    # Mark debug runs so they are easy to find and delete
    if args.debug_mode: args.logdir += " (debug mode)"

    # Add an optional short comment to the logdir
    if args.comment != "": args.logdir += " (" + args.comment + ")"

    return args


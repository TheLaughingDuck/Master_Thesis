#!/bin/bash

# To start these jobs, give this file executable permissions, by running:
#chmod +x my_jobs.sh

# Then start the jobs by running:
#./my_jobs.sh

# The echo lines will print to the log, adding on (not overwriting)


#########################################
###   This is for KAWASAKI Training   ###
#########################################

# Paths
path_to_json_list='/home/simjo484/master_thesis/Master_Thesis/BSF_finetuning/jsons/brats21_folds_T2_modality.json'
path_to_data_dir='/local/data2/simjo484/BRATScommon/BRATS21/'
fold=4

#pretrained model
path_to_pretrained_dir="/local/data2/simjo484/BrainSegFounder_models/BraTS/ssl" #"/path/to/pretrained/models/"
path_to_checkpoint_dir='model_bestValRMSE-fold4.pt'
depths="2 2 2 2"  
num_heads="3 6 12 24" 
#training
batch_size=2
optim_lr=1e-4
logdir='/local/data2/simjo484/Training_outputs/BSF_finetuning/runs/'$(date "+%Y-%m-%d-%H:%M:%S")


echo "Started training" >> launch_training.log
python /home/simjo484/master_thesis/Master_Thesis/BSF_finetuning/main_FinetuningSwinUNETR_4Channels.py \
--json_list=$path_to_json_list --data_dir=$path_to_data_dir --val_every=5 --noamp --pretrained_model_name=$path_to_checkpoint_dir \
--pretrained_dir=$path_to_pretrained_dir --fold=$fold --roi_x=128 --roi_y=128 --roi_z=128 --in_channels=4 \
--spatial_dims=3 --use_checkpoint --resume_ckpt --feature_size=48 --depths="$depths" --num_heads="$num_heads" --batch_size=$batch_size \
--optim_lr=$optim_lr --save_checkpoint --logdir=$logdir --freeze
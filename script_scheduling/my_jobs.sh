#!/bin/bash

# To start these jobs, give this file executable permissions, by running:
#chmod +x my_jobs.sh

# Then start the jobs by running:
#./my_jobs.sh

# Note: You can break *just one* of the jobs below, with Ctrl+C. The next job will then begin. Neat!

# The echo lines will print to the log, adding on (not overwriting)

echo "Training 0" >> my_jobs.log
python /home/simjo484/master_thesis/Master_Thesis/classifier_training/main_classifier.py \
--logdir="/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/" \
--feature_extractor="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt" \
--max_epochs=300 \
--lrschedule="warmup_cosine" \
--batch_size=3 \
--optim_lr=5e-5 \
--val_every=5 \
--reg_weight=1e-4 \
--warmup_epochs=50 \
--freeze_blocks=0 \
--comment="T1GD and T2 standardized training (freeze 0)" \
--T1GD_T2


echo "Training 1" >> my_jobs.log
python /home/simjo484/master_thesis/Master_Thesis/classifier_training/main_classifier.py \
--logdir="/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/" \
--feature_extractor="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt" \
--max_epochs=300 \
--lrschedule="warmup_cosine" \
--batch_size=3 \
--optim_lr=5e-5 \
--val_every=5 \
--reg_weight=1e-4 \
--warmup_epochs=50 \
--freeze_blocks=1 \
--comment="T1GD and T2 standardized training (freeze 1)" \
--T1GD_T2



echo "Training 2" >> my_jobs.log
python /home/simjo484/master_thesis/Master_Thesis/classifier_training/main_classifier.py \
--logdir="/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/" \
--feature_extractor="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt" \
--max_epochs=300 \
--lrschedule="warmup_cosine" \
--batch_size=3 \
--optim_lr=5e-5 \
--val_every=5 \
--reg_weight=1e-4 \
--warmup_epochs=50 \
--freeze_blocks=2 \
--comment="T1GD and T2 standardized training (freeze 2)" \
--T1GD_T2



echo "Training 3" >> my_jobs.log
python /home/simjo484/master_thesis/Master_Thesis/classifier_training/main_classifier.py \
--logdir="/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/" \
--feature_extractor="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt" \
--max_epochs=300 \
--lrschedule="warmup_cosine" \
--batch_size=3 \
--optim_lr=5e-5 \
--val_every=5 \
--reg_weight=1e-4 \
--warmup_epochs=50 \
--freeze_blocks=3 \
--comment="T1GD and T2 standardized training (freeze 3)" \
--T1GD_T2




echo "Training 4" >> my_jobs.log
python /home/simjo484/master_thesis/Master_Thesis/classifier_training/main_classifier.py \
--logdir="/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/" \
--feature_extractor="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt" \
--max_epochs=300 \
--lrschedule="warmup_cosine" \
--batch_size=3 \
--optim_lr=5e-5 \
--val_every=5 \
--reg_weight=1e-4 \
--warmup_epochs=50 \
--freeze_blocks=4 \
--comment="T1GD and T2 standardized training (freeze 4)" \
--T1GD_T2




echo "Training 5" >> my_jobs.log
python /home/simjo484/master_thesis/Master_Thesis/classifier_training/main_classifier.py \
--logdir="/local/data2/simjo484/Training_outputs/classifier_training/standardized/runs (freezing)/" \
--feature_extractor="/local/data2/simjo484/Training_outputs/BSF_finetuning/runs_t1gd_and_t2/2025-03-28-23:14:58/model_final.pt" \
--max_epochs=300 \
--lrschedule="warmup_cosine" \
--batch_size=3 \
--optim_lr=5e-5 \
--val_every=5 \
--reg_weight=1e-4 \
--warmup_epochs=50 \
--freeze_blocks=5 \
--comment="T1GD and T2 standardized training (freeze 5)" \
--T1GD_T2
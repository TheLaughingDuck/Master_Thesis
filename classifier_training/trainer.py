'''
Script that provides training functions for a classifier.

This script was adapted from scripts used to train the BrainSegFounder models.
This script was copied and modified in March of 2025. See
https://github.com/lab-smile/BrainSegFounder
for the original source code, that falls under the LICENSE that is also available in this dir.
'''

import os
import pdb
import shutil
import time

import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from utils import *


#############################
###### TRAIN ONE EPOCH ######
#############################
def train_epoch(model, loader, optimizer, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    #num_batches = len(loader)

    all_preds = []
    all_targets = []
    
    for batch_id, batch_data in enumerate(loader):
        data, target = batch_data["images"].to(args.cl_device), batch_data["label"].to(args.cl_device)

        pred = model(data)
        loss = loss_func(pred, target)
        run_loss.update(loss.item(), n=args.batch_size)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save loss
        # run_loss.update(loss.item(), n=args.batch_size)
        #run_loss += loss

        # Save preds and targets (for the full epoch)
        all_preds += pred.argmax(1).tolist()
        all_targets += target.tolist()

        print(
            "Epoch {}/{}, batch {}/{},".format(epoch, args.max_epochs, batch_id, len(loader)),
            "loss: {:.4f},".format(run_loss.avg),#loss.item()),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    
    # Create train set confusion matrix
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    conf_matrix = get_conf_matrix(all_preds=all_preds.tolist(), all_targets=all_targets.tolist())
    create_conf_matrix_fig(conf_matrix, save_fig_as=args.logdir+"/training_matrix", epoch=epoch, title="Training confusion matrix")


    return {"avg_loss": run_loss.avg}


################################
###### VALIDATE ONE EPOCH ######
################################
def val_epoch(model, loader, epoch, loss_func, args):
    model.eval()
    #n_observations = len(loader.dataset)
    #num_batches = len(loader)
    start_time = time.time()
    run_loss = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(loader):
            data, target = batch_data["images"].to(args.cl_device), batch_data["label"].to(args.cl_device)

            # Calculate predicitons and loss
            pred = model(data)
            loss = loss_func(pred, target) # Calculate loss
            run_loss.update(loss.item(), n=args.batch_size)
            #run_loss += val_loss

            # Save preds and targets (for the full epoch)
            all_preds += pred.argmax(1).tolist()
            all_targets += target.tolist()

            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, batch_id, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    
    # Calculate metrics
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    
    metrics = get_metrics(all_preds=all_preds, all_targets=all_targets, num_classes=3, args=args, epoch=epoch, conf_matr_title="Validation confusion matrix") # this also makes conf matrices now.

    return run_loss.avg, metrics


#############################
###### SAVE CHECKPOINT ######
#############################
def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("\nSaving checkpoint", filename)


###############################
###### RUN TRAINING LOOP ######
###############################
def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    args,
    scheduler=None,
    start_epoch=0
    ):

    # Setup TrainingTracker object
    TT = TrainingTracker(args)
    
    # Setup log writer
    writer = None
    if args.logdir is not None:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing Tensorboard logs to ", args.logdir)

        # Print arguments to run dir (currently collides with inspect_progress.py)
        with open(args.logdir + "/arguments.txt", "w") as f:
            #args = parser.parse_args()
            f.write("=== Arguments ===\n")
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
            f.write("=================\n")

    # The best validation accuracy so far
    val_acc_max = 0

    # The training loop!
    for epoch in range(start_epoch, args.max_epochs):
        print("\n===", time.ctime(), "Epoch:", epoch, "===")
        epoch_time = time.time()

        # Run one training epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, args=args
        )

        # Format the estimated end time
        epoch_duration = time.time() - epoch_time
        estimated_end_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time() + (args.max_epochs-epoch) * epoch_duration))
        
        # Print results of one training epoch
        print(
            "\nFinal training results: epoch  {}/{},".format(epoch, args.max_epochs - 1),
            "avg loss: {:.4f},".format(train_metrics["avg_loss"]),
            "time: {:.2f}s".format(time.time() - epoch_time),
            "\nEstimated completion on: {} (not taking validation epochs into account)".format(estimated_end_time)
        )
        
        # Write results of one training epoch
        if writer is not None:
            writer.add_scalar("avg_train_loss", train_metrics["avg_loss"], epoch)
            TT.update_epoch({"avg_train_loss":{"step":[epoch],"value":[float(train_metrics["avg_loss"])]}})
        b_new_best = False

        # Possibly run a validation epoch
        if (epoch + 1) % args.val_every == 0:
            print('\n=== Validation ===')
            epoch_time = time.time()

            avg_loss, val_metrics = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                loss_func=loss_func,
                args=args
            )

            print(
                "\nFinal validation stats {}/{}, time: {:.2f}s \N{Dragon Face}\n".format(epoch, args.max_epochs - 1, time.time() - epoch_time),
                "\tAccuracy (global, unweighted): {:>0.1f}".format(val_metrics["acc"]),
                "\n\tPrecision (by class): {:>0.1f}, {:>0.1f}, {:>0.1f}".format(*val_metrics["prec"].tolist()),
                "\n\tRecall (by class): {:>0.1f}, {:>0.1f}, {:>0.1f}".format(*val_metrics["rec"].tolist()),
                "\n\tAvg val loss: {:>8f} \N{Whale}".format(avg_loss)
            )

            if writer is not None:
                # Here we take "avg" to mean across all observations in the epoch,
                # and "mean" to mean across the different diagnoses/groups
                writer.add_scalar("avg_val_loss", avg_loss, epoch)
                writer.add_scalar("acc", val_metrics["acc"], epoch)
                TT.update_epoch({"avg_valid_loss":{"step":[epoch],"value":[float(avg_loss)]}})
                TT.update_epoch({"acc_glob_unweighted":{"step":[epoch],"value":[float(val_metrics["acc"])]}})

                for label in [0,1,2]:
                    writer.add_scalar("prec_class_"+str(label), val_metrics["prec"].tolist()[label], epoch)
                    writer.add_scalar("rec_class_"+str(label), val_metrics["rec"].tolist()[label], epoch)
                    TT.update_epoch({"prec_class_"+str(label):{"step":[epoch],"value":[val_metrics["prec"].tolist()[label]]}})
                    TT.update_epoch({"rec_class_"+str(label):{"step":[epoch],"value":[val_metrics["rec"].tolist()[label]]}})

            if val_metrics["acc"] > val_acc_max:
                print("New best val accuracy ({:.6f} --> {:.6f}). ".format(val_acc_max, val_metrics["acc"]))
                val_acc_max = val_metrics["acc"]
                b_new_best = True
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_metrics["acc"], optimizer=optimizer, scheduler=scheduler
                    )
            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_metrics["acc"], filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!! \N{Sauropod} \N{T-Rex} \N{Crocodile} \N{Spouting Whale} \N{Dragon}")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
        
        # At the end of an epoch, save the metrics
        TT.update_epoch({"learning_rate":{"step":[epoch],"value":[scheduler.get_last_lr()[0]]}})
        TT.make_key_fig(["avg_train_loss", "avg_valid_loss"], kwargs={"avg_train_loss": {"color": "blue", "label": "Training"}, "avg_valid_loss": {"color": "orange", "label": "Validation"}}, title="CrossEntropy loss")
        
        # Change to next learning rate value
        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best validation accuracy: {:>0.1f}".format(val_acc_max.item()))
    print("\N{Sauropod} \N{Sauropod}")

    TT.make_key_fig(["avg_train_loss", "avg_valid_loss"], kwargs={"avg_train_loss": {"color": "blue", "label": "Training"}, "avg_valid_loss": {"color": "orange", "label": "Validation"}}, title="CrossEntropy loss")
    TT.make_key_fig(["learning_rate"], title="Learning rate")
    TT.make_key_fig(["acc_glob_unweighted"], title="Acc. (glob. unweighted)")
    TT.to_json()

    return val_acc_max

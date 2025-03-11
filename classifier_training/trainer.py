import os
import pdb
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils import *

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, epoch, loss_func, args, feature_extractor=None):
    model.train()
    # n_observations = len(loader.dataset)
    num_batches = len(loader)
    
    run_loss = 0 #AverageMeter()
    start_time = time.time()

    for batch_id, batch_data in enumerate(loader):
        sectional_start_time = time.time()
        data, target = batch_data["images"].to(args.cl_device), batch_data["label"].to(args.cl_device)
        #print("\tData loading time: {:.2f}s".format(time.time()-sectional_start_time))

        # Extract features
        sectional_start_time = time.time()
        data = feature_extractor(data) # Taking a lot of time
        #print("\tFeature extraction time: {:.2f}s".format(time.time()-sectional_start_time))
        
        sectional_start_time = time.time()
        pred = model(data)
        loss = loss_func(pred, target)
        #print("\tClassifier forward pass (and loss calc) time: {:.2f}s".format(time.time()-sectional_start_time))
        

        # Backpropagation
        sectional_start_time = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print("\tBackprop time: {:.2f}s".format(time.time()-sectional_start_time))
        

        # Save loss
        # run_loss.update(loss.item(), n=args.batch_size)
        run_loss += loss

        print(
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, batch_id, len(loader)),
            "loss: {:.4f}".format(loss),#loss.item()),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    
    avg_loss = run_loss / num_batches

    return {"avg_loss": avg_loss} # avg_loss #run_loss.item()


def val_epoch(model, loader, epoch, loss_func, args, feature_extractor):
    model.eval()
    n_observations = len(loader.dataset)
    num_batches = len(loader)
    val_loss, correct = 0, 0
    start_time = time.time()
    # run_loss = AverageMeter()
    run_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(loader):
            data, target = batch_data["images"].to(args.cl_device), batch_data["label"].to(args.cl_device)

            # Extract features
            data = feature_extractor(data)

            # Calculate predicitons and loss
            pred = model(data)
            val_loss = loss_func(pred, target).item() # Calculate loss
            run_loss += val_loss

            # Save preds and targets (for the full epoch)
            all_preds += pred.argmax(1).tolist()
            all_targets += target.tolist()

            #correct += (pred.argmax(1) == target).type(torch.float).sum().item()

            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, batch_id, len(loader)),
                "loss: {:.4f}".format(val_loss),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    
    # Calculate metrics
    avg_loss = run_loss / num_batches
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    
    metrics = get_metrics(all_preds, all_targets, num_classes=3)

    # mean_acc = multiclass_accuracy(all_preds, target=all_targets, num_classes=3, average="macro")
    # mean_prec = multiclass_precision(all_preds, target=all_targets, num_classes=3, average="macro")
    # mean_rec = multiclass_recall(all_preds, target=all_targets, num_classes=3, average="macro")
    #conf_mat = multiclass_confusion_matrix(all_preds, target=all_targets, num_classes=3)
    
    #print(f"Validation Error: \n Accuracy: {(100*correct/n_observations):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    #return {"avg_loss": avg_loss, "acc": (100*correct/n_observations)} #avg_loss #run_loss.avg
    #return {"avg_loss": avg_loss, "mean_acc": mean_acc, "mean_prec": mean_prec, "mean_rec": mean_rec}
    return avg_loss, metrics


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("\nSaving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    #acc_func,
    args,
    start_epoch=0,
    feature_extractor=None
    ):
    
    writer = None
    if args.logdir is not None:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing Tensorboard logs to ", args.logdir)

    # The best validation accuracy so far
    val_acc_max = 0

    # The training loop!
    for epoch in range(start_epoch, args.max_epochs):
        print("\n===", time.ctime(), "Epoch:", epoch, "===")
        epoch_time = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, args=args, feature_extractor=feature_extractor
        )
        
        print(
            "\nFinal training  {}/{}".format(epoch, args.max_epochs - 1),
            "avg loss: {:.4f}".format(train_metrics["avg_loss"]),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        
        if writer is not None:
            writer.add_scalar("avg_train_loss", train_metrics["avg_loss"], epoch)
        b_new_best = False

        # Time for a validation check?
        if (epoch + 1) % args.val_every == 0:
            print('\n=== Validation ===')
            epoch_time = time.time()

            avg_loss, val_metrics = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                loss_func=loss_func,
                args=args,
                feature_extractor=feature_extractor
            )

            print(
                "\nFinal validation stats {}/{}, time: {:.2f}s \n".format(epoch, args.max_epochs - 1, time.time() - epoch_time),
                "\tAccuracy (global, unweighted): {:>0.1f}".format(val_metrics["acc"]),
                "\n\tPrecision (by class): {:>0.1f}, {:>0.1f}, {:>0.1f}".format(*val_metrics["prec"].tolist()),
                "\n\tRecall (by class): {:>0.1f}, {:>0.1f}, {:>0.1f}".format(*val_metrics["rec"].tolist()),
                "\n\tAvg val loss: {:>8f}".format(avg_loss)
            )

            if writer is not None:
                # Here we take "avg" to mean across all observations in the epoch,
                # and "mean" to mean across the different diagnoses/groups
                writer.add_scalar("avg_val_loss", avg_loss, epoch)
                writer.add_scalar("acc", val_metrics["acc"], epoch)
                for label in [0,1,2]:
                    writer.add_scalar("prec_class_"+str(label), val_metrics["prec"].tolist()[label], epoch)
                    writer.add_scalar("rec_class_"+str(label), val_metrics["rec"].tolist()[label], epoch)

            if val_metrics["acc"] > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_metrics["acc"]))
                val_acc_max = val_metrics["acc"]
                b_new_best = True
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_metrics["acc"], optimizer=optimizer
                    )
            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_metrics["acc"], filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

    print("Training Finished !, Best validation accuracy: ", val_acc_max)

    return val_acc_max

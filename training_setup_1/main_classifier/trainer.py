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
from utils.utils import AverageMeter

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, epoch, loss_func, args, feature_extractor=None):
    model.train()

    start_time = time.time()
    run_loss = AverageMeter()
    for batch_id, batch_data in enumerate(loader):
        data, target = batch_data["images"], batch_data["label"]

        # Extract features
        data = feature_extractor(data)
        
        pred = model(data.to(args.cl_device))
        loss = loss_func(pred, target.to(args.cl_device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Save loss
        run_loss.update(loss.item(), n=args.n_batches)

        print(
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, batch_id, len(loader)),
            "loss: {:.4f}".format(loss.item()),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()

    return loss.item()


def val_epoch(model, loader, epoch, loss_func, args, feature_extractor):
    model.eval()
    size = len(loader.dataset)
    num_batches = len(loader)
    val_loss, correct = 0, 0
    start_time = time.time()
    run_loss = AverageMeter()

    with torch.no_grad():
        for batch_id, batch_data in enumerate(loader):
            data, target = batch_data["images"], batch_data["label"]

            # Extract features
            data = feature_extractor(data)

            # 
            pred = model(data.to(args.cl_device))
            target = target.to(args.cl_device) # send target to gpu
            val_loss = loss_func(pred, target).item() # Calculate loss

            # Calculate loss and number of correct classifications
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
            run_loss.update(val_loss, n=args.n_batches)

            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, batch_id, len(loader)),
                "loss: {:.4f}".format(val_loss),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    
    print(f"Validation Error: \n Accuracy: {(100*correct/size):>0.1f}%, Avg loss: {run_loss.avg:>8f} \n")

    return run_loss.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


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

    val_acc_max = 0.0

    # The training loop!
    for epoch in range(start_epoch, args.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, args=args, feature_extractor=feature_extractor
        )
        
        print(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False

        # Time for a validation check?
        if (epoch + 1) % args.val_every == 0:
            print('Validation')
            epoch_time = time.time()

            val_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                loss_func=loss_func,
                args=args,
                feature_extractor=feature_extractor
            )

            print(
                "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                ", Accuracy:",
                val_loss,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )

            if writer is not None:
                writer.add_scalar("Mean_Val_Acc", val_loss, epoch)
            if val_loss > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_loss))
                val_acc_max = val_loss
                b_new_best = True
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_acc_max, optimizer=optimizer
                    )
            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max

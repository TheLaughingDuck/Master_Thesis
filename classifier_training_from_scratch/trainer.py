
import torch

from utils import AverageMeter, get_metrics, get_conf_matrix, create_conf_matrix_fig
from tracking import TrainingTracker

import os
import time

def run_training(
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        scheduler,
        args):
    
    # Setup TrainingTracker object
    tracker = TrainingTracker(args)

    # Setup logdir
    if args.logdir is not None:
        print("\n\N{Writing Hand}    Writing outputs to:", args.logdir, "\n")

        # Save arguments to logdir

        os.mkdir(os.path.join(os.getcwd(), args.logdir))
        with open(args.logdir+"/args.txt", "w") as f:
            # Save the arguments
            f.write("=== Arguments ===\n")
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
            f.write("=================\n")

            # f.write("=== Model ===\n")
            # for arg in vars(args):
            #     f.write("{}: {}\n".format(arg, getattr(args, arg)))
            # f.write("=================\n")
    

    # The best validation accuracy so far
    val_acc_max = 0

    # The training loop!
    for epoch in range(0, args.max_epochs):
        print("\n===", time.ctime(), "Epoch:", epoch, "===")
        epoch_time = time.time()


        #### RUN ONE TRAINING EPOCH
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_fn, args=args
        )

        tracker.update_epoch({"avg_train_loss":{"step":[epoch],"value":[float(train_metrics["avg_loss"])]}})

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

        # Possibly run a validation epoch
        if (epoch + 1) % args.val_every == 0:
            print('\n=== Validation ===')
            epoch_time = time.time()

            avg_loss, val_metrics = val_epoch(
                model,
                valid_loader,
                epoch=epoch,
                loss_func=loss_fn,
                args=args
            )

            print(
                "\nFinal validation stats {}/{}, time: {:.2f}s \N{Dragon Face}\n".format(epoch, args.max_epochs - 1, time.time() - epoch_time),
                "\tAccuracy (global, unweighted): {:>0.1f}".format(val_metrics["acc"]),
                "\n\tPrecision (by class): {:>0.1f}, {:>0.1f}, {:>0.1f}".format(*val_metrics["prec"].tolist()),
                "\n\tRecall (by class): {:>0.1f}, {:>0.1f}, {:>0.1f}".format(*val_metrics["rec"].tolist()),
                "\n\tAvg val loss: {:>8f} \N{Whale}".format(avg_loss)
            )
    
            tracker.update_epoch({"avg_valid_loss":{"step":[epoch],"value":[float(avg_loss)]}})
            tracker.update_epoch({"acc_glob_unweighted":{"step":[epoch],"value":[float(val_metrics["acc"])]}})

            for label in [0,1,2]:
                tracker.update_epoch({"prec_class_"+str(label):{"step":[epoch],"value":[val_metrics["prec"].tolist()[label]]}})
                tracker.update_epoch({"rec_class_"+str(label):{"step":[epoch],"value":[val_metrics["rec"].tolist()[label]]}})
    
        # At the end of an epoch, save the metrics
        tracker.update_epoch({"learning_rate":{"step":[epoch],"value":[scheduler.get_last_lr()[0]]}})
        tracker.make_key_fig(["avg_train_loss", "avg_valid_loss"], kwargs={"avg_train_loss": {"color": "blue", "label": "Training"}, "avg_valid_loss": {"color": "orange", "label": "Validation"}}, title="CrossEntropy loss")
        tracker.make_key_fig(["acc_glob_unweighted"], title="Acc. (glob. unweighted)")
        tracker.make_key_fig(["learning_rate"], title="Learning rate")
        tracker.to_json()

        # Change to next learning rate value
        if scheduler is not None:
            scheduler.step()
        
        print(f"CURRENT LR: {optimizer.param_groups[0]["lr"]}")



#############################
###### TRAIN ONE EPOCH ######
#############################
def train_epoch(model, loader, optimizer, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    all_preds = []
    all_targets = []
    
    for batch_id, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(args.device), batch_data["label"].to(args.device)

        #print(f"DATA SHAPE: {data.shape}")
        
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
            data, target = batch_data["image"].to(args.device), batch_data["label"].to(args.device)

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
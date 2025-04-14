import datetime
import os
import time

import numpy as np
import torch
import wandb
from torch.nn.functional import cross_entropy

from src.arguments import get_args
from src.dataloaders import get_dataloaders, get_aux_dataloader
from src.measures import AverageMeter, Accuracy
from src.models import get_models
from src.neve.utils import activate_hooks, evaluate_velocity, update_schedulers
from src.optimizers import get_optimizers
from src.schedulers import get_schedulers
from src.utils import set_seeds


def run(args, model, dataloader, optimizers, scaler, epoch, run_type):
    acc = Accuracy((1, 5))

    accuracy_meter_1 = AverageMeter()
    accuracy_meter_5 = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()

    train = optimizers is not None
    model.train(train)

    if train:
        for optimizer in optimizers:
            optimizer.zero_grad()

    t1 = time.time()

    # Define the loss function
    loss_fn = cross_entropy

    for batch, (images, target) in enumerate(dataloader):
        images, target = images.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(args.device == "cuda" and args.amp)):
                output = model(images)
                loss = loss_fn(output, target)

        if train:
            scaler.scale(loss).backward()

            for optimizer in optimizers:
                scaler.step(optimizer)

            scaler.update()

            for optimizer in optimizers:
                optimizer.zero_grad()

        accuracy = acc(output, target)
        accuracy_meter_1.update(accuracy[0].item(), target.shape[0])
        accuracy_meter_5.update(accuracy[1].item(), target.shape[0])
        loss_meter.update(loss.item(), target.shape[0])

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)

        print(f"{run_type}: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
              f"BT {batch_time.avg:.3f}\t"
              f"ETA {datetime.timedelta(seconds=eta)}\t"
              f"loss {loss_meter.avg:.3f}\t")
    return {
        'loss': loss_meter.avg,
        'accuracy': {
            "top1": accuracy_meter_1.avg / 100,
            "top5": accuracy_meter_5.avg / 100
        }
    }


def train(args, model, train_loader, valid_loader, test_loader, aux_loader,
          optimizers, hooks, scaler, epoch):
    neve_new_metrics = []
    logs = {"velocity": {}, "rho": {}, "neve": {}}
    metrics_log_data = {"velocity": {}}

    activate_hooks(hooks, False)

    # Training phase
    train_perf = run(args, model, train_loader, optimizers, scaler, epoch, "Train")

    # Validation phase
    valid_perf = run(args, model, valid_loader, None, scaler, epoch, "Validation")

    # Test phase
    test_perf = run(args, model, test_loader, None, scaler, epoch, "Test")

    # Validation phase on random validation
    if aux_loader:
        activate_hooks(hooks, True)
        # Update hooks
        for h in hooks:
            hooks[h].reset()

        _ = run(args, model, aux_loader, None, scaler, epoch, "Auxiliary")

        metrics_log_data, neve_new_metrics = evaluate_velocity(hooks, logs)

        # Update logs for wandb logging
        for key in logs["velocity"].keys():
            logs["velocity"][key] = wandb.Histogram(np_histogram=np.histogram(logs["velocity"][key], bins=32))
        for key in logs["rho"].keys():
            logs["rho"][key] = wandb.Histogram(np_histogram=np.histogram(logs["rho"][key], bins=32))

    logs["train"] = train_perf
    logs["validation"] = valid_perf
    logs["test"] = test_perf

    # Train results
    print("Train:", train_perf)
    print("Validation:", valid_perf)
    print("Test:", test_perf)

    return logs, metrics_log_data, neve_new_metrics


def train_cycle(epochs, args, model, optimizers, schedulers, scaler, train_loader, valid_loader, test_loader,
                aux_loader,
                hooks, consider_baseline_scheduler=True):
    if aux_loader:
        # First run on validation to get the PSP for epoch -1
        activate_hooks(hooks, True)
        _ = run(args, model, aux_loader, None, scaler, -1, "Auxiliary")

    early_stop_reached = False
    # Epochs
    for e in range(epochs):
        print(f"Epoch {e}")

        # Perform train, validation and test for the current epoch
        logs, metrics_log_data, neve_metrics = train(args, model, train_loader, valid_loader, test_loader, aux_loader,
                                                     optimizers, hooks, scaler, e)

        # Log optimizer(s) learning rate(s)
        logs["lr"] = {i: {} for i, _ in enumerate(optimizers)}
        for i, optimizer in enumerate(optimizers):
            for j, group in enumerate(optimizer.param_groups):
                logs["lr"][i][j] = group["lr"]

        # Update the scheduler(s)
        if args.use_scheduler:
            update_schedulers(schedulers, args.scheduler_metric, logs["validation"]["loss"],
                              metrics_log_data["velocity"],
                              consider_baseline_scheduler=consider_baseline_scheduler)

        # TODO: ADD HERE EARLY-STOP
        if aux_loader and not early_stop_reached:
            if metrics_log_data.get("model_avg_value", np.inf) < 1e-3:
                early_stop_reached = True
                wandb.log({"neve_stop": e}, commit=False)
        # Log on wandb project
        wandb.log(logs)


def main(args):
    print(args)

    # Set reproducibility seeds
    set_seeds(args.seed)

    # Get model and attach hooks
    model, hooks = get_models(args)
    # Get optimizer(s)
    optimizers = get_optimizers(args, model)
    # Get scheduler(s)
    schedulers = get_schedulers(args, optimizers)
    # Get Dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(args)
    # Get Auxiliary Dataloader
    aux_loader = None
    if args.scheduler_metric == "vloss":
        aux_loader = valid_loader
    elif args.scheduler_metric == "neve":
        aux_loader = get_aux_dataloader(args)

    if not args.generate_neve_metrics:
        aux_loader = None
    # Create amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda" and args.amp))

    # Init wandb project
    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # Perform the training
    train_cycle(args.epochs, args, model, optimizers, schedulers, scaler,
                train_loader, valid_loader, test_loader, aux_loader,
                hooks)

    # Save model
    if not os.path.exists(os.path.join(args.checkpoint, wandb.run.id)):
        os.makedirs(os.path.join(args.checkpoint, wandb.run.id))
    torch.save(model.state_dict(), os.path.join(args.checkpoint, wandb.run.id, "last.pt"))

    # End wandb run
    wandb.run.finish()


if __name__ == '__main__':
    main(get_args())

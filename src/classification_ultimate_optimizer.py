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
from src.neve.utils import activate_hooks, evaluate_velocity
from src.ultimate_optimizer import gdtuo
from src.utils import set_seeds


def evaluate(args, model, dataloader, epoch, run_type):
    acc = Accuracy((1, 5))

    accuracy_meter_1 = AverageMeter()
    accuracy_meter_5 = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    t1 = time.time()

    for batch, (images, target) in enumerate(dataloader):
        images, target = images.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)

        with torch.set_grad_enabled(False):
            with torch.cuda.amp.autocast(enabled=(args.device == "cuda" and args.amp)):
                output = model.forward(images)
                loss = cross_entropy(output, target)

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


def train_epoch(args, model_wrapper, dataloader, epoch):
    acc = Accuracy((1, 5))

    accuracy_meter_1 = AverageMeter()
    accuracy_meter_5 = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()

    t1 = time.time()

    for batch, (images, target) in enumerate(dataloader):
        model_wrapper.begin()  # call this before each step, enables gradient tracking on desired params
        images, target = images.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(args.device == "cuda" and args.amp)):
            output = model_wrapper.forward(images)
            loss = cross_entropy(output, target)

            model_wrapper.zero_grad()
            loss.backward(create_graph=True)  # important! use create_graph=True
            model_wrapper.step()

        accuracy = acc(output, target)
        accuracy_meter_1.update(accuracy[0].item(), target.shape[0])
        accuracy_meter_5.update(accuracy[1].item(), target.shape[0])
        loss_meter.update(loss.item(), target.shape[0])

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)

        print(f"Train: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
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


def train(args, model, model_wrapper, train_loader, valid_loader, test_loader, aux_loader,
          hooks, old_metrics, epoch):
    neve_new_metrics = []
    logs = {"velocity": {}, "neve": {}}
    metrics_log_data = {"velocity": {}}

    activate_hooks(hooks, False)

    # Training phase
    train_perf = train_epoch(args, model_wrapper, train_loader, epoch)

    # Validation phase
    valid_perf = evaluate(args, model, valid_loader, epoch, "Validation")

    # Test phase
    test_perf = evaluate(args, model, test_loader, epoch, "Test")

    # Validation phase on random validation
    if aux_loader:
        activate_hooks(hooks, True)
        # Save the activations into the dict
        for k in hooks:
            hooks[k].reset()
        _ = evaluate(args, model, aux_loader, epoch, "Auxiliary")

        metrics_log_data, neve_new_metrics = evaluate_velocity(hooks, logs, old_metrics)

        # Update logs for wandb logging
        for key in logs["velocity"].keys():
            logs["velocity"][key] = wandb.Histogram(np_histogram=np.histogram(logs["velocity"][key], bins=32))

    logs["train"] = train_perf
    logs["validation"] = valid_perf
    logs["test"] = test_perf

    # Train results
    print("Train:", train_perf)
    print("Validation:", valid_perf)
    print("Test:", test_perf)

    return logs, metrics_log_data, neve_new_metrics


def train_cycle(epochs, args, model, train_loader, valid_loader, test_loader, aux_loader, hooks):
    neve_metrics = []

    optim = gdtuo.SGD(alpha=args.lr, mu=args.momentum,
                      optimizer=gdtuo.SGD(alpha=(args.lr ** 2) * 1e-3, mu=(1 / (1 - args.momentum)) * 1e-6))
    model_wrapper = gdtuo.ModuleWrapper(model, optimizer=optim)
    model_wrapper.initialize()

    if aux_loader:
        # First run on validation to get the PSP for epoch -1
        activate_hooks(hooks, True)
        _ = evaluate(args, model, aux_loader, -1, "Auxiliary")

    previous_lr = model_wrapper.get_lr()
    print(f"LR: {previous_lr}")
    # Epochs
    for e in range(epochs):
        print(f"Epoch {e}")

        # Perform train, validation and test for the current epoch
        logs, metrics_log_data, neve_metrics = train(args, model, model_wrapper, train_loader, valid_loader,
                                                     test_loader,
                                                     aux_loader, hooks, neve_metrics, e)

        # Log optimizer(s) learning rate(s)
        logs["lr"] = previous_lr
        previous_lr = model_wrapper.get_lr()

        # TODO: ADD HERE EARLY-STOP
        if aux_loader and not early_stop_reached:
            if metrics_log_data.get("model_avg_value", np.inf) < 1e-3:
                early_stop_reached = True
                wandb.log({"neve_stop": e}, commit=False)
        # Log on wandb project
        wandb.log(logs)


def main(args):
    args.project_name = "NeVe-UOptimizer"
    print(args)

    # Set reproducibility seeds
    set_seeds(args.seed)

    # Get model and attach hooks
    model, hooks = get_models(args)
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

    # Init wandb project
    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # Perform the training
    train_cycle(args.epochs, args, model,
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

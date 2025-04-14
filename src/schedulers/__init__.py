from torch.optim.lr_scheduler import MultiStepLR

from src.schedulers.scheduler import ReduceLROnLocalPlateau


def get_scheduler(args, optimizer):
    if not args.use_scheduler:
        return None
    if args.scheduler_metric == "baseline":
        if args.dataset in ["cifar10", "cifar100", "imagenet100"]:
            milestones = [100, 150]
            if args.dataset in ["cifar10", "cifar100"]:
                milestones = [100, 150]
            elif args.dataset in ["imagenet100"]:
                milestones = [30, 60]
            return MultiStepLR(optimizer, milestones=milestones)
        else:
            return None
    else:
        return ReduceLROnLocalPlateau(optimizer, mode="min", threshold_mode="rel", verbose=True,
                                      patience=args.scheduler_patience, factor=args.scheduler_factor)


def get_schedulers(args, optimizers):
    print("Initialize scheduler")

    schedulers = []

    for optimizer in optimizers:
        schedulers.append(get_scheduler(args, optimizer))

    return schedulers

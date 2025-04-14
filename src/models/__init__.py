from src.models.cifar_resnet import resnet32, resnet20
from src.models.imagenet_resnet import resnet50
from src.neve.utils import attach_hooks


def get_models(args):
    print(f"Initialize model {args.arch}")

    if args.arch == "resnet32-cifar":
        assert args.dataset in ["cifar10", "cifar100"]
        # 10 classes is default for cifar10
        n_classes = 10
        if args.dataset == "cifar100":
            n_classes = 100
        model = resnet32(num_classes=n_classes)
    elif args.arch == "resnet20-cifar":
        assert args.dataset in ["cifar10", "cifar100"]
        # 10 classes is default for cifar10
        n_classes = 10
        if args.dataset == "cifar100":
            n_classes = 100
        model = resnet20(num_classes=n_classes)
    elif args.arch == "resnet50-imagenet":
        assert args.dataset in ["imagenet100"]
        # 1000 classes is default for imagenet
        n_classes = 1000
        if args.dataset == "imagenet100":
            n_classes = 100
        model = resnet50(num_classes=n_classes, weights=None)
    else:
        raise ValueError(f"No such model {args.arch}")

    hooks = attach_hooks(args, model)
    model.to(args.device)

    return model, hooks

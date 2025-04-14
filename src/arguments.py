import argparse


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def _add_common_arguments(parser):
    # General
    parser.add_argument("--seed", type=int, default=1,
                        help="Reproducibility seed.")
    parser.add_argument("--root", type=str, default="/scratch/",
                        help="Dataset root folder.")
    parser.add_argument("--dataset-path", type=str, default="datasets/",
                        help="Dataset from root main folder.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/",
                        help="Folder where we save the last model of the training.")
    parser.add_argument("--amp", type=int2bool, choices=[0, 1], default=True,
                        help="If True use torch.cuda.amp.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                        help="Device type.")

    # Scheduler
    parser.add_argument("--use-scheduler", type=int2bool, choices=[0, 1], default=True,
                        help="If not selected (0), use a constant learning rate.")
    parser.add_argument("--scheduler-metric", type=str, choices=["baseline", "vloss", "neve"], default="neve",
                        help="What metric to use to update scheduler: baseline, validation loss or neve (velocity).")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                        help="Factor by which the learning rate will be reduced. new_lr = lr * factor.")
    parser.add_argument("--scheduler-patience", type=int, default=10,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")

    # Optimizer
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Optimizer momentum.")

    # Dataset
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Number of workers (threads) per process.")
    parser.add_argument("--aux-dataset", type=str, default="",
                        help="Auxiliary Dataset folder name.")
    parser.add_argument("--aux-samples", type=int, default=100,
                        help="Maximum number of samples in the auxiliary set")

    # NeVe
    parser.add_argument("--generate-neve-metrics", type=int2bool, choices=[0, 1], default=True,
                        help="True if we want to generate NEq data, False if we don't.")
    parser.add_argument("--neve-use-gpu", type=int2bool, choices=[0, 1], default=False,
                        help="1 if we store neve data in gpu, 0 if we use cpu (big models might use a lot of vram)")
    parser.add_argument("--velocity-mu", type=float, default=0.5,
                        help="Velocity momentum.")

    # Logging
    parser.add_argument("--run-name", type=str, default=None)


def _get_classification_arguments(parser):
    # Default
    _add_common_arguments(parser)

    # Model
    parser.add_argument("--arch", type=str, choices=["resnet20-cifar", "resnet32-cifar", "resnet50-imagenet"],
                        default="resnet32-cifar",
                        help="Architecture name.")

    # Train
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of benchmark epochs.")

    parser.add_argument("--scheduler", type=str,
                        choices=[""],
                        default="",
                        help="LR Scheduler.")

    # Optimizer
    parser.add_argument("--optim", type=str, choices=["sgd", "adam"], default="sgd",
                        help="Optimizer.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Optimizer weight decay.")

    # Dataset
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "imagenet100"],
                        default="cifar10",
                        help="Source dataset.")
    parser.add_argument("--valid-size", type=float, default=0.1,
                        help="Validation size as portion of the whole train set.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size per process.")

    # Logging
    parser.add_argument("--project-name", type=str, default="NeVe-Classification")

    args = parser.parse_args()
    return args


def get_args(script="classification"):
    assert script in ["classification"]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if script == "classification":
        return _get_classification_arguments(parser)

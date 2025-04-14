import os
import sys

import torch

from src.dataloaders.cifar import get_cifar10, get_cifar100
from src.dataloaders.imagenet import get_imagenet100
from src.dataloaders.random import RandomDataset


def get_dataloaders(args):
    print(f"Initialize dataset {args.dataset}")
    assert args.dataset in ["cifar10", "cifar100", "imagenet100"]

    if args.dataset == "cifar10":
        train_dataloader, valid_dataloader, test_dataloader = get_cifar10(args)
    elif args.dataset == "cifar100":
        train_dataloader, valid_dataloader, test_dataloader = get_cifar100(args)
    elif args.dataset == "imagenet100":
        train_dataloader, valid_dataloader, test_dataloader = get_imagenet100(args)
    else:
        print("Dataset not managed yet!")
        sys.exit(-1)

    return train_dataloader, valid_dataloader, test_dataloader


def get_aux_dataloader(args):
    print(f"Initialize random dataset {args.dataset}")
    assert args.dataset in ["cifar10", "cifar100", "imagenet100"]

    if args.dataset in ["cifar10", "cifar100"]:
        aux_dataloader = get_random_dataset(args, (3, 32, 32), args.aux_samples)
    elif args.dataset in ["imagenet100"]:
        aux_dataloader = get_random_dataset(args, (3, 224, 224), args.aux_samples)
        print("Aux Dataset not managed yet!")
        sys.exit(-1)

    return aux_dataloader


def get_random_dataset(args, shape, number_samples, labels_as_segmap=False, is_image=True, output_as_float=False,
                       save_to_disk=False):
    aux_dataset = RandomDataset(path=str(os.path.join(args.root, args.aux_dataset)) if args.aux_dataset else None,
                                img_shape=shape, dataset_size=number_samples, label_as_segmap=labels_as_segmap,
                                is_image=is_image, output_as_float=output_as_float, save_to_disk=save_to_disk)

    num_workers = args.num_workers if hasattr(args, "num_workers") else args.workers
    aux_dataloader = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True,
                                                 persistent_workers=num_workers > 0)
    return aux_dataloader

import os.path

import torch
import torchvision
from torchvision import transforms

from src.dataloaders.utils import split_dataset, MapDataset

CIFAR_CROP = 32
CIFAR_PAD = 4

CIFAR10_mean = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_std = (0.2023, 0.1994, 0.2010)
CIFAR10 = [
    transforms.Compose([
        transforms.RandomCrop(CIFAR_CROP, padding=CIFAR_PAD),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_mean, CIFAR10_std)
    ]),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_mean, CIFAR10_std)
    ])
]

CIFAR100_mean = (0.5071, 0.4867, 0.4408)
CIFAR100_std = (0.2675, 0.2565, 0.2761)
CIFAR100 = [
    transforms.Compose([
        transforms.RandomCrop(CIFAR_CROP, padding=CIFAR_PAD),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_mean, CIFAR100_std)
    ]),
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_mean, CIFAR100_std)
    ])
]


def get_cifar10(args):
    # Train and Validation
    base_ds_path = os.path.join(args.root, args.dataset_path)
    train_dataset = torchvision.datasets.CIFAR10(base_ds_path, train=True, transform=None, download=True)
    train, validation = split_dataset(train_dataset, args.valid_size, args.seed)

    train, validation = MapDataset(train, CIFAR10[0]), MapDataset(validation, CIFAR10[1])

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   persistent_workers=args.num_workers > 0)

    valid_dataloader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   persistent_workers=args.num_workers > 0)

    test_dataset = torchvision.datasets.CIFAR10(base_ds_path, train=False, transform=CIFAR10[1], download=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True,
                                                  persistent_workers=args.num_workers > 0)

    return train_dataloader, valid_dataloader, test_dataloader


def get_cifar100(args):
    # Train and Validation
    base_ds_path = os.path.join(args.root, args.dataset_path)
    train_dataset = torchvision.datasets.CIFAR100(base_ds_path, train=True, transform=None, download=True)
    train, validation = split_dataset(train_dataset, args.valid_size, args.seed)

    train, validation = MapDataset(train, CIFAR100[0]), MapDataset(validation, CIFAR100[1])

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   persistent_workers=args.num_workers > 0)

    valid_dataloader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=True,
                                                   persistent_workers=args.num_workers > 0)

    test_dataset = torchvision.datasets.CIFAR100(base_ds_path, train=False, transform=CIFAR100[1], download=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True,
                                                  persistent_workers=args.num_workers > 0)

    return train_dataloader, valid_dataloader, test_dataloader

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os

import numpy as np
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Generator
from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from src.swin_transformer.data.cached_image_folder import CachedImageFolder
from src.swin_transformer.data.imagenet22k_dataset import IN22KDATASET
from src.swin_transformer.data.samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def split_dataset(dataset, percentage, random_seed):
    # If percentage is between [0, 1] we treat it as a percentage
    if 0 <= percentage <= 1:
        dataset_length = len(dataset)
        valid_length = int(np.floor(percentage * dataset_length))
        train_length = dataset_length - valid_length
        train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                    generator=Generator().manual_seed(random_seed))

        return train_dataset, valid_dataset
    # if percentage value is greater than 1 we use it as integer,
    # and it's the number of validation samples we want
    else:
        valid_length = int(percentage)
        train_length = len(dataset) - valid_length
        train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                    generator=Generator().manual_seed(random_seed))

        return train_dataset, valid_dataset


IMAGENET_mean = (0.485, 0.456, 0.406)
IMAGENET_std = (0.229, 0.224, 0.225)
# IMAGENET_RESIZE_SIZE = 320
IMAGENET_CROP_SIZE = 256

IMAGENET = [
    transforms.Compose([
        # transforms.Resize(IMAGENET_RESIZE_SIZE),
        transforms.RandomResizedCrop(IMAGENET_CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_mean, IMAGENET_std)
    ]),
    transforms.Compose([
        # transforms.Resize(IMAGENET_RESIZE_SIZE),
        transforms.CenterCrop(IMAGENET_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_mean, IMAGENET_std)
    ])
]


class MapDataset(Dataset):
    def __init__(self, dataset, map_fn, with_target=False):
        self.dataset = dataset
        self.map = map_fn
        self.with_target = with_target

    def __getitem__(self, index):
        if self.with_target:
            return self.map(self.dataset[index][0], self.dataset[index][1])
        else:
            return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


imagenet100_classes = ['n02869837',
                       'n01749939',
                       'n02488291',
                       'n02107142',
                       'n13037406',
                       'n02091831',
                       'n04517823',
                       'n04589890',
                       'n03062245',
                       'n01773797',
                       'n01735189',
                       'n07831146',
                       'n07753275',
                       'n03085013',
                       'n04485082',
                       'n02105505',
                       'n01983481',
                       'n02788148',
                       'n03530642',
                       'n04435653',
                       'n02086910',
                       'n02859443',
                       'n13040303',
                       'n03594734',
                       'n02085620',
                       'n02099849',
                       'n01558993',
                       'n04493381',
                       'n02109047',
                       'n04111531',
                       'n02877765',
                       'n04429376',
                       'n02009229',
                       'n01978455',
                       'n02106550',
                       'n01820546',
                       'n01692333',
                       'n07714571',
                       'n02974003',
                       'n02114855',
                       'n03785016',
                       'n03764736',
                       'n03775546',
                       'n02087046',
                       'n07836838',
                       'n04099969',
                       'n04592741',
                       'n03891251',
                       'n02701002',
                       'n03379051',
                       'n02259212',
                       'n07715103',
                       'n03947888',
                       'n04026417',
                       'n02326432',
                       'n03637318',
                       'n01980166',
                       'n02113799',
                       'n02086240',
                       'n03903868',
                       'n02483362',
                       'n04127249',
                       'n02089973',
                       'n03017168',
                       'n02093428',
                       'n02804414',
                       'n02396427',
                       'n04418357',
                       'n02172182',
                       'n01729322',
                       'n02113978',
                       'n03787032',
                       'n02089867',
                       'n02119022',
                       'n03777754',
                       'n04238763',
                       'n02231487',
                       'n03032252',
                       'n02138441',
                       'n02104029',
                       'n03837869',
                       'n03494278',
                       'n04136333',
                       'n03794056',
                       'n03492542',
                       'n02018207',
                       'n04067472',
                       'n03930630',
                       'n03584829',
                       'n02123045',
                       'n04229816',
                       'n02100583',
                       'n03642806',
                       'n04336792',
                       'n03259280',
                       'n02116738',
                       'n02108089',
                       'n03424325',
                       'n01855672',
                       'n02090622']


class ImageNet100(ImageFolder):
    def find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = list(filter(lambda name: name in imagenet100_classes, classes))
        print("Loaded", len(classes), "classes")

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_loader(config, args):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    dataset_train, dataset_valid = split_dataset(dataset_train, args.valid_size, args.seed)
    if config.MODEL.NUM_CLASSES == 100:
        dataset_train, dataset_valid = MapDataset(dataset_train, IMAGENET[0]), MapDataset(dataset_valid, IMAGENET[1])
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank  successfully build train dataset")
    dataset_test, _ = build_dataset(is_train=False, config=config)
    dataset_test = MapDataset(dataset_test, IMAGENET[1])
    print(f"local rank {config.LOCAL_RANK} / global rank  successfully build val dataset")

    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_train = torch.utils.data.DistributedSampler(
        #    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        # )

    if config.TEST.SEQUENTIAL:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_valid = torch.utils.data.SequentialSampler(dataset_valid)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        # sampler_test = torch.utils.data.distributed.DistributedSampler(
        #    dataset_test, shuffle=config.TEST.SHUFFLE
        # )
        sampler_valid = torch.utils.data.SequentialSampler(dataset_valid)
        # sampler_valid = torch.utils.data.distributed.DistributedSampler(
        #    dataset_valid, shuffle=config.TEST.SHUFFLE
        # )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, sampler=sampler_valid,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #    mixup_fn = Mixup(
    #        mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #        prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #        label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_valid, dataset_test, data_loader_train, data_loader_valid, data_loader_test, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet100':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = ImageNet100(root=root)
        nb_classes = 100
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

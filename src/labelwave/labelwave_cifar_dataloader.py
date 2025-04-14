import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.labelwave.noise_build import dataset_split


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')
    return dict


def initial_data(dataset, r, noise_mode, file_name, root_dir, random_seed):
    print('============ Initialize data')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    noise_label = []

    dataset = dataset.lower()
    root_dir = os.path.join(root_dir, f"{dataset}-python")

    if dataset == "cifar-10":
        num_classes = 10
    else:
        num_classes = 100
    test_dic = unpickle('%s/test' % root_dir)
    test_data = test_dic[b'data']
    test_data = test_data.reshape((10000, 3, 32, 32))
    test_data = test_data.transpose((0, 2, 3, 1))
    if "labels" in test_dic.keys():
        test_label = test_dic[b'labels']
    else:
        test_label = test_dic[b'fine_labels']

    train_dic = unpickle('%s/train' % root_dir)
    train_data = train_dic[b'data']
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))
    if "labels" in train_dic.keys():
        train_label = train_dic[b'labels']
    else:
        train_label = train_dic[b'fine_labels']

    noise_label = dataset_split(train_images=train_data,
                                train_labels=train_label,
                                noise_rate=r,
                                noise_type=noise_mode,
                                random_seed=random_seed,
                                num_classes=num_classes)
    print('============ Actual clean samples number: ', sum(np.array(noise_label) == np.array(train_label)))

    num_samples = int(noise_label.shape[0])

    train_set_index = np.random.choice(num_samples, int(num_samples * 0.8), replace=False)
    index = np.arange(train_data.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = train_data[train_set_index, :], train_data[val_set_index, :]
    train_labels, val_labels = noise_label[train_set_index], noise_label[val_set_index]
    train_clean_labels, val_clean_labels = np.array(train_label)[train_set_index], np.array(train_label)[val_set_index]

    orignal_train_data = train_set
    orignal_train_label = train_clean_labels
    orignal_noise_label = train_labels
    orignal_test_data = test_data
    orignal_test_label = test_label
    orignal_val_label = val_labels
    orignal_val_data = val_set
    np.savetxt(file_name + '/' + file_name + '_noise_label.csv', orignal_noise_label, delimiter=',')
    np.savetxt(file_name + '/' + file_name + '_train_label.csv', orignal_train_label, delimiter=',')
    return train_set, train_clean_labels, train_labels, test_data, test_label, val_set, val_labels


class cifar_dataset(Dataset):
    def __init__(self, data, real_label, label, roundindex, transform, mode, strong_transform=None, pred=[],
                 probability=[], test_log=None, id_list=None):
        self.data = None
        self.label = None
        self.transform = transform
        self.strong_aug = transform
        self.mode = mode
        self.pred = pred
        self.probability = None
        self.real_label = real_label
        self.id_list = id_list
        self.data = data
        self.label = label
        self.roundindex = roundindex

    def __getitem__(self, index):

        if self.mode == 'all':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'roundtrain':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            roundindex1 = self.roundindex[index]
            return img, target, roundindex1
        elif self.mode == 'test':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode == 'val':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, file_name, root_dir, random_seed,
                 noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.random_seed = random_seed
        self.file_name = file_name
        self.noise_file = noise_file
        self.train_data, self.train_label, self.noise_label, self.test_data, self.test_label, self.val_set, self.val_labels = initial_data(
            self.dataset, self.r, self.noise_mode, self.file_name, self.root_dir, self.random_seed)

        roundindex = []
        self.roundindex = roundindex
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def run(self, mode, pred=[], prob=[], test_log=None):
        if mode == 'train':
            labeled_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.roundindex,
                                            self.transform_train, mode='all',
                                            strong_transform=None, pred=pred, probability=prob, test_log=test_log)
            train_loader = DataLoader(dataset=labeled_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)
            return train_loader


        elif mode == 'etrain':
            labeled_dataset = cifar_dataset(self.train_data, self.train_label, self.noise_label, self.roundindex,
                                            self.transform_train,
                                            mode='all',
                                            strong_transform=None, pred=pred, probability=prob, test_log=test_log)
            etrain_loader = DataLoader(dataset=labeled_dataset, batch_size=1, shuffle=False,
                                       num_workers=self.num_workers)
            return etrain_loader


        elif mode == 'test':
            test_dataset = cifar_dataset(self.test_data, self.train_label, self.test_label, self.roundindex,
                                         self.transform_train, mode='test',
                                         strong_transform=None, pred=pred, probability=prob)
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
            return test_loader

        elif mode == 'val':
            val_dataset = cifar_dataset(self.val_set, self.train_label, self.val_labels, self.roundindex,
                                        self.transform_train, mode='val',
                                        strong_transform=None, pred=pred, probability=prob)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
            return val_loader

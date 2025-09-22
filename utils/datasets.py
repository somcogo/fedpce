import os
import bisect

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, USPS, SVHN, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
import h5py

class ImageNetDataSet(Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        h5_file = h5py.File(os.path.join(data_dir, 'tiny_imagenet_{}.hdf5'.format(mode)), 'r')
        self.data = h5_file['data']
        self.targets = np.array(h5_file['labels'])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index]).permute(2, 0, 1)
        target = self.targets[index]
        return img, target

def get_cifar10_datasets(data_dir):
    train_mean = [0.4914, 0.4822, 0.4465]
    train_std = [0.2470, 0.2435, 0.2616]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.4942, 0.4851, 0.4504]
    val_std = [0.2467, 0.2429, 0.2616]
    val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

    dataset = CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    dataset.targets = np.array(dataset.targets)
    val_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=val_transform)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_cifar100_datasets(data_dir):
    train_mean = [0.4914, 0.4822, 0.4465]
    train_std = [0.2470, 0.2435, 0.2616]
    train_transform = Compose([ToTensor(), Normalize(train_mean, train_std)])
    val_mean = [0.4914, 0.4822, 0.4465]
    val_std = [0.2470, 0.2435, 0.2616]
    val_transform = Compose([ToTensor(), Normalize(val_mean, val_std)])

    dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    dataset.targets = np.array(dataset.targets)
    val_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_mnist_datasets(data_dir):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transforms = Compose([
        Resize((32, 32)),
        ToTensor(),
        Lambda(lambda x: x.repeat(3, 1, 1)),
        Normalize(mean, std)
    ])

    dataset = MNIST(root=data_dir, train=True, download=True, transform=transforms)
    dataset.targets = np.array(dataset.targets)
    val_dataset = MNIST(root=data_dir, train=False, transform=transforms)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_usps_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([Resize((32, 32)),
                          ToTensor(),
                          Lambda(lambda x: x.repeat(3, 1, 1)),
                          Normalize(mean, std)])

    dataset = USPS(root=data_dir, train=True, download=True, transform=transforms)
    dataset.targets = np.array(dataset.targets)
    val_dataset = USPS(root=data_dir, train=False, download=True, transform=transforms)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_svhn_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([Resize((32, 32)),
                          ToTensor(),
                          Normalize(mean, std)])

    dataset = SVHN(root=data_dir, split='train', download=True, transform=transforms)
    dataset.targets = np.array(dataset.labels)
    val_dataset = SVHN(root=data_dir, split='test', download=True, transform=transforms)
    val_dataset.targets = np.array(val_dataset.labels)

    return dataset, val_dataset

def get_syn_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([Resize((32, 32)),
                          ToTensor(),
                          Normalize(mean, std)])

    dataset = ImageFolder(root=os.path.join(data_dir, 'synthetic_digits', 'imgs_train'), transform=transforms)
    dataset.targets = np.array(dataset.targets)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'synthetic_digits', 'imgs_valid'), transform=transforms)
    val_dataset.targets = np.array(val_dataset.targets)

    return dataset, val_dataset

def get_digits_dataset(data_dir):
    mnist, val_mnist = get_mnist_datasets(data_dir)
    usps, val_usps = get_usps_dataset(data_dir)
    svhn, val_svhn = get_svhn_dataset(data_dir)
    syn, val_syn = get_syn_dataset(data_dir)

    return ConcatWithTargets([mnist, usps, svhn, syn]), ConcatWithTargets([val_mnist, val_usps, val_svhn, val_syn])


class ConcatWithTargets(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.targets = np.concatenate([dset.targets for dset in datasets])
        self.cumulative_sizes = np.cumsum(np.array([len(dset) for dset in self.datasets]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

def get_image_net_dataset(data_dir):
    dataset = ImageNetDataSet(data_dir=data_dir, mode='trn')
    val_dataset = ImageNetDataSet(data_dir=data_dir, mode='val')
    return dataset, val_dataset
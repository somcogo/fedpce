import random
import os

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_image_net_dataset, get_digits_dataset
from .partition import partition_wrap

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_datasets(data_dir, dataset):
    if dataset == 'cifar10':
        trn_dataset, val_dataset = get_cifar10_datasets(data_dir=data_dir)
    elif dataset == 'cifar100':
        trn_dataset, val_dataset = get_cifar100_datasets(data_dir=data_dir)
    elif dataset == 'imagenet':
        trn_dataset, val_dataset = get_image_net_dataset(data_dir=data_dir)
    elif dataset == 'digits':
        trn_dataset, val_dataset = get_digits_dataset(data_dir=data_dir)
    return trn_dataset, val_dataset


def get_dl_lists(dataset, batch_size, degs, site_number, alpha, cl_per_site, main_dir, num_classes, trn_set_size=None, cross_val_id=None, part_seed=None, gl_seed=None, dl_shuffle=True, **kwargs):
    data_path = os.path.join(main_dir, 'data')
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset)

    g = torch.Generator()
    loader_seed = gl_seed if gl_seed is not None else 0
    g.manual_seed(loader_seed)

    (trn_map, val_map) = partition_wrap(data_path, dataset, degs, site_number, trn_dataset, val_dataset, num_classes, alpha, cl_per_site, seed=part_seed)
    trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in trn_map.values()]
    val_ds_list = [Subset(val_dataset, idx_map) for idx_map in val_map.values()]

    if cross_val_id is not None:
        merged_ds_list = [ConcatDataset([trn_ds, val_ds]) for trn_ds, val_ds in zip(trn_ds_list, val_ds_list)]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        splits = [list(kfold.split(range(len(merged_ds)))) for merged_ds in merged_ds_list]
        indices = [split[cross_val_id] for split in splits]
        trn_ds_list = [Subset(ds, idx_map[0]) for ds, idx_map in zip(merged_ds_list, indices)]
        val_ds_list = [Subset(ds, idx_map[1]) for ds, idx_map in zip(merged_ds_list, indices)]

    if trn_set_size is not None:
        trn_ds_list = [Subset(ds, np.arange(trn_set_size)) for ds in trn_ds_list]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=dl_shuffle, pin_memory=True, num_workers=0, worker_init_fn=seed_worker, generator=g) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=1024 if batch_size==64 else batch_size, shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker, generator=g) for val_ds in val_ds_list]
    return trn_dl_list, val_dl_list
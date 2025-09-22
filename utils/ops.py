import numpy as np
import torch
from torchvision.transforms import (
     functional,
     Compose,
     Pad,
     RandomCrop,
     RandomHorizontalFlip,
     RandomVerticalFlip,
     RandomApply,
     RandomRotation,
     RandomErasing)

def get_ft_indices(site_number, ft_site_number, degs, seed=0):
    part_rng = np.random.default_rng(seed)
    if degs == ['digits']:
        deg_nr = 4
        site_per_deg = site_number // 4
        ft_site_per_deg = ft_site_number // 4
    else:
        deg_nr = len(degs)
        site_per_deg = site_number // len(degs)
        ft_site_per_deg = ft_site_number // len(degs)
    indices = []
    for i in range(deg_nr):
        indices.append(part_rng.permutation(np.arange(i * site_per_deg, (i+1) * site_per_deg))[:ft_site_per_deg])
    indices = np.concatenate(indices)

    return indices

def transform_image(batch, labels, mode, img_tr, target_tr, dataset):
    if mode == 'trn':
        batch, labels = aug_image(batch, labels, dataset)
    if img_tr is not None:
        # For some transforms, e.g. colorjitter, the pixelvalues need to be in [0, 1]
        b_max = torch.amax(batch, dim=(1, 2, 3), keepdim=True)
        b_min = torch.amin(batch, dim=(1, 2, 3), keepdim=True)
        batch = (batch - b_min) / (b_max - b_min + 1e-5)
        batch = img_tr(batch)
        batch = (b_max - b_min) * batch + b_min
    if target_tr is not None:
        labels = target_tr(labels)
    return batch, labels

def aug_image(batch, labels, dataset):
    if dataset == 'imagenet':
        img_size = 64
    else:
        img_size = 32
    trans = Compose([
        Pad(4),
        RandomCrop(img_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply(torch.nn.ModuleList([
            RandomRotation(degrees=15)
        ]), p=0.25),
        RandomErasing(p=0.5, scale=(0.015625, 0.25), ratio=(0.25, 4))
    ])
    batch = trans(batch)
        
    return batch, labels

def get_transforms(site_number, tr_gen_seed, degs, device, var_add, **kwargs):
    if type(degs) != list:
        degs = [degs]

    assert site_number % len(degs) == 0
    site_per_deg = site_number // len(degs)
    transforms = []
    for deg in degs:
        deg_transforms = get_deg_transforms(site_per_deg, tr_gen_seed, deg, device, var_add)
        transforms.extend(deg_transforms)
    return transforms

def get_deg_transforms(site_number, seed, degradation, device, var_add):
    rng = np.random.default_rng(seed)
    transforms = []
    if degradation == 'gauss':
        variances = np.linspace(var_add[0], var_add[1], site_number)
        for i in range(site_number):
            transforms.append(DetAddGaussianNoise(rng=rng, device=device, var_add=variances[i]))
    else:
        for i in range(site_number):
            transforms.append(None)

    return transforms

class DetAddGaussianNoise:
    def __init__(self, rng, device, var_add):
        self.rng = rng
        self.var_add = var_add
        self.device = device

    def __call__(self, img:torch.Tensor):
        sigma = self.var_add**0.5
        gauss = self.rng.normal(0, sigma, img.shape)
        gauss = torch.tensor(gauss, device=self.device)
        img = img + gauss
        return img.float()
    
def get_target_transforms(site_number, degs, dataset, **kwargs):
    transforms = []
    for i in range(site_number):
        transforms.append(None)
    return transforms
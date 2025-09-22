import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch

from config import get_config
from main import main

torch.set_num_threads(8)
main_dir = 'path/to/dir'

def run():
    # Script to run FedAS
    dset = 'cifar10' # cifar10, cifar100, imagenet or digits
    degs = ['gauss'] # degredation: ['gauss'], ['dirichlet'] or ['digits']
    site_number = 20 
    xval = 0 # cross validation fold: 0, 1, 2, 3, 4
    comment = f'fedas'
    experiment = 'gcpr2025'

    alpha = 1 if degs == ['dirichlet'] else None
    var = 1 if degs == ['gauss'] else None
    site_str = f'/s{site_number}' if dset == 'cifar10' and site_number != 20 else ''
    logdir = f'{experiment}/{dset}/fedas{site_str}/{degs[0][:4]}'
    args= {
        'dataset':dset,
        'site_number':site_number,
        'deg':degs,
        'alpha':alpha,
        'comment':comment,
        'logdir':logdir,
        'experiment':'fedas',
        'main_dir':main_dir,
        'cross_val_id':xval,
        'var_add':(0.005, var),
    }
    # Training
    config = get_config(**args)
    trn_state_dict, _ = main(**config)
    # Personalization
    args['only_ft'] = True
    for trn_set_size in [2, 5, 10, 25, 50, None]:
        args['trn_set_size'] = trn_set_size
        args['state_dict'] = trn_state_dict
        config = get_config(**args)
        _, _ = main(**config)


if __name__ == '__main__':
    run()
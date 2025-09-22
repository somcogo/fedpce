import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

from config import get_config
from main import main

torch.set_num_threads(8)
main_dir = 'path/to/dir'

def run():
    # script for FedPCE+head
    dset = 'cifar10' # cifar10, cifar100, imagenet or digits
    degs = ['gauss'] # degredation: ['gauss'], ['dirichlet'] or ['digits']
    site_number = 20 
    xval = 0 # cross validation fold: 0, 1, 2, 3, 4
    comment = f'embhead'
    emb_dim = 32
    emb_exp = 'emb_head'
    experiment = 'gcpr2025'

    alpha = 1 if degs == ['dirichlet'] else None
    var = 1 if degs == ['gauss'] else None

    site_str = f'/s{site_number}' if dset == 'cifar10' and site_number != 20 else ''
    args= {
        'dataset':dset,
        'site_number':site_number,
        'deg':degs,
        'comment':comment,
        'main_dir':main_dir,
        'do_ft':False,
        'alpha':alpha,
        'cross_val_id':xval,
        'var_add':(0.005, var),
    }
    # Load already trained FedBN models, make sure the path is right
    paths = glob.glob(f'saved_models/fedbns/{dset}{site_str}/{degs[0][:4]}/*-xval{xval}-*oftFalse*')
    assert len(paths) == 1
    path = paths[0]
    fedbn_state_dict = torch.load(path, map_location='cpu')['model_state']

    # Distillation
    args['experiment'] = emb_exp
    args['emb_dim'] = emb_dim
    args['logdir'] = f'{experiment}/{dset}/emb_head{site_str}/{degs[0][:4]}'
    args['trn_state_dict'] = fedbn_state_dict
    config = get_config(**args)
    emb_state_dict, _ = main(**config)

    # Personalization
    args['only_ft'] = True
    args.pop('trn_state_dict')
    for trn_set_size in [2, 5, 10, 25, 50, None]:
        args['trn_set_size'] = trn_set_size
        args['state_dict'] = emb_state_dict
        if (degs == ['dirichlet'] or degs == ['classshard']) and dset == 'cifar10':
            args['batch_running_stats'] = False
        config = get_config(**args)
        _, _ = main(**config)
        args['batch_running_stats'] = True

if __name__ == '__main__':
    run()
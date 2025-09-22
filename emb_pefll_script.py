import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

from config import get_config
from main import main

torch.set_num_threads(8)
main_dir = 'path/to/dir'

def run():
    # To compare FedPCE to PeFLL
    experiment = 'cnn_test'
    dset = 'cifar10' # cifar10, cifar100, imagenet or digits
    degs = ['gauss'] # degredation: ['gauss'], ['dirichlet'] or ['digits']
    site_number = 20 
    xval = 0 # cross validation fold: 0, 1, 2, 3, 4
    comment = f'var{var}'
    model_name = 'cnn' # change model for this experiment
    emb_dim = 32
    emb_exp = 'emb'

    alpha = 1 if degs == ['dirichlet'] else None
    var = 1 if degs == ['gauss'] else None

    site_str = f'/s{site_number}' if dset == 'cifar10' and site_number != 20 else ''
    logdir = f'{experiment}/{dset}/fedbn{site_str}/{degs[0][:4]}'
    args= {
        'dataset':dset,
        'site_number':site_number,
        'deg':degs,
        'comment':comment,
        'logdir':logdir,
        'experiment':'fedbn',
        'main_dir':main_dir,
        'do_ft':False,
        'alpha':alpha,
        'cross_val_id':xval,
        'var_add':(0.005, var),
        'model_name':model_name
    }
    # First train FedBN
    config = get_config(**args)
    fedbn_state_dict, _ = main(**config)
    
    # Distillation
    args['experiment'] = emb_exp
    args['emb_dim'] = emb_dim
    args['logdir'] = f'{experiment}/{dset}/emb{site_str}/{degs[0][:4]}'
    args['trn_state_dict'] = fedbn_state_dict
    config = get_config(**args)
    emb_state_dict, _ = main(**config)

    # Personalization
    args['only_ft'] = True
    args.pop('trn_state_dict')
    for trn_set_size in [2, 5, 10, 25, 50, None]:
        args['trn_set_size'] = trn_set_size

        args['experiment'] = 'fedbn'
        args['logdir'] = f'{experiment}/{dset}/fedbn{site_str}/{degs[0][:4]}'
        args['state_dict'] = fedbn_state_dict
        config = get_config(**args)
        _, _ = main(**config)

        args['experiment'] = emb_exp
        args['emb_dim'] = emb_dim
        args['logdir'] = f'{experiment}/{dset}/emb{site_str}/{degs[0][:4]}'
        args['state_dict'] = emb_state_dict
        if (degs == ['dirichlet'] or degs == ['classshard']) and dset == 'cifar10':
            args['batch_running_stats'] = False
        config = get_config(**args)
        _, _ = main(**config)
        args['batch_running_stats'] = True

if __name__ == '__main__':
    run()
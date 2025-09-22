import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from config import get_config
from main import main

torch.set_num_threads(8)
main_dir = 'path/to/dir'

def run():
    # Script to run FedAvg and personalize it using full fine-tuning and LoRA
    dset = 'cifar10' # cifar10, cifar100, imagenet or digits
    degs = ['gauss'] # degredation: ['gauss'], ['dirichlet'] or ['digits']
    site_number = 20 
    xval = 0 # cross validation fold: 0, 1, 2, 3, 4
    comment = f'fedavg_and_pers'
    experiment = 'gcpr2025'
    lora_rank = 32 # lora rank

    alpha = 1 if degs == ['dirichlet'] else None
    var = 1 if degs == ['gauss'] else None
    site_str = f'/s{site_number}' if dset == 'cifar10' and site_number != 20 else ''
    logdir = f'{experiment}/{dset}/fedavg{site_str}/{degs[0][:4]}'
  
    args= {
        'dataset':dset,
        'site_number':site_number,
        'deg':degs,
        'alpha':alpha,
        'comment':comment,
        'logdir':logdir,
        'experiment':'fedavg',
        'main_dir':main_dir,
        'cross_val_id':xval,
        'var_add':(0.005, var),
    }
    config = get_config(**args)
    trn_state_dict, _ = main(**config)
    args['only_ft'] = True

    for trn_set_size in [2, 5, 10, 25, 50, None]:
        args['trn_set_size'] = trn_set_size
        args['state_dict'] = trn_state_dict
        args['experiment'] = 'fullft'
        args['logdir'] = f'{experiment}/{dset}/fullft{site_str}/{degs[0][:4]}'
        config = get_config(**args)
        main(**config)


        args['state_dict'] = None
        args['lora_st_dict'] = trn_state_dict
        args['experiment'] = 'lora'
        args['fedlora'] = lora_rank
        args['logdir'] = f'{experiment}/{dset}/lora{site_str}/{degs[0][:4]}'
        config = get_config(**args)
        main(**config)

if __name__ == '__main__':
    run()
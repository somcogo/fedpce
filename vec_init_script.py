import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from config import get_config
from main import main

torch.set_num_threads(8)
main_dir = 'path/to/dir'

def run():
    emb_vec = torch.rand(32)
    for xval in range(5):
        comment = 'randomxval'
        for tsetsize in [None]:
            path = sorted(glob.glob(f'saved_models/icml/cifar10/emb/s20/gaus/*-gaus-cifar10-s20-*edim32*xval{xval}*oftFalse-aNone.*'))
            site_dict = torch.load(path[0], map_location='cpu')['model_state']
            args= {
                'dataset':'cifar10',
                'site_number':20,
                'deg':['gauss'],
                'alpha':None,
                'comment':comment,
                'logdir':f'gcpr2025/cifar10/emb/s20/gaus/vecinit',
                'experiment':'emb',
                'main_dir':main_dir,
                'state_dict':site_dict,
                'only_ft':True,
                'trn_set_size':tsetsize,
                'cross_val_id':xval,
                'ft_emb_vec':emb_vec
            }
            config = get_config(**args)
            main(**config)

if __name__ == '__main__':
    run()
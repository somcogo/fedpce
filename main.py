import os
import random

import torch
import numpy as np

from utils.data_loader import get_dl_lists
from utils.ops import get_ft_indices, get_transforms, get_target_transforms
from training import EmbeddingTraining
from training_peffl import PeFLLTraining
from utils.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

def get_sites(deg, site_number, ft_site_number, gl_seed, main_dir, **config):
    trn_dl_list, val_dl_list = get_dl_lists(site_number=site_number, degs=deg, gl_seed=gl_seed, main_dir=main_dir, **config)
    img_tr_list = get_transforms(site_number=site_number, degs=deg, **config)
    target_tr_list = get_target_transforms(site_number=site_number, degs=deg, **config)
    if deg == ['digits']:
        site_degs = np.repeat(np.arange(4), site_number // 4)
    else:
        site_per_deg = site_number // len(deg) if type(deg) == list else site_number
        site_degs = np.repeat(np.arange(site_number // site_per_deg), site_per_deg)
    site_dict = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'img_tr': img_tr_list[ndx],
                    'target_tr': target_tr_list[ndx],
                    'deg':site_degs[ndx]}
                    for ndx in range(site_number)]

    ft_indices = get_ft_indices(site_number, ft_site_number, deg)
    trn_site_dict = [site_dict[i] for i in range(len(site_dict)) if i not in ft_indices]
    ft_site_dict = [site_dict[i] for i in range(len(site_dict)) if i in ft_indices]
    return trn_site_dict, ft_site_dict

def main(logdir, comment, deg, site_number, ft_site_number, state_dict, trn_state_dict, only_ft, ft_strategy, ft_emb_lr, ft_emb_vec, gl_seed, main_dir, fedbabu, do_ft=True, pefll=False, **config):
    log.info(comment)
    torch.manual_seed(gl_seed)
    random.seed(gl_seed)
    np.random.seed(gl_seed)
    trainer_class = PeFLLTraining if pefll else EmbeddingTraining
    trn_sites, ft_sites = get_sites(deg, site_number, ft_site_number, gl_seed, main_dir, **config)
    
    res = {}
    if len(trn_sites) > 0 and not only_ft:
        if fedbabu:
            trainer = trainer_class(logdir=logdir, comment=comment, sites=trn_sites, state_dict=trn_state_dict, main_dir=main_dir, finetuning=False, emb_vec=None, fedbabu=True, **config)
            body_acc, body_state_dict = trainer.train()
            head_comment = comment + '-headft'
            ft_logdir = logdir
            config['comm_rounds'] = 200
            config['T_max'] = 200
            config['strategy'] = 'fedper_ft'
            head_trainer = trainer_class(logdir=logdir, comment=head_comment, sites=trn_sites, state_dict=trn_state_dict, main_dir=main_dir, finetuning=True, emb_vec=None, fedbabu=False, **config)
            head_acc, state_dict = head_trainer.train(state_dict=body_state_dict)
            res['trn'] = head_acc
        else:
            trainer = trainer_class(logdir=logdir, comment=comment, sites=trn_sites, state_dict=trn_state_dict, main_dir=main_dir, finetuning=False, emb_vec=None, fedbabu=False, **config)
            acc, state_dict = trainer.train()
            res['trn'] = acc
    trn_strategy = config['strategy']

    if len(ft_sites) > 0 and (do_ft or only_ft):
        ft_comment = comment + '-' + ft_strategy
        ft_logdir = logdir
        config['comm_rounds'] = 200
        config['T_max'] = 200
        config['strategy'] = ft_strategy
        config['emb_lr'] = ft_emb_lr
        config['emb_vec'] = ft_emb_vec
        ft_trainer = trainer_class(logdir=ft_logdir, comment=ft_comment, sites=ft_sites, state_dict=state_dict, main_dir=main_dir, finetuning=True, fedbabu=False, **config)
        ft_acc, ft_state_dict = ft_trainer.train()
        res['ft'] = ft_acc
    else:
        ft_state_dict = None
    
    if len(res.keys()) > 0:
        save_path = os.path.join(main_dir, 'results', logdir)
        os.makedirs(save_path, exist_ok=True)
        res['strats'] = [trn_strategy, ft_strategy]
        torch.save(res, os.path.join(save_path, comment + '.pt'))

    return state_dict, ft_state_dict
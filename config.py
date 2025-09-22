import math

import numpy as np
import torch

def get_config(logdir,
               comment,
               experiment,
               dataset,
               site_number,
               deg,
               model_name='resnet18',
               comm_rounds=1000,
               lr=1e-4,
               weight_decay=1e-4,
               optimizer_type='adam',
               betas=(0.5,0.9),
               scheduler_mode='cosine',
               batch_size=64,
               iterations=50,
               emb_dim=32,
               ft_site_number=None,
               cross_val_id=0,
               norm_layer='bn',
               gen_hidden_layer=64,
               gen_depth=1,
               batch_running_stats=True,
               cl_per_site=2,
               alpha=None,
               feature_dim=64,
               trn_set_size=None,
               ft_emb_vec=None,
               dl_shuffle=True,
               gl_seed=0,
               part_seed=0,
               tr_gen_seed=0,
               save_model=True,
               cifar=True,
               state_dict=None,
               trn_state_dict=None,
               only_ft=False,
               do_ft=True,
               fedlora=None,
               fedbabu=False,
               fedas=False,
               lora_st_dict=None,
               pefll=False,
               main_dir='.',
               device='cuda' if torch.cuda.is_available() else 'cpu',
               var_add=(0.005, 0.1)):

    if experiment == 'emb':
        model_type = 'embedding'
        strategy = 'noemb'
        ft_strategy = 'onlyemb'
        fed_prox = 0.
    elif experiment == 'emb_head':
        model_type = 'embedding'
        strategy = 'norm_and_last'
        ft_strategy = 'norm_and_last_ft'
        fed_prox = 0.
    elif experiment == 'fedbn':
        model_type = 'vanilla'
        strategy = 'fedbn'
        ft_strategy = 'fedbn_ft'
        fed_prox = 0.
    elif experiment == 'fedavg':
        model_type = 'vanilla'
        strategy = 'all'
        ft_strategy = 'nomerge'
        fed_prox = 0.
    elif experiment == 'lora':
        model_type = 'vanilla'
        strategy = 'all'
        ft_strategy = 'lora'
        fed_prox = 0.
    elif experiment == 'fedper':
        model_type = 'vanilla'
        strategy = 'fedper'
        ft_strategy = 'fedper_ft'
        fed_prox = 0.
    elif experiment == 'fedprox':
        model_type = 'vanilla'
        strategy = 'all'
        ft_strategy = 'nomerge'
        fed_prox = 0.01
    elif experiment == 'fullft':
        model_type = 'vanilla'
        strategy = 'all'
        ft_strategy = 'all'
        fed_prox = 0.
    elif experiment == 'emb2layer':
        model_type = 'embedding'
        strategy = 'noemb'
        ft_strategy = 'onlyemb'
        fed_prox = 0.
        gen_depth = 2
    elif experiment == 'emb3layer':
        model_type = 'embedding'
        strategy = 'noemb'
        ft_strategy = 'onlyemb'
        fed_prox = 0.
        gen_depth = 3
    elif experiment == 'fedbabu':
        model_type = 'vanilla'
        strategy = 'fedper'
        ft_strategy = 'fedper_ft'
        fed_prox = 0.
        fedbabu = True
    elif experiment == 'fedas':
        model_type = 'vanilla'
        strategy = 'fedper'
        ft_strategy = 'fedper_ft'
        fed_prox = 0.
        fedas = True
    elif experiment == 'pefll':
        model_type = None
        strategy = None
        ft_strategy = ''
        fed_prox = None

    if dataset == 'cifar10':
        num_classes = 10
        in_channels = 3
    elif dataset == 'cifar100':
        num_classes = 100
        in_channels = 3
    elif dataset == 'imagenet':
        num_classes = 200
        in_channels = 3
    elif dataset == 'digits':
        num_classes = 10
        in_channels = 3

    task = 'classification'
    ft_site_number = site_number // 5 if ft_site_number is None else ft_site_number
    emb_dim = None if model_type == 'vanilla' else emb_dim
    
    fflr = 1e-4 if model_type != 'vanilla' else None
    emb_lr = 1e-1 if model_type != 'vanilla' else None
    ft_emb_lr = 1e-2 if model_type != 'vanilla' else None
    feat_dim_string = feature_dim
    feature_dim = feature_dim * np.array([1, 2, 4, 8]) if model_name == 'resnet18' else feature_dim * np.array([1, 2, 4, 8, 16])

    state_dict_str = 'N' if state_dict is None and trn_state_dict is None else 'Y'
    deg_str = ''.join([name[:4] for name in deg]) if type(deg) == list else deg

    alpha_str = str(int(math.log10(alpha))) if alpha is not None else 'None'
    var_str = str(var_add[1]).replace('.', '') if deg == ['gauss'] else 'None'

    ext_comm = f'{comment}-{experiment}-{deg_str}-{dataset}-s{str(site_number)}-fs{str(ft_site_number)}-edim{emb_dim}-xval{cross_val_id}-nl-{norm_layer}-fd{feat_dim_string}-tss{trn_set_size}-bstats{batch_running_stats}-sdict{state_dict_str}-oft{only_ft}-a{alpha_str}-var{var_str}-{model_name}'

    config = {'logdir':logdir,
            'comment':ext_comm,
            'task':task,
            'model_name':model_name,
            'model_type':model_type,
            'deg':deg,
            'site_number':site_number,
            'emb_dim':emb_dim,
            'batch_size':batch_size,
            'cross_val_id':cross_val_id,
            'comm_rounds':comm_rounds,
            'lr':lr,
            'gen_lr':fflr,
            'emb_lr':emb_lr,
            'ft_emb_lr':ft_emb_lr,
            'weight_decay':weight_decay,
            'optimizer_type':optimizer_type,
            'betas':betas,
            'scheduler_mode':scheduler_mode,
            'T_max':comm_rounds,
            'save_model':save_model,
            'strategy':strategy,
            'cifar':cifar,
            'part_seed':part_seed,
            'tr_gen_seed':tr_gen_seed,
            'dataset':dataset,
            'alpha':alpha,
            'iterations':iterations,
            'ft_strategy':ft_strategy,
            'fed_prox':fed_prox,
            'ft_site_number':ft_site_number,
            'gl_seed':gl_seed,
            'norm_layer':norm_layer,
            'batch_running_stats':batch_running_stats,
            'ft_emb_vec':ft_emb_vec,
            'cl_per_site':cl_per_site,
            'feature_dims':feature_dim,
            'trn_set_size':trn_set_size,
            'device':device,
            'var_add':var_add,
            'dl_shuffle':dl_shuffle,
            'state_dict':state_dict,
            'trn_state_dict':trn_state_dict,
            'only_ft':only_ft,
            'do_ft':do_ft,
            'main_dir':main_dir,
            'gen_hidden_layer':gen_hidden_layer,
            'gen_depth':gen_depth,
            'num_classes':num_classes,
            'channels':in_channels,
            'gen_depth':gen_depth,
            'gen_hidden_layer':gen_hidden_layer,
            'fedbabu':fedbabu,
            'fedas':fedas,
            'fedlora':fedlora,
            'lora_st_dict':lora_st_dict,
            'pefll':pefll}
    
    if pefll:
        config['lr'] = 2e-4
        config['weight_decay'] = 1e-3
        config['inner_lr'] = 2e-3
        config['inner_wd'] = 5e-5
        config['optimizer_type'] = 'adam'
        config['hyper_nhid'] = 3
        config['peffl_embed_dim'] = int(1 + site_number / 4)
        config['embed_lr'] = None
        config['hyper_hid'] = 100
        config['n_kernels'] = 16

    return config
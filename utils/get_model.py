import numpy as np
import torch
import torch.nn as nn

from models.assembler import ModelAssembler

def get_model(site_number, emb_vec, **kwargs):
    config = get_model_config(**kwargs)
    models = []
    for _ in range(site_number):
        model = ModelAssembler(**config)
        models.append(model)
    
    if config['model_type'] == 'embedding':
        mu_init = np.eye(site_number, config['emb_dim'])
        for i, model in enumerate(models):
            init_weight = torch.from_numpy(mu_init[i]).to(torch.float) if emb_vec is None else emb_vec
            model.embedding = torch.nn.Parameter(init_weight)
    return models

def get_model_config(model_name, task, feature_dims, **kwargs):
    if model_name == 'resnet18':
        config = {
            'backbone_name':'resnet',
            'layers':[2, 2, 2, 2],
            'feature_dims':feature_dims
        }
    elif model_name == 'cnn':
        config = {
            'backbone_name':'cnn',
            'nkernels':16,
            'feature_dims':[16, 32, 320, 120, 84]
        }

    if task == 'classification':
        config['head_name'] = 'classifier'

    config.update(kwargs)

    return config
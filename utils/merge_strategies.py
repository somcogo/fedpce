def get_layer_list(task, strategy, model):
    original_list = list(model.state_dict().keys())
    last_layer = 'fc' if task == 'classification' else 'head.head'

    if strategy == 'all':
        layer_list = original_list
    elif strategy == 'nomerge':
        layer_list = []
    elif strategy == 'lora':
        layer_list = []

    elif strategy == 'noemb':
        layer_list = [name for name in original_list if not ('embedding' in name)]
    elif strategy == 'onlyemb':
        layer_list = [name for name in original_list if 'embedding' in name]

    elif strategy == 'fedper':
        layer_list = [name for name in original_list if not ('embedding' in name or last_layer in name)]
    elif strategy == 'fedper_ft':
        layer_list = [name for name in original_list if 'embedding' in name or last_layer in name]

    elif strategy == 'norm_and_last':
        layer_list = [name for name in original_list if not ('embedding' in name or last_layer in name or 'norm' in name)]
    elif strategy == 'norm_and_last_ft':
        layer_list = [name for name in original_list if 'embedding' in name or last_layer in name or 'norm' in name]

    elif strategy == 'fedbn':
        layer_list = [name for name in original_list if not ('embedding' in name or 'norm' in name)]
    elif strategy == 'fedbn_ft':
        layer_list = [name for name in original_list if 'embedding' in name or 'norm' in name]

    elif strategy == 'embgennorm':
        layer_list = [name for name in original_list if not ('embedding' in name or 'generator' in name or 'norm' in name)]

    return layer_list
import os
import datetime
import math
import copy
from functools import partial

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision
import numpy as np
import minlora

from utils.logconf import logging
from utils.merge_strategies import get_layer_list
from utils.get_model import get_model
from utils.ops import transform_image
from models.embedding_functionals import BatchNormEmb

torchvision.disable_beta_transforms_warning()
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class EmbeddingTraining:
    def __init__(self, comm_rounds, logdir, lr, comment, dataset,
                 optimizer_type, scheduler_mode,
                 T_max, strategy, finetuning,
                 emb_lr, gen_lr, task, sites, model_type, betas,
                 weight_decay, state_dict, iterations,
                 fed_prox, batch_running_stats,
                 main_dir, num_classes, fedbabu, fedlora, lora_st_dict,
                 fedas,
                 **kwargs):
        
        log.info(comment)
        self.main_dir = main_dir
        self.logdir_name = logdir
        self.comment = comment
        self.dataset = dataset
        self.optimizer_type = optimizer_type
        self.scheduler_mode = scheduler_mode
        self.T_max = T_max
        self.iterations = iterations
        self.strategy = strategy
        self.model_type = model_type
        self.finetuning = finetuning
        self.num_classes = num_classes
        self.task = task
        self.state_dict = state_dict
        self.comm_rounds = comm_rounds
        self.fed_prox = fed_prox
        self.fedbabu_trn = fedbabu
        self.fedlora = fedlora
        self.fedas = fedas
        self.batch_running_stats = batch_running_stats
        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.device = kwargs['device']
        self.trn_writer = None
        self.val_writer = None

        self.trn_dls = [site['trn_dl'] for site in sites]
        self.val_dls = [site['val_dl'] for site in sites]
        self.img_trs = [site['img_tr'] for site in sites]
        self.target_trs = [site['target_tr'] for site in sites]
        self.degs = torch.tensor([site['deg'] for site in sites], device=self.device)
        if self.fedas and not self.finetuning:
            self.fim_trace_histories = [[] for site in sites]
        self.site_number = len(sites)
        self.present_classes = range(19)

        self.models = self.init_models(num_classes=num_classes, lora_st_dict=lora_st_dict, **kwargs)
        self.optims, self.emb_optims = self.init_optimizers(lr, weight_decay=weight_decay, emb_lr=emb_lr, gen_lr=gen_lr, betas=betas)
        self.schedulers, self.emb_schedulers = self.init_schedulers(emb_lr)
        self.freeze_appr_params()

        tensorboard_dir = os.path.join(main_dir, 'runs', self.logdir_name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.trn_writer = SummaryWriter(
            log_dir=tensorboard_dir + '/trn-' + self.comment)
        self.val_writer = SummaryWriter(
            log_dir=tensorboard_dir + '/val-' + self.comment)

    def init_models(self, lora_st_dict, **kwargs):
        models = get_model(self.site_number, task=self.task, model_type=self.model_type, **kwargs)

        if self.fedlora is not None:
            lora_config = {
                nn.Linear: {
                    "weight": partial(minlora.LoRAParametrization.from_linear, rank=self.fedlora),
                },
                nn.Conv2d: {
                    "weight": partial(minlora.LoRAParametrization.from_conv2d, rank=self.fedlora),
                },
            }
            for model in models:
                model.load_state_dict(lora_st_dict)
                minlora.add_lora(model, lora_config=lora_config)
        if self.device == 'cuda':
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for model in models:
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
        return models

    def init_optimizers(self, lr, weight_decay, betas, emb_lr, gen_lr):
        if self.fedlora is None:
            optims = []
            emb_optims = []
            if self.finetuning:
                all_names = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
            elif self.fedbabu_trn:
                all_names = get_layer_list(task=self.task, strategy='fedper', model=self.models[0])
            else:
                all_names = get_layer_list(task=self.task, strategy='all', model=self.models[0])
            emb_names = [name for name in all_names if 'embedding' in name]
            gen_names = [name for name in all_names if 'generator' in name]
            if self.optimizer_type == 'adam':
                opt_fn = Adam
            elif self.optimizer_type == 'adamw':
                opt_fn = AdamW

            for model in self.models:
                param_groups = []
                if len(gen_names) > 0:
                    gen_lr = lr if gen_lr is None else gen_lr
                    param_groups.append({'params':[param for name, param in model.named_parameters() if name in gen_names and name in all_names], 'lr':gen_lr})
                param_groups.append({'params':[param for name, param in model.named_parameters() if not name in emb_names and not name in gen_names  and name in all_names]})
                optim = opt_fn(params=param_groups, lr=lr, weight_decay=weight_decay, betas=betas)
                optims.append(optim)

                if emb_lr is not None:
                    emb_param_group = [param for name, param in model.named_parameters() if name in emb_names and name in all_names]
                    emb_optim = opt_fn(params=emb_param_group, lr=lr, weight_decay=weight_decay, betas=betas)
                    emb_optims.append(emb_optim)
        else:
            optims = []
            emb_optims = []
            if self.optimizer_type == 'adam':
                opt_fn = Adam
            elif self.optimizer_type == 'adamw':
                opt_fn = AdamW
            for model in self.models:
                lora_params = minlora.get_lora_params(model)
                optim = opt_fn(params=lora_params, lr=lr, weight_decay=weight_decay, betas=betas)
                optims.append(optim)

        return optims, emb_optims
    
    def init_schedulers(self, emb_lr):
        schedulers = []
        for optim in self.optims:
            if self.scheduler_mode == 'cosine':
                scheduler = CosineAnnealingLR(optim, T_max=self.T_max, eta_min=1e-6)
            schedulers.append(scheduler)
        emb_schedulers = []
        for emb_optim in self.emb_optims:
            if self.scheduler_mode == 'cosine':
                emb_scheduler = CosineAnnealingLR(emb_optim, T_max=self.T_max, eta_min=emb_lr * 1e-3)
            emb_schedulers.append(emb_scheduler)

        return schedulers, emb_schedulers
    
    def freeze_appr_params(self):
        for model in self.models:
            if self.finetuning and self.fedlora is None:
                all_names = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
                for name, param in model.named_parameters():
                    if name not in all_names:
                        param.requires_grad = False

            if self.model_type == 'embedding':
                non_norms = get_layer_list(task=self.task, strategy='embgennorm', model=model)
                for name, param in model.named_parameters():
                    if name in non_norms:
                        param.requires_grad = False

            if self.fedbabu_trn:
                head_names = get_layer_list(task=self.task, strategy='fedper_ft', model=model)
                for name, param in model.named_parameters():
                    if name in head_names:
                        param.requires_grad = False

    def train(self, state_dict=None):
        log.info("Starting {}".format(type(self).__name__))
        state_dict = self.state_dict if self.state_dict is not None else state_dict
        self.merge_models(is_init=True, state_dict=state_dict)

        saving_criterion = 0
        validation_cadence = 5

        if self.finetuning:
            val_metrics = self.do_validation(self.val_dls)
            self.log_metrics(0, 'val', val_metrics)
            metric_to_report = val_metrics['overall/accuracy'] if self.task == 'classification' else val_metrics['average/mean dice']
            log.info('Round {} of {}, accuracy/dice {}, val loss {}'.format(0, self.comm_rounds, metric_to_report, val_metrics['mean loss']))
        
        if self.strategy == 'nomerge' and self.finetuning:
            saving_criterion = max(metric_to_report, saving_criterion)
            self.save_model(0, val_metrics)
            best_state_dict = self.models[0].state_dict()
        else:
            for comm_round in range(1, self.comm_rounds + 1):
                logging_index = comm_round % 10**(math.floor(math.log(comm_round, 10))) == 0

                if comm_round == 1:
                    log.info("Round {} of {}, training on {} sites, using {} device".format(
                        comm_round,
                        self.comm_rounds,
                        len(self.trn_dls),
                        (torch.cuda.device_count() if self.device == 'cuda' else 1),
                    ))

                trn_metrics = self.do_training(self.trn_dls)
                self.log_metrics(comm_round, 'trn', trn_metrics)
                if self.model_type == 'embedding':
                    self.save_embs(comm_round, trn_metrics)

                if self.fedas and not self.finetuning:
                    self.calc_fim_trace(self.trn_dls)

                if comm_round == 1 or comm_round % validation_cadence == 0:
                    val_metrics = self.do_validation(self.val_dls)
                    self.log_metrics(comm_round, 'val', val_metrics)
                    metric_to_report = val_metrics['overall/accuracy'] if self.task == 'classification' else val_metrics['average/mean dice']
                    saving_criterion = max(metric_to_report, saving_criterion)

                    if metric_to_report==saving_criterion:
                        self.save_model(comm_round, val_metrics)
                        best_state_dict = self.models[0].state_dict()
                    if logging_index:
                        log.info('Round {} of {}, accuracy/dice {}, val loss {}'.format(comm_round, self.comm_rounds, metric_to_report, val_metrics['mean loss']))
                
                if self.scheduler_mode == 'cosine':
                    for scheduler in self.schedulers:
                        scheduler.step()
                    for emb_scheduler in self.emb_schedulers:
                        emb_scheduler.step()

                if self.site_number > 1 and not self.finetuning:
                    self.merge_models()
                    if self.fedas and not self.finetuning:
                        self.do_fedas_alignment(self.trn_dls)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        return saving_criterion, best_state_dict

    def do_training(self, trn_dls):
        for model in self.models:
            model.train()
            if (self.finetuning and not self.batch_running_stats):
                for module in model.modules():
                    if isinstance(module, BatchNormEmb):
                        module.eval()

        metrics = []
        for ndx, trn_dl in enumerate(trn_dls):
            iter_ndx = 0
            site_metrics = self.get_empty_metrics()
            while iter_ndx < self.iterations:

                for batch_tuple in trn_dl:
                    if batch_tuple[0].shape[0] == 1:
                        continue
                    self.optims[ndx].zero_grad()
                    if len(self.emb_optims) > 0:
                        self.emb_optims[ndx].zero_grad()
                    loss = self.compute_batch_loss(batch_tuple, self.models[ndx], site_metrics, 'trn', ndx)
                    loss.backward()
                    self.optims[ndx].step()
                    if len(self.emb_optims) > 0:
                        self.emb_optims[ndx].step()
                    iter_ndx += 1
                    if iter_ndx >= self.iterations:
                        break

            metrics.append(site_metrics)
        trn_metrics = self.calculate_global_metrics_from_local(metrics)

        return trn_metrics

    def do_validation(self, val_dls):
        with torch.no_grad():
            for model in self.models:
                model.eval()

            metrics = []
            for ndx, val_dl in enumerate(val_dls):
                site_metrics = self.get_empty_metrics()
                for batch_tuple in val_dl:
                    self.compute_batch_loss(batch_tuple, self.models[ndx], site_metrics, 'val', ndx)
                metrics.append(site_metrics)
            val_metrics = self.calculate_global_metrics_from_local(metrics)

        return val_metrics
    
    def do_fedas_alignment(self, trn_dls):
        for site_id, trn_dl in enumerate(trn_dls):
            self.prev_models[site_id].eval()
            self.prev_models[site_id].to(self.device)
            self.prev_models[site_id].head.use_fc = False
            self.models[site_id].train()
            self.models[site_id].head.use_fc = False

            # Get class-specific prototypes from the local model
            local_prototypes = [[] for _ in range(self.num_classes)]
            for batch_tup in trn_dl:
                batch, labels = batch_tup
                batch = batch.to(device=self.device, non_blocking=True).float()
                labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

                batch, labels = transform_image(batch, labels, 'trn', self.img_trs[site_id], self.target_trs[site_id], self.dataset)

                with torch.no_grad():
                    proto_batch = self.prev_models[site_id](batch)

                # Scatter the prototypes based on their labels
                for proto, label in zip(proto_batch, labels):
                    local_prototypes[label.item()].append(proto)

            mean_prototypes = []

            # print(f'client{self.id}')
            for class_prototypes in local_prototypes:

                if not class_prototypes == []:
                    # Stack the tensors for the current class
                    stacked_protos = torch.stack(class_prototypes)

                    # Compute the mean tensor for the current class
                    mean_proto = torch.mean(stacked_protos, dim=0)
                    mean_prototypes.append(mean_proto)
                else:
                    mean_prototypes.append(None)

            # Align global model's prototype with the local prototype
            alignment_optimizer = torch.optim.SGD(self.models[site_id].parameters(), lr=0.01)  # Adjust learning rate and optimizer as needed
            alignment_loss_fn = torch.nn.MSELoss()

            # print(f'client{self.id}')
            for _ in range(1):  # Iterate for 1 epochs; adjust as needed
                for batch_tup in trn_dl:
                    batch, labels = batch_tup
                    batch = batch.to(device=self.device, non_blocking=True).float()
                    labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

                    batch, labels = transform_image(batch, labels, 'trn', self.img_trs[site_id], self.target_trs[site_id], self.dataset)
                    global_proto_batch = self.models[site_id](batch)
                    loss = 0
                    for label in labels.unique():
                        if mean_prototypes[label.item()] is not None:
                            loss += alignment_loss_fn(
                                global_proto_batch[labels == label],
                                mean_prototypes[label.item()].unsqueeze(0).expand((labels == label).sum(), -1)
                            )
                    alignment_optimizer.zero_grad()
                    loss.backward()
                    alignment_optimizer.step()

            self.prev_models[site_id].head.use_fc = True
            self.models[site_id].head.use_fc = True
    
    def calc_fim_trace(self, trn_dls):
        for site_id, trn_dl in enumerate(trn_dls):
            self.models[site_id].eval()
            fim_trace_sum = 0
            for batch_tup in trn_dl:
                batch, labels = batch_tup
                batch = batch.to(device=self.device, non_blocking=True).float()
                labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

                batch, labels = transform_image(batch, labels, 'trn', self.img_trs[site_id], self.target_trs[site_id], self.dataset)

                pred = self.models[site_id](batch)
                nll = -torch.nn.functional.log_softmax(pred, dim=1)[range(len(labels)), labels].mean()
                grads = torch.autograd.grad(nll, self.models[site_id].parameters())
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()
            self.fim_trace_histories[site_id].append(fim_trace_sum)

    def compute_batch_loss(self, batch_tup, model, metrics, mode, site_id):
        batch, labels = batch_tup
        batch = batch.to(device=self.device, non_blocking=True).float()
        labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

        batch, labels = transform_image(batch, labels, mode, self.img_trs[site_id], self.target_trs[site_id], self.dataset)

        pred = model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss()
        xe_loss = loss_fn(pred, labels)

        prox_term = torch.tensor(0., device=self.device)
        if self.fed_prox > 0 and not self.finetuning:
            layers = get_layer_list(self.task, self.strategy, model=self.global_model)
            for (layer, glob_p), loc_p in zip(self.global_model.named_parameters(), model.parameters()):
                if layer in layers:
                    prox_term += torch.pow(torch.norm(glob_p - loc_p), 2)

        loss = xe_loss + self.fed_prox / 2 * prox_term

        metrics = self.calculate_local_metrics(pred_label, labels, loss, metrics, mode) if metrics is not None else None

        return loss.sum()
    
    def get_empty_metrics(self):
        metrics = {'loss':0,
                   'total':0}
        if self.task == 'classification':
            metrics['correct'] = 0
            for cls in range(self.num_classes):
                metrics[cls] = {'correct':0,
                                'total':0}
        return metrics
    
    def calculate_local_metrics(self, pred_label, labels, loss, local_metrics, mode):
        local_metrics['loss'] += loss.sum()
        local_metrics['total'] += pred_label.shape[0]
        if self.task == 'classification':
            correct_mask = pred_label == labels
            correct = torch.sum(correct_mask)
            local_metrics['correct'] += correct

            for cls in range(self.num_classes):
                class_mask = labels == cls
                local_metrics[cls]['correct'] += torch.sum(correct_mask[class_mask])
                local_metrics[cls]['total'] += torch.sum(class_mask)

    def calculate_global_metrics_from_local(self, local_metrics):
        eps = 1e-5

        total = sum([d['total'] for d in local_metrics])
        gl_metrics = {'mean loss':sum([d['loss'] for d in local_metrics]) / total}

        if self.task == 'classification':
            gl_metrics['overall/accuracy'] = sum([d['correct'] for d in local_metrics]) / total

            for site_ndx, local_m in enumerate(local_metrics):
                gl_metrics['accuracy_by_site/{}'.format(site_ndx)] = local_m['correct'] / local_m['total']
                gl_metrics['loss by site/{}'.format(site_ndx)] = local_m['loss'] / local_m['total']

            for cls in range(self.num_classes):
                cls_total = sum([d[cls]['total'] for d in local_metrics])
                gl_metrics['accuracy by class/{}'.format(cls)] = sum([d[cls]['correct'] for d in local_metrics]) / cls_total

        for k in gl_metrics.keys():
            gl_metrics[k] = gl_metrics[k].detach().cpu()

        return gl_metrics

    def log_metrics(self, comm_round, mode_str, metrics):
        iter_number = comm_round * self.iterations
        writer = getattr(self, mode_str + '_writer')
        for k, v in metrics.items():
            writer.add_scalar(k, scalar_value=v, global_step=iter_number)
        writer.flush()

    def save_model(self, epoch_ndx, val_metrics):
        model_file_path = os.path.join(self.main_dir, 'saved_models',
                                       self.logdir_name,
                                       f'{self.time_str}-{self.comment}.state')
        data_file_path = os.path.join(self.main_dir, 'saved_metrics',
                                      self.logdir_name,
                                      f'{self.time_str}-{self.comment}.state')
        
        os.makedirs(os.path.dirname(model_file_path), mode=0o755, exist_ok=True)
        os.makedirs(os.path.dirname(data_file_path), mode=0o755, exist_ok=True)

        data_state = {'valmetrics':val_metrics,
                      'epoch': epoch_ndx,}
        model_state = {'epoch': epoch_ndx}
        model_state['model_state'] = self.models[0].state_dict()
        layer_list = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
        for ndx, model in enumerate(self.models):
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            state_dict = model.state_dict()
            if self.finetuning:
                site_state_dict = {key: state_dict[key] for key in state_dict.keys() if key in layer_list}
            else:
                site_state_dict = {key: state_dict[key] for key in state_dict.keys() if key not in layer_list}
            model_state[ndx] = {
                'site_model_state': site_state_dict,
                'optimizer_state': self.optims[ndx].state_dict(),
                'scheduler_state': self.schedulers[ndx].state_dict(),
                }
            if len(self.emb_optims) > 0:
                model_state[ndx]['emb_optimizer_state'] = self.emb_optims[ndx].state_dict()
                model_state[ndx]['emb_scheduler_state'] = self.emb_schedulers[ndx].state_dict()
            if hasattr(model, 'embedding') and model.embedding is not None:
                data_state[ndx] = {'emb_vector':model.embedding.detach().cpu()}

        torch.save(model_state, model_file_path)
        log.debug("Saved model params to {}".format(model_file_path))
        torch.save(data_state, data_file_path)
        log.debug("Saved training metrics to {}".format(data_file_path))
    
    def save_embs(self, comm_round, metrics):
        embedding_file_path = os.path.join(self.main_dir, 'embeddings',
                                    self.logdir_name,
                                    f'{self.time_str}-{self.comment}',
                                    f'{comm_round}.pt')
        os.makedirs(os.path.dirname(embedding_file_path), mode=0o755, exist_ok=True)

        embeddings = torch.zeros((len(self.models), self.models[0].embedding.shape[0]))
        for i, model in enumerate(self.models):
            embeddings[i] = model.embedding

        dict_to_save = {'embedding':embeddings,
                        'metrics':metrics}
        torch.save(dict_to_save, embedding_file_path)
        log.debug("Saved embeddings to {}".format(embedding_file_path))

    def merge_models(self, is_init=False, state_dict=None):
        if is_init:
            if state_dict is None:
                state_dict = self.models[0].state_dict()
            state_dict.pop('embedding', None)
            for model in self.models:
                model.load_state_dict(state_dict, strict=False)
            self.global_model = copy.deepcopy(self.models[0])
            layer_list = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
        else:
            layer_list = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
            state_dicts = [model.state_dict() for model in self.models]
            updated_params = {layer: torch.zeros_like(state_dicts[0][layer]) for layer in layer_list}

            if self.fedas and not self.finetuning:
                site_weights = torch.tensor([fim_trace_history[-1] for fim_trace_history in self.fim_trace_histories])
            else:
                site_weights = torch.tensor([len(dl.dataset) for dl in self.trn_dls])
            for layer in layer_list:
                for weight, state_dict in zip(site_weights, state_dicts):
                    updated_params[layer] += weight * state_dict[layer]
                updated_params[layer] = updated_params[layer] / site_weights.sum()

            self.global_model.load_state_dict(updated_params, strict=False)
            if self.fedas:
                self.prev_models = [copy.deepcopy(model) for model in self.models]
            for model in self.models:
                model.load_state_dict(updated_params, strict=False)

if __name__ == '__main__':
    EmbeddingTraining().train()

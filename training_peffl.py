import os
import datetime
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision
import numpy as np
import minlora

from utils.logconf import logging
from utils.merge_strategies import get_layer_list
from models.pefll_models import CNNEmbed, CNNHyper, CNNTarget, EmbedHyper
from utils.ops import transform_image

torchvision.disable_beta_transforms_warning()
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class PeFLLTraining:
    def __init__(self, comm_rounds, logdir, lr, comment, dataset,
                 optimizer_type, scheduler_mode,
                 T_max, strategy, finetuning,
                 emb_lr, gen_lr, task, sites, model_type, betas,
                 weight_decay, state_dict, iterations,
                 fed_prox, batch_running_stats,
                 main_dir, num_classes, fedbabu, fedlora, lora_st_dict,
                 fedas, inner_lr, inner_wd, peffl_embed_dim, n_kernels,
                 hyper_hid, hyper_nhid, embed_lr,
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
        self.inner_lr = inner_lr
        self.inner_wd = inner_wd
        self.peffl_embed_dim = peffl_embed_dim
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
        self.site_number = len(sites)

        self.init_models(n_kernels=n_kernels, peffl_emb_dim=peffl_embed_dim, hyper_hid=hyper_hid, hyper_nhid=hyper_nhid)
        self.init_optimizers(lr=lr, embed_lr=embed_lr, weight_decay=weight_decay)

        tensorboard_dir = os.path.join(main_dir, 'runs', self.logdir_name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.trn_writer = SummaryWriter(
            log_dir=tensorboard_dir + '/trn-' + self.comment)
        self.val_writer = SummaryWriter(
            log_dir=tensorboard_dir + '/val-' + self.comment)

    def init_models(self, n_kernels, peffl_emb_dim, hyper_hid, hyper_nhid):
        self.enet = CNNEmbed(True, 10, peffl_emb_dim, self.device).to(device=self.device)
        self.hnet = CNNHyper(self.site_number, peffl_emb_dim, hidden_dim=hyper_hid, n_hidden=hyper_nhid, n_kernels=n_kernels).to(device=self.device)
        self.joint = EmbedHyper(self.enet, self.hnet).to(device=self.device)
        self.net = CNNTarget(n_kernels=n_kernels).to(device=self.device)

    def init_optimizers(self, lr, embed_lr, weight_decay):
        embed_lr = embed_lr if embed_lr is not None else lr
        optimizers = {
            'sgd': torch.optim.SGD(
                [
                    {'params': [p for n, p in self.joint.named_parameters() if 'embed' not in n]},
                    {'params': [p for n, p in self.joint.named_parameters() if 'embed' in n], 'lr': embed_lr}
                ], lr=lr, momentum=0.9, weight_decay=weight_decay
            ),
            'adam': torch.optim.Adam(params=self.joint.parameters(), lr=lr)
        }
        self.optimizer = optimizers[self.optimizer_type]
    
    def train(self, state_dicts=None):
        log.info("Starting {}".format(type(self).__name__))
        if state_dicts is not None:
            self.enet.load_state_dict(state_dicts[0])
            self.hnet.load_state_dict(state_dicts[1])
        elif self.state_dict is not None:
            self.enet.load_state_dict(self.state_dict[0])
            self.hnet.load_state_dict(self.state_dict[1])

        saving_criterion = 0
        validation_cadence = 5
        self.m, self.s = None, None

        if self.finetuning:
            val_metrics = self.do_validation(self.val_dls)
            self.log_metrics(0, 'val', val_metrics)
            metric_to_report = val_metrics['overall/accuracy'] if self.task == 'classification' else val_metrics['average/mean dice']
            log.info('PeFLL: round {} of {}, accuracy/dice {}, val loss {}'.format(0, self.comm_rounds, metric_to_report, val_metrics['mean loss']))
            saving_criterion = max(metric_to_report, saving_criterion)
            best_enet = self.enet.state_dict()
            best_hnet = self.hnet.state_dict()
        else:
            for comm_round in range(1, self.comm_rounds + 1):
                logging_index = comm_round % 10**(math.floor(math.log(comm_round, 10))) == 0

                if comm_round == 1:
                    log.info("PeFLL: round {} of {}, training on {} sites, using {} device".format(
                        comm_round,
                        self.comm_rounds,
                        len(self.trn_dls),
                        (torch.cuda.device_count() if self.device == 'cuda' else 1),
                    ))

                trn_metrics = self.do_training(self.trn_dls)
                self.log_metrics(comm_round, 'trn', trn_metrics)

                if comm_round == 1 or comm_round % validation_cadence == 0:
                    val_metrics = self.do_validation(self.val_dls)
                    self.log_metrics(comm_round, 'val', val_metrics)
                    metric_to_report = val_metrics['overall/accuracy'] if self.task == 'classification' else val_metrics['average/mean dice']
                    saving_criterion = max(metric_to_report, saving_criterion)

                    if metric_to_report==saving_criterion:
                        best_enet = self.enet.state_dict()
                        best_hnet = self.hnet.state_dict()
                        self.save_model(comm_round, val_metrics)
                    if logging_index:
                        log.info('Round {} of {}, accuracy/dice {}, val loss {}'.format(comm_round, self.comm_rounds, metric_to_report, val_metrics['mean loss']))
                
            if hasattr(self, 'trn_writer'):
                self.trn_writer.close()
                self.val_writer.close()

        return saving_criterion, (best_enet, best_hnet)

    def do_training(self, trn_dls):
        self.joint.train()

        all_metrics = []
        all_grads = []
        criteria = nn.CrossEntropyLoss()
        for site_id, trn_dl in enumerate(trn_dls):
            site_metrics = self.get_empty_metrics()

            embedding = self.calc_local_emb(dl=trn_dl, site_id=site_id, mode='trn')

            if self.m is None:
                with torch.no_grad():
                    self.m, self.s = torch.mean(embedding), torch.std(embedding)

            embedding = (embedding - self.m) / self.s
            weights = self.hnet(embedding)
            self.net.load_state_dict(weights)

            # init inner optimizer
            inner_optim = torch.optim.SGD(
                self.net.parameters(), lr=self.inner_lr, momentum=.9, weight_decay=self.inner_wd
            )

            # storing theta_i for later calculating delta theta
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            # inner updates -> obtaining theta_tilda
            self.net.train()
            for batch_tup in trn_dl:
                inner_optim.zero_grad()
                self.optimizer.zero_grad()
                batch, labels = batch_tup
                batch = batch.to(device=self.device, non_blocking=True).float()
                labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)
                batch, labels = transform_image(batch, labels, 'trn', self.img_trs[site_id], self.target_trs[site_id], self.dataset)
                pred = self.net(batch)
                loss = criteria(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 50)

                pred_label = torch.argmax(pred, dim=1)
                self.calculate_local_metrics(pred_label, labels, loss, site_metrics, 'trn')

                inner_optim.step()

            self.optimizer.zero_grad()

            final_state = self.net.state_dict()

            # calculating delta theta
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # calculating phi gradient
            joint_grads = torch.autograd.grad(
                list(weights.values()), self.joint.parameters(), grad_outputs=list(delta_theta.values()), allow_unused=True
            )
            all_grads.append(joint_grads)
            all_metrics.append(site_metrics)

        sum_grads = [0. for _ in range(len(all_grads[0]))]

        for g in all_grads:
            sum_grads = [s_i + g_i for (s_i, g_i) in zip(sum_grads, g)]

        avg_grads = [s_i / len(trn_dls) for s_i in sum_grads]
        # update hnet weights
        for p, g in zip(self.joint.parameters(), avg_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(self.joint.parameters(), 50)
        self.optimizer.step()
        trn_metrics = self.calculate_global_metrics_from_local(all_metrics)

        return trn_metrics

    def do_validation(self, val_dls):
        with torch.no_grad():
            criteria = nn.CrossEntropyLoss()
            self.joint.eval()
            all_metrics = []
            embeddings = []
            for site_id, val_dl in enumerate(val_dls):
                site_metrics = self.get_empty_metrics()
                embedding = self.calc_local_emb(dl=val_dl, site_id=site_id, mode='val')
                if self.m is None:
                    with torch.no_grad():
                        self.m, self.s = torch.mean(embedding), torch.std(embedding)
                embedding = (embedding - self.m) / self.s

                embeddings.append(embedding.cpu().detach().numpy())

                weights = self.joint.hypernet(embedding)
                self.net.load_state_dict(weights)
                self.net.eval()

                for batch_count, batch_tup in enumerate(val_dl):
                    batch, labels = batch_tup
                    batch = batch.to(device=self.device, non_blocking=True).float()
                    labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)
                    batch, labels = transform_image(batch, labels, 'val', self.img_trs[site_id], self.target_trs[site_id], self.dataset)
                    pred = self.net(batch)
                    loss = criteria(pred, labels)
                    pred_label = torch.argmax(pred, dim=1)
                    self.calculate_local_metrics(pred_label, labels, loss, site_metrics, 'val')
                all_metrics.append(site_metrics)
            val_metrics = self.calculate_global_metrics_from_local(all_metrics)

        return val_metrics
    
    def calc_local_emb(self, dl, site_id, mode):
        embedding = torch.zeros(self.peffl_embed_dim).to(self.device)
        l = 0
        for B in dl:
            batch, labels = B
            batch = batch.to(device=self.device, non_blocking=True).float()
            labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)
            batch, labels = transform_image(batch, labels, mode, self.img_trs[site_id], self.target_trs[site_id], self.dataset)
            labels = nn.functional.one_hot(labels, self.num_classes)
            l += len(batch)
            embedding += self.enet((batch, labels)).sum(0)
        embedding = embedding / l
        return embedding
    
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
        model_state['hnet'] = self.hnet.state_dict()
        model_state['enet'] = self.enet.state_dict()
        model_state['optimizer_state'] = self.optimizer.state_dict()

        torch.save(model_state, model_file_path)
        log.debug("Saved model params to {}".format(model_file_path))
        torch.save(data_state, data_file_path)
        log.debug("Saved training metrics to {}".format(data_file_path))
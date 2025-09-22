import torch
import torch.nn as nn

from models.embedding_functionals import BatchNormEmb, InstanceNormEmb
from models.resnet_with_embedding import CustomResnet
from models.heads import ClassifierHead
from models.pefll_models import CNNTargetEmb

def get_backbone(backbone_name, **model_config):
    if backbone_name == 'resnet':
        backbone = CustomResnet(**model_config)
    elif backbone_name == 'cnn':
        backbone = CNNTargetEmb(**model_config)

    return backbone

def get_head(head_name, **model_config):
    if head_name == 'classifier':
        head = ClassifierHead(**model_config)

    return head

class ModelAssembler(nn.Module):
    def __init__(self, model_type, emb_dim, **model_config):
        super().__init__()

        self.embedding = nn.Parameter(torch.zeros(emb_dim, dtype=torch.float32)) if model_type == 'emb' else None

        self.backbone = get_backbone(model_type=model_type, emb_dim=emb_dim, **model_config)
        self.head = get_head(model_type=model_type, emb_dim=emb_dim, **model_config)
        for m in self.modules():
            if type(m) in [BatchNormEmb, InstanceNormEmb]:
                m.init_norm_params()

    def forward(self, x):
        for m in self.modules():
            if type(m) in [BatchNormEmb, InstanceNormEmb]:
                m.update_weights_with_emb(self.embedding)

        features = self.backbone(x)
        x = self.head(x, features)
        return x
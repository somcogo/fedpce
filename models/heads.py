import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding_functionals import EmbNormLayer
    
class ClassifierHead(nn.Module):
    def __init__(self, feature_dims, num_classes, **kwargs):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=feature_dims[-1], out_features=num_classes)
        self.norm = EmbNormLayer(num_features=feature_dims[-1], **kwargs)
        self.use_fc = True

    def forward(self, _,  features):
        x = features[-1]

        if x.dim() > 2:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.norm(x)
        if self.use_fc:
            x = self.fc(x)
        return x
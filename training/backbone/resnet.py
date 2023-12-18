import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import os
import numpy as np



class ResNet(nn.Module):
    def __init__(self, args, n_class):
        super(ResNet, self).__init__()
        self.pretrained = args.pretrained
        self.model_choice = args.model_choice
        self.n_class = n_class
        self.base_model = self._model_choice()
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.base_model(x) 
        return x

    def _model_choice(self):
        if self.model_choice == 18:
            return models.resnet18(pretrained=self.pretrained, num_classes=self.n_class)
        if self.model_choice == 34:
            return models.resnet34(pretrained=self.pretrained, num_classes=self.n_class)
        elif self.model_choice == 50:
            return models.resnet50(pretrained=self.pretrained, num_classes=self.n_class)
        elif self.model_choice == 101:
            return models.resnet101(pretrained=self.pretrained, num_classes=self.n_class)
        elif self.model_choice == 152:
            return models.resnet152(pretrained=self.pretrained, num_classes=self.n_class)
        elif self.model_choice == 200:
            return timm.create_model('resnet200', pretrained=self.pretrained)
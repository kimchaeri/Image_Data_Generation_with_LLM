import os
import datetime
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from transformers import ViTFeatureExtractor, ViTForImageClassification
from backbone.resnet import *


def get_dataset(args):
    # Class for the task
    n_classes = 0
    normalize = None
    train_data_path, test_data_path = '', ''
    train_dataset, test_dataset = None, None
    
    if args.data_type == "generated_data":
        train_data_path = os.path.join(args.data_path, args.data_type, args.dataset)
    elif args.data_type == "augmented_data":
        train_data_path = os.path.join(args.data_path, args.data_type, args.dataset)
    elif args.data_type == "origin":
        train_data_path = os.path.join(args.data_path, "data", args.dataset, 'train')
    test_data_path = os.path.join(args.data_path, "data", args.dataset, 'test')
    
    if args.dataset=='cifar10':
        normalize = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
        cifar10_dataset_path = "data/cifar10"
        if args.data_type == 'origin':
            train_dataset = torchvision.datasets.CIFAR10(root=cifar10_dataset_path, train=True, download=True, transform=normalize)
        test_dataset = torchvision.datasets.CIFAR10(root=cifar10_dataset_path, train=False, download=True, transform = normalize)

    elif args.dataset=='cifar100':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        cifar100_dataset_path = "data/cifar100"
        if args.data_type == 'origin':
            train_dataset = torchvision.datasets.CIFAR10(root=cifar100_dataset_path, train=True, download=True, transform=normalize)
        test_dataset = torchvision.datasets.CIFAR100(root=cifar100_dataset_path, train=False, download=True, transform = normalize)

    elif args.dataset=='cub2011':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48562077, 0.49943104, 0.43238935], [0.17436638, 0.1736659, 0.18543857])
        ])

    elif args.dataset=='dtd':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5342086, 0.47681832, 0.42735654], [0.1527844, 0.15404493, 0.14999966])
        ])

    elif args.dataset=='oxfordpets':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48043793, 0.44833282, 0.39613166], [0.23136571, 0.22845998, 0.23037408])
        ])
        

    elif args.dataset=='102flowers':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5144969, 0.4092727, 0.3269492], [0.2524271, 0.20233019, 0.20998113])
        ])
        

    elif args.dataset=='eurosat':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.3444887, 0.38034907, 0.40781093], [0.0900263, 0.061832733, 0.051150024])
        ])
        

    elif args.dataset=='miniimagenet':
        normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4733471, 0.44912496, 0.4034593], [0.22521427, 0.2207067, 0.22094156])
        ])
        
    if train_dataset is None:
        train_dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = normalize)
    if test_dataset is None:
        test_dataset = torchvision.datasets.ImageFolder(root = test_data_path, transform = normalize)

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    n_classes = len(class_name)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    print(len(train_dataset))
    print(class_name)
    print(len(class_name))
    
    return train_dataloader, test_dataloader, n_classes




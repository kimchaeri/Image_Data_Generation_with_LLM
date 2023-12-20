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
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy, RandAugment

from transformers import ViTFeatureExtractor, ViTForImageClassification
from backbone.resnet import *
from utils.transforms import *

def get_data_path(args):
    classified_path = ''
    if args.use_classifier:
        classified_path = "_correct_predictions"
        
    if args.data_type == "generated_data":
        train_data_path = os.path.join(args.data_path, args.data_type, args.dataset + classified_path)
    elif args.data_type == "origin":
        train_data_path = os.path.join(args.data_path, "data", args.dataset, 'train')
    elif args.data_type == "augmented_data":
        train_data_path = [os.path.join(args.data_path, "data", args.dataset, "train"), os.path.join(args.data_path, "gen_data", args.dataset + classified_path)]
    elif args.data_type == "gpt4":
        train_data_path = [os.path.join(args.data_path, "data", args.dataset, "train"), os.path.join(args.data_path, "gen_data_gpt4", args.dataset + classified_path)]
    elif args.data_type == "my_data":
        train_data_path = [os.path.join("/data5/Data_Generation/data", args.dataset, "train"), os.path.join(args.data_path, args.dataset+ classified_path)]
    test_data_path = os.path.join("/data5/Data_Generation/data", args.dataset, 'test')
    
    return train_data_path, test_data_path
    
def get_dataset(args):
    # Class for the task
    n_classes = 0
    class_name = []
    origin_dataset, train_dataset, test_dataset = None, None, None

    train_data_path, test_data_path = get_data_path(args)
    train_transform, test_transform = get_transforms(args)
    
    # Get train dataset
    if type(train_data_path) == str:
        train_dataset = torchvision.datasets.ImageFolder(root = train_data_path, transform = train_transform)
    elif type(train_data_path) == list:
        origin_dataset = torchvision.datasets.ImageFolder(root = train_data_path[0], transform = train_transform)
        generated_dataset = torchvision.datasets.ImageFolder(root = train_data_path[1], transform = train_transform)
        if not args.data_num == 0:
            print("It is few shot learning..")
            del_list = []
            cur_class_idx = 0
            check_idx = 0
            count_data_num = 1
            for i, data in enumerate(generated_dataset):
                cur_class_idx = data[1]
                if cur_class_idx != check_idx:
                    check_idx = cur_class_idx
                    count_data_num = 1
                if count_data_num > args.data_num and cur_class_idx == check_idx:
                    del_list.append(generated_dataset.imgs[i])
                else:
                    count_data_num += 1
            for data in del_list:
                generated_dataset.imgs.remove(data)

        train_dataset = ConcatDataset([origin_dataset, generated_dataset])
    
    # Get test dataset
    if args.dataset == "cifar10":
        test_dataset = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform = test_transform)
    if args.dataset == "cifar100":
        test_dataset = torchvision.datasets.CIFAR100(root="data/cifar100", train=False, download=True, transform = test_transform)
    else:
        test_dataset = torchvision.datasets.ImageFolder(root = test_data_path, transform = test_transform)
        
    class_name = test_dataset.classes
    #class_name = [cls_name.lower() for cls_name in class_name]
    n_classes = len(class_name)
    print(train_dataset)
    print(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    if not args.data_type == "origin":
        print("num of original dataset = ", len(origin_dataset))
        print("num of generated dataset = ", len(generated_dataset))
    print("num of total dataset = ", len(train_dataset))
    print(class_name)
    print(len(class_name))
    
    return train_dataloader, test_dataloader, n_classes, class_name



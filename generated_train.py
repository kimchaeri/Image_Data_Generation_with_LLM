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

from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from transformers import CLIPProcessor, CLIPModel

from utils.args import get_args
from utils.get_dataset import get_dataset
from backbone.resnet import *

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def get_optimizer(model, lr = 0.01, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

def get_triangular_lr(lr_low, lr_high, iterations):
    iter1 = int(0.35*iterations)
    iter2 = int(0.85*iter1)
    iter3 = iterations - iter1 - iter2
    delta1 = (lr_high - lr_low)/iter1
    delta2 = (lr_high - lr_low)/(iter1 -1)
    lrs1 = [lr_low + i*delta1 for i in range(iter1)]
    lrs2 = [lr_high - i*(delta1) for i in range(0, iter2)]
    delta2 = (lrs2[-1] - lr_low)/(iter3)
    lrs3 = [lrs2[-1] - i*(delta2) for i in range(1, iter3+1)]
    return lrs1+lrs2+lrs3


def train(args, model, train_dataloader, test_dataloader, device, lr_low= 1e-7, lr_high=1*1e-5):
    seed_everything(42)
    start_time = datetime.datetime.now()
    loss = None
    
    criterion = nn.CrossEntropyLoss()
    '''
    exp_num = 1
    '''
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    '''
    exp_num = 2
    '''
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    '''
    exp_num = 3
    '''
    #iterations = num_epochs*len(train_dataloader)
    #lrs = get_triangular_lr(lr_low, lr_high, iterations)
    
    for epoch in range(args.epoch):  
        model = model.to(device)
        epoch_loss = 0.0
        running_loss = 0.0
        
        folder_name = args.data_type + "_" + args.dataset
        if args.pretrained:
            folder_name = folder_name + "_pretrained"
        checkpoints_dir = os.path.join(args.checkpoint_folder, folder_name)
        
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        correct_tr = 0
        total_tr = 0
        for i, data in enumerate(train_dataloader):
            #optimizer = get_optimizer(model, lr = lrs[i], wd =0)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            if args.model_type == "Resnet":
                loss = criterion(outputs, labels)
            elif args.model_type == "ViT":
                loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            running_loss += loss.item()

            if (i + 1) % 100 == 0: 
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                epoch_loss += running_loss
                running_loss = 0.0
        
        if epoch % args.save_every == 0:
            checkpoint_name = "-".join(["checkpoint", str(epoch) + ".pt"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                os.path.join(checkpoints_dir, checkpoint_name),
            )

        correct_te = 0
        total_te = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                if args.model_type == "Resnet":
                    _, predicted = torch.max(outputs, 1)
                elif args.model_type == "ViT":
                    _, predicted = torch.max(outputs.logits, 1)
                total_te += labels.size(0)
                correct_te += (predicted == labels).sum().item()
        
        print("Accuracy of the network on test set at epoch %d: %.3f %%" % (epoch, 100 * correct_te / total_te))

    total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
    print("Finished training in %.2f minutes" % total_minutes)
    
    

def get_model(args, n_classes):
    model = None
    
    if args.model_type == "Resnet":
        model = ResNet(pre_trained=args.pretrained, n_class=n_classes, model_choice = args.model_choice)

    elif args.model_type == "ViT":
        if args.pretrained:
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
            new_classifier = nn.Sequential(
                nn.Linear(model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_classes)
            )
            model.classifier = new_classifier

        else:
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=n_classes)
            model = ViTForImageClassification(config=config)
            model.config.num_labels = n_classes


    elif args.model_type == "CLIP_ViT":
        if args.pretrained:
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            new_classifier = nn.Sequential(
                nn.Linear(model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_classes)
            )
            model.classifier = new_classifier

    return model



if __name__ == "__main__":
    
    # Bring args
    args = get_args()
    
    # Set device
    if torch.cuda.is_available():
        device = 'cuda'
        print(device)
    else:
        device = 'cpu'
        print(device)
    
    train_dataloader, test_dataloader, n_classes = get_dataset(args)
    model = get_model(args, n_classes)
    
    train(args, model, train_dataloader, test_dataloader, device)


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, help='dataset', default="caltech101")
argparser.add_argument('--epoch', type=int, help='training epoch', default=101)
argparser.add_argument('--checkpoint_folder', type=str, help='checkpoint save folder', default=None)
argparser.add_argument('--model_choice', type=int, help='number of layers in the model', default=34)
argparser.add_argument('--pretrained', type=str2bool, help='use pretrained model or not', default=False)
argparser.add_argument('--save_every', type=int, help='save point of checkpoints', default=10)
args = argparser.parse_args()
print(args)


if args.dataset=='cifar10':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    cifar10_dataset_path = "data/cifar10"
    train_dataset = torchvision.datasets.CIFAR10(root=cifar10_dataset_path, train=True, download=True, transform=normalize)
    test_dataset = torchvision.datasets.CIFAR10(root=cifar10_dataset_path, train=False, download=True, transform=normalize)

    print(train_dataset)
    print(test_dataset)

    net = ResNet(pre_trained=args.pretrained, n_class=10, model_choice = args.model_choice)

if args.dataset=='cifar100':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
       transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])
    
    cifar100_dataset_path = "data/cifar100"
    train_dataset = torchvision.datasets.CIFAR100(root=cifar100_dataset_path, train=True, download=True, transform=normalize)
    test_dataset = torchvision.datasets.CIFAR100(root=cifar100_dataset_path, train=False, download=True, transform=normalize)

    print(train_dataset)
    print(test_dataset)

    net = ResNet(pre_trained=args.pretrained, n_class=100, model_choice = args.model_choice)

if args.dataset=='cub2011':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.48562077, 0.49943104, 0.43238935], [0.17436638, 0.1736659, 0.18543857])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/CUB_200_2011/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/CUB_200_2011/test', transform = normalize)
    print(len(train_dataset))

    net = ResNet(pre_trained=args.pretrained, n_class=200, model_choice = args.model_choice)

if args.dataset=='dtd':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5342086, 0.47681832, 0.42735654], [0.1527844, 0.15404493, 0.14999966])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/DTD/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/DTD/test', transform = normalize)
    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

#already fixed split
if args.dataset=='102flowers':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5144969, 0.4092727, 0.3269492], [0.2524271, 0.20233019, 0.20998113])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/102flowers/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/102flowers/test', transform = normalize)
    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

if args.dataset=='food101':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5449867, 0.44349685, 0.34360173], [0.23354195, 0.24430154, 0.24236311])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/food101/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/food101/test', transform = normalize)
    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

if args.dataset=='eurosat':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.3444887, 0.38034907, 0.40781093], [0.0900263, 0.061832733, 0.051150024])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/eurosat/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/eurosat/test', transform = normalize)
    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

if args.dataset=='miniimagenet':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4733471, 0.44912496, 0.4034593], [0.22521427, 0.2207067, 0.22094156])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/miniimagenet/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/miniimagenet/test', transform = normalize)
    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

if args.dataset=='oxfordpets':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.48043793, 0.44833282, 0.39613166], [0.23136571, 0.22845998, 0.23037408])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/oxfordpets/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/oxfordpets/test', transform = normalize)

    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

if args.dataset=='caltech101':
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.54520583, 0.52520037, 0.4967313], [0.24364278, 0.24105375, 0.24355316])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/caltech101/train', transform = normalize)
    test_dataset = torchvision.datasets.ImageFolder(root = '/home/s20225103/Data_Generation/data/caltech101/test', transform = normalize)

    print(len(train_dataset))
    print(len(test_dataset))

    class_name = train_dataset.classes
    class_name = [cls_name.lower() for cls_name in class_name]
    print(class_name)
    print(len(class_name))

    net = ResNet(pre_trained=args.pretrained, n_class=len(class_name), model_choice = args.model_choice)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

if torch.cuda.is_available():
    device = 'cuda'
    print(device)
else:
    device = 'cpu'
    print(device)


def train(net, num_epochs, train_dataloader, test_dataloader, device, checkpoint_folder, save_every, lr_low= 1e-7, lr_high=1*1e-5):
    seed_everything(42)
    start_time = datetime.datetime.now()
    
    
    criterion = nn.CrossEntropyLoss()
    '''
    exp_num = 1
    '''
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    '''
    exp_num = 2
    '''
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    '''
    exp_num = 3
    '''
    #iterations = num_epochs*len(train_dataloader)
    #lrs = get_triangular_lr(lr_low, lr_high, iterations)
    
    for epoch in range(num_epochs):  
        net = net.to(device)
        epoch_loss = 0.0
        running_loss = 0.0
        
        checkpoints_dir = os.path.join("checkpoints", checkpoint_folder)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        correct_tr = 0
        total_tr = 0
        for i, data in enumerate(train_dataloader):
            #optimizer = get_optimizer(net, lr = lrs[i], wd =0)

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            running_loss += loss.item()

            if (i + 1) % 100 == 0: 
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                epoch_loss += running_loss
                running_loss = 0.0
        
        if epoch % save_every == 0:
            checkpoint_name = "-".join(["checkpoint", str(epoch) + ".pt"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
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
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total_te += labels.size(0)
                correct_te += (predicted == labels).sum().item()
        
        print("Accuracy of the network on test set at epoch %d: %.3f %%" % (epoch, 100 * correct_te / total_te))

    total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
    print("Finished training in %.2f minutes" % total_minutes)

train(net, args.epoch, train_dataloader, test_dataloader, device, args.checkpoint_folder, args.save_every)
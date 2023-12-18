
import torchvision.transforms as transforms

from utils.autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy


def get_transforms(args):

    norm = {'cifar10' : ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), 'cifar100' : ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    'cub2011' : ([0.48562077, 0.49943104, 0.43238935], [0.17436638, 0.1736659, 0.18543857]), 'dtd' : ([0.5342086, 0.47681832, 0.42735654], [0.1527844, 0.15404493, 0.14999966]),
    'oxfordpets' : ([0.48043793, 0.44833282, 0.39613166], [0.23136571, 0.22845998, 0.23037408]), '102flowers' : ([0.5144969, 0.4092727, 0.3269492], [0.2524271, 0.20233019, 0.20998113]),
    'eurosat' : ([0.3444887, 0.38034907, 0.40781093], [0.0900263, 0.061832733, 0.051150024]), 'miniimagenet' : ([0.4733471, 0.44912496, 0.4034593], [0.22521427, 0.2207067, 0.22094156]),
    'caltech101' : ([0.54520583, 0.52520037, 0.4967313], [0.24364278, 0.24105375, 0.24355316]), 
    }

    train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
        ])

    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
        ])
    
    
    if args.aug_strategy == "AA":
        if args.dataset == "cifar10" or args.dataset == "cifar100":
            train_transform = transforms.Compose([
                    CIFAR10Policy(), 
                    transforms.ToTensor(), 
                ])
        else:
            train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    ImageNetPolicy(), 
                    transforms.ToTensor(),
                    transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
                ])


    if args.aug_strategy=="Nothing" or args.aug_strategy=='mixup' or args.aug_strategy=='cutmix':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
        ])
    if args.aug_strategy=='rotate':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
        ])
    if args.aug_strategy=='crop':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(128, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
        ])
    if args.aug_strategy=='horizontal_flip':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
        ])
    if args.aug_strategy=='crop_rotate_horizontal_flip' or args.aug_strategy=='crop_rotate_horizontal_flip_mixup':
        if args.model_type == "ViT":
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=30),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm[args.dataset][0], norm[args.dataset][1])
            ])
    return train_transform, test_transform


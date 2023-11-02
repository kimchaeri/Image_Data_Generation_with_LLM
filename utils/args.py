import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset', choices=['cifar10', 'cifar100', 'cub2011', 'dtd', 'oxfordpets', '102flowers', 'eurosat', 'miniimagenet'], default=None)
    argparser.add_argument('--data_path', type=str, help='data path', default='/home/s20225103/Data_Generation')
    argparser.add_argument('--data_type', type=str, help='data type', choices=['generated_data', 'origin'], default='generated_data')
    
    argparser.add_argument('--epoch', type=int, help='training epoch', default=101)
    argparser.add_argument('--checkpoint_folder', type=str, help='checkpoint save folder', default="/home1/s20225168/cvpr2023/checkpoints")
    argparser.add_argument('--model_choice', type=int, help='number of layers in the model', default=34)
    argparser.add_argument('--pretrained', type=str2bool, help='use pretrained model or not', default=False)
    argparser.add_argument('--save_every', type=int, help='save point of checkpoints', default=10)
    argparser.add_argument('--model_type', type=str, help='model to use', choices=['Resnet', 'ViT'], default='Resnet')
    args = argparser.parse_args()
    print(args)
    
    return args
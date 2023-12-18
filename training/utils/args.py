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
    argparser.add_argument('--dataset', type=str, help='dataset', choices=['cifar10', 'cifar100', 'cub2011', 'dtd', 'oxfordpets', '102flowers', 'eurosat', 'miniimagenet', 'caltech101'], default="oxfordpets")
    argparser.add_argument('--data_path', type=str, help='data path', default='/home/s20225103/Data_Generation')
    argparser.add_argument('--data_type', type=str, help='data type', choices=['generated_data', 'origin', 'augmented_data', 'generated_data_gpt3', 'gpt4', 'my_data'], default='origin')
    argparser.add_argument('--data_num', type=int, help='number of data to use', default=0)
    argparser.add_argument('--aug_strategy', type=str, help='augmentation strategy', choices=['mixup', 'cutmix', 'crop', 'rotate', 'horizontal_flip', 'crop_rotate_horizontal_flip', 'crop_rotate_horizontal_flip_mixup', "Nothing", "AA", "FA", "RA", "PBA"], default="Nothing")
    argparser.add_argument('--mixup_alpha', type=int, help='alpha', default=1)
    argparser.add_argument('--epoch', type=int, help='training epoch', default=101)
    argparser.add_argument('--checkpoint_folder', type=str, help='checkpoint save folder', default="/home1/s20225168/cvpr2023/Image_Data_Generation_with_LLM/training/checkpoints")
    argparser.add_argument('--model_choice', type=int, help='number of layers in the model', default=34)
    argparser.add_argument('--pretrained', type=str2bool, help='use pretrained model or not', default=False)
    argparser.add_argument('--save_every', type=int, help='save point of checkpoints', default=50)
    argparser.add_argument('--model_type', type=str, help='model to use', choices=['Resnet', 'ViT', 'CLIP_ViT'], default='Resnet')
    argparser.add_argument('--batch_size', type=int, help='batch size', default=64)
    argparser.add_argument('--exp_num', type=int, help='experiment number', default=0)
    argparser.add_argument('--prompt_num', type=int, help='prompt number', default=4)
    argparser.add_argument('--generate_num', type=int, help='how many images to generate', default=2)
    argparser.add_argument('--diffusion_model', type=str, help='diffusion model to use', choices=['1.4', 'sdxl', '1.4_finetuned'], default='1.4_finetuned')
    argparser.add_argument('--use_classifier', action='store_true', help='Set the flag to True')

    args = argparser.parse_args()
    print(args)
    
    return args
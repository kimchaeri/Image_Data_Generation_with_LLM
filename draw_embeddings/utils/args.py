import argparse

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path_txt', type=str, help='dataset', default="./image_paths.txt")
    argparser.add_argument('--dst_path', type=str, help='dataset', default="./embedding_images")
    argparser.add_argument('--file_name', type=str, help='dataset', default="embedding")
    argparser.add_argument('--device', type=str, help='device is?', default="cuda")
    argparser.add_argument('--perplexity', type=int, help='If error occur, reduce this value', default=30)
    argparser.add_argument('--pca', action='store_true')
    argparser.add_argument('--tsne', action='store_true')
    args = argparser.parse_args()
    print(args)
    
    return args
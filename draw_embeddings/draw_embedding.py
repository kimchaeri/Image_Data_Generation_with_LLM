import os
import torch
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.args import get_args
from utils.utils import get_model, draw_images, get_img_folders

if __name__ == "__main__":
    args = get_args()
    
    model, feature_extractor = get_model()
    image_folders, labels = get_img_folders(args)
    
    draw_images(args, model, feature_extractor, image_folders, labels)

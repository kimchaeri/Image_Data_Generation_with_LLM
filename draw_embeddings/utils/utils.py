import os
import copy
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel

def get_model():
    model_name = "google/vit-base-patch16-224-in21k"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    # model = ViTForImageClassification.from_pretrained(model_name)
    return model, feature_extractor

def get_img_folders(args):
    # get data_path
    path_txt = args.path_txt
    
    image_folders = []
    labels = []
    with open(path_txt, "r") as file:
        path_lines = file.read().split('\n')
    
    for i, path_line in enumerate(path_lines):
        if i == 0 or i == 2:
            continue
        else:
            split_line = path_line.split(' ')
            image_folders.append(split_line[0])
            labels.append(split_line[1])
            
    return image_folders, labels
    
    
def draw_images(args, model, feature_extractor, image_folders, labels):
    # colors
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    # extract image features
    features = []
    save_num_images = [0]
    num_images = 0
    for image_folder in image_folders:
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path)
            
            check_dim = np.array(img)
            if len(check_dim.shape) < 3:
                # num += 1
                check_dim = np.expand_dims(check_dim, axis=-1)
                check_dim = np.repeat(check_dim, 3, axis=-1)
                img = check_dim
            try:
                image = feature_extractor(images=img, return_tensors="pt")
                output = model(**image, output_hidden_states=True)
                feature_vector = output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                features.append(feature_vector)
                num_images += 1
            except:
                pass
        save_num_images.append(copy.deepcopy(num_images))
            
    features_array = np.array(features)
    
    if not os.path.exists(args.dst_path):
        os.makedirs(args.dst_path)
    
    if args.pca:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features_array)
        
        avg_poses = []
        distances = []
        plt.title("2D Embedding of Images using ViT and PCA")
        plt.figure(figsize=(8, 6))
        for i, num_image in enumerate(save_num_images):
            if i == 0:
                continue
            else:
                plt.scatter(reduced_features[save_num_images[i-1]:save_num_images[i], 0], reduced_features[save_num_images[i-1]:save_num_images[i], 1], color=colors[i-1], label=labels[i-1])
                output_list = [sum(inner_list) / len(inner_list) for inner_list in zip(*reduced_features[save_num_images[i-1]:save_num_images[i]])]
                avg_poses.append(output_list)

        for i, avg_pos in enumerate(avg_poses):
            if i == 0:
                continue
            else:
                distance = np.sqrt((avg_poses[0][0] - avg_pos[0])**2 + (avg_poses[0][1] - avg_pos[1])**2)
                text = labels[0] + " - " + labels[i] + " = " + str(distance)
                distances.append(text)

        plt.legend()
        plt.savefig(os.path.join(args.dst_path, args.file_name + "_pca.png"))
        plt.show()
        f= open(os.path.join(args.dst_path, args.file_name + "_pca.txt"),"w+")
        for distance in distances:
            f.write(distance + "\n")
        


    if args.tsne:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42)
        reduced_features = tsne.fit_transform(features_array)

        avg_poses = []
        distances = []
        plt.figure(figsize=(8, 6))
        plt.title('t-SNE 2D Embedding')
        for i, num_image in enumerate(save_num_images):
            if i == 0:
                continue
            else:
                plt.scatter(reduced_features[save_num_images[i-1]:save_num_images[i], 0], reduced_features[save_num_images[i-1]:save_num_images[i], 1], color=colors[i-1], label=labels[i-1])
                output_list = [sum(inner_list) / len(inner_list) for inner_list in zip(*reduced_features[save_num_images[i-1]:save_num_images[i]])]
                avg_poses.append(output_list)
                
        for i, avg_pos in enumerate(avg_poses):
            if i == 0:
                continue
            else:
                distance = np.sqrt((avg_poses[0][0] - avg_pos[0])**2 + (avg_poses[0][1] - avg_pos[1])**2)
                text = labels[0] + " - " + labels[i] + " = " + str(distance)
                distances.append(text)
                
        plt.legend()
        plt.savefig(os.path.join(args.dst_path, args.file_name + "_tsne.png"))
        plt.show()
        f= open(os.path.join(args.dst_path, args.file_name + "_tsne.txt"),"w+")
        for distance in distances:
            f.write(distance + "\n")


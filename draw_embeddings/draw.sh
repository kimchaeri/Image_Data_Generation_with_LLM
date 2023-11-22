#!/bin/bash 

#PBS -l select=1:ncpus=4:ngpus=1

#PBS -N G1C4_BJY_embeddings

#PBS -q pleiades3

#PBS -r n 

#PBS -j oe 

source activate cvpr

cd /home1/s20225168/cvpr2023/Image_Data_Generation_with_LLM/draw_embeddings

python draw_embedding.py --path_txt "./image_paths.txt" --dst_path "./embedding_images" --tsne --file_name "embedding" --perplexity 5
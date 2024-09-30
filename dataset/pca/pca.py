import faiss
from open_clip import create_model_from_pretrained
import torch
from PIL import Image

from tqdm import tqdm


from pathlib import Path
import numpy as np
import glob
import sys
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from myvlm.common import VALID_IMAGE_EXTENSIONS

MODEL_NAME = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"

def image_path_load(concepts_name, data_type = 'positives'):
    data_root_path = Path('./dataset/data_concept_head/')
    data_root_path = data_root_path / data_type

    data_path = []
    for concept in sorted(concepts_name):
        concept_path = data_root_path / concept
        # print(concept_path)
        # data_path[concept] = glob.glob(os.path.join(concept_path, "*.jpg"))
        data_path.append([concept,[str(p) for p in concept_path.glob('*') if p.suffix.lower() in VALID_IMAGE_EXTENSIONS][:3]])

    return data_path


def pca_embedding(positives_image_path, negatives_image_path, device = 'cuda'):

    # model load
    model, preprocess = create_model_from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print(model)
    # exit()

    image_embeddings = []
    image_class = []
    image_type = []

    for index, concept_info in enumerate(tqdm(positives_image_path)):
        concept = concept_info[0]
        data_concept = concept_info[1]
        for image_path in data_concept:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_embedding = model.encode_image(image)
                    # print(image_embedding)
                    # print(image_embedding.size())
                    image_embeddings.append(image_embedding)
                    image_class.append(index)
                    image_type.append(1)

    for index, concept_info in enumerate(tqdm(negatives_image_path)):
        concept = concept_info[0]
        data_concept = concept_info[1]
        for image_path in data_concept:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_embedding = model.encode_image(image)
                    # print(image_embedding)
                    # print(image_embedding.size())
                    image_embeddings.append(image_embedding)
                    image_class.append(index)
                    image_type.append(0)
    
    print(image_embeddings)
    image_embeddings = torch.cat(image_embeddings).cpu().numpy()
    image_class = image_class
    image_type = image_type

    # print(image_embeddings)

    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(image_embeddings)

    markers = ["o", "s", "^", "P", "D"]
    colors = ['red', 'blue']

    plt.figure(figsize=(10, 10))

    for i, Iclass, Itype in zip(range(len(embedding_pca)), image_class, image_type):
        color = colors[Itype]
        marker = markers[Iclass]

        plt.scatter(embedding_pca[i, 0], embedding_pca[i, 1], 
                c=color, marker=marker, edgecolor='black', s=100, label=f'Class {Itype}' if i == 0 else "")
        
    # 범례 설정 (중복 방지를 위해 첫 번째 샘플만 레이블 표시)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='billy_dog / positives'),
           plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='cat_statue / positives'),
           plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='dangling_child / positives'),
           plt.Line2D([0], [0], marker='P', color='w', markerfacecolor='blue', markersize=10, label='running_shoes / positives'),
           plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=10, label='iverson_funko_pop / positives'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='billy_dog / negatives'),
           plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='cat_statue / negatives'),
           plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='dangling_child / negatives'),
           plt.Line2D([0], [0], marker='P', color='w', markerfacecolor='red', markersize=10, label='running_shoes / negatives'),
           plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=10, label='iverson_funko_pop / negatives')]
    
    plt.legend(handles=handles)
    plt.title('PCA of CLIP Image Embeddings with Positives/Negatives and Class Markers')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    plt.savefig('./dataset/pca/pca_image.png')

    plt.show()

    # print(len(image_embeddings))
    # print(embedding_pca)
    
    # print(image_class)

    # for index, embedding in zip(image_class, image_embeddings):
        # print(index, embedding.size())


if __name__ == '__main__':

    # 논문에서 pca로 나타낸 concept들
    concepts_name = ['billy_dog', 'cat_statue', 'dangling_child', 'iverson_funko_pop', 'running_shoes']
    positives_image_path = image_path_load(concepts_name, 'positives')
    negatives_image_path = image_path_load(concepts_name, 'negatives')

    print(positives_image_path)
    print(negatives_image_path)

    positives_embedding = pca_embedding(positives_image_path, negatives_image_path)
    




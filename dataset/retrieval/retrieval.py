import faiss
from open_clip import create_model_from_pretrained
import torch
from PIL import Image

from tqdm import tqdm


from pathlib import Path
import numpy as np
import glob
import os
import sys

import json

import shutil

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from myvlm.common import VALID_IMAGE_EXTENSIONS

MODEL_NAME = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"

def save_embedding_info(dataset_path, device = 'cuda'):
    # model load
    model, preprocess = create_model_from_pretrained(MODEL_NAME, precision='fp16')
    model.to(device)
    model.eval()

    # image list load
    image_list = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
    # print(image_list)

    image_embeddings = []

    for image_path in tqdm(image_list):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_embedding = model.encode_image(image)
                print(image_embedding)
                image_embeddings.append(image_embedding)
    # print(len(image_embeddings))
    # print(image_embeddings[0].size())

    # cpu로 이동해서 numpy로 변경
    image_embeddings = torch.cat(image_embeddings).cpu().numpy()
    # data 형 변환(faiss vector가 float32만 호환)
    image_embeddings = image_embeddings.astype(np.float32)
    # print(image_embeddings.shape)

    # print(image_embeddings[0].shape)

    d = image_embeddings[0].shape[0]
    # faiss vector 생성
    faiss_index = faiss.IndexFlatL2(d)
    image_embeddings = np.vstack(image_embeddings)

    faiss_index.add(image_embeddings)

    faiss.write_index(faiss_index, "./dataset/retrieval/coco_faiss_index.bin")



def retrieval_top_k(concept, k, output_dir, image_list, device='cuda'):

    embedding_path = Path('./dataset/data_concept_embedding/') / concept

    output_dir = Path(output_dir) / concept
    output_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess = create_model_from_pretrained(MODEL_NAME, precision='fp16')
    model.to(device)
    model.eval()

    # 저장된 FAISS 인덱스 파일 로드
    index = faiss.read_index("./dataset/retrieval/coco_faiss_index.bin")
    
    query_image_paths = sorted([str(p) for p in embedding_path.glob('*') if p.suffix.lower() in VALID_IMAGE_EXTENSIONS])
    # print(query_image_path)
    # exit()
    
    data = {}

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for image_path in query_image_paths:
                query_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                query_embedding = model.encode_image(query_image).cpu().numpy()
    
                query_embedding = query_embedding.astype(np.float32)

                distances, indices = index.search(query_embedding, k)
                retrieval_image = [image_list[int(idx)] for idx in indices[0].tolist()]
                for image in retrieval_image:
                    shutil.copy(image, './dataset/retrieval/' + concept + '/'+ image.split('/')[-1])
                data[image_path] = [distances[0].tolist(), indices[0].tolist(), retrieval_image]

    with open(output_dir/'retrieval.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data

# # CLIP 모델과 변환기 로드
# model, preprocess = clip.load("ViT-B/32", device="cuda")

# # 검색할 이미지 로드
# image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device="cuda")

# # 텍스트 쿼리 입력
# text = clip.tokenize(["a photo of a cat"]).to(device="cuda")

# # 이미지와 텍스트를 임베딩
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

# # 이미지와 텍스트 간의 유사도 계산
# similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

# print("Similarity:", similarity.item())

if __name__ == '__main__':
    # save_embedding_info("/home/dataset/coco2017/train2017")
    # save_embedding_info("./dataset/data_concept_embedding/asian_doll")
    image_list = sorted(glob.glob(os.path.join('/home/dataset/coco2017/train2017', "*.jpg")))
    # concept = 'asian_doll'
    # concepts = ['billy_dog', 'cat_statue', 'dangling_child', 'running_shoes', 'iverson_funko_pop']
    concepts = ['billy_dog']
    for concept in concepts:
        data = retrieval_top_k(concept, 5, './dataset/retrieval/', image_list)
        print(data)
    # image_list = sorted(glob.glob(os.path.join('/home/dataset/coco2017/train2017', "*.jpg")))
    # for idx in indices[0]:
    #     print(image_list[int(idx)])
    #     shutil.copy(image_list[int(idx)], './dataset/retrieval/' + concept + '/'+image_list[int(idx)].split('/')[-1])

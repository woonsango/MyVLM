#!/bin/bash

python3 ./example_configs/llava/concept_embedding_training_vqa/write_yaml.py

# 특정 디렉토리 설정
DIRECTORY="./dataset/data_concept_embedding/"

# 파일 이름들을 배열에 저장
concepts=($(ls "$DIRECTORY"))

# 배열의 파일 이름들을 하나씩 출력하는 for문
for concept in "${concepts[@]}"
do
    CUDA_VISIBLE_DEVICES=5,6,7 python3 ./concept_embedding_training/train.py --config_path example_configs/llava/concept_embedding_training_vqa/concept/"${concept}"_concept_embedding_training_vqa.yaml
done

# 안되는 애들
# CUDA_VISIBLE_DEVICES=3,4,5 python3 ./concept_embedding_training/train.py --config_path example_configs/llava/concept_embedding_training_vqa/concept/mam_concept_embedding_training_vqa.yaml
# CUDA_VISIBLE_DEVICES=3,4,5 python3 ./concept_embedding_training/train.py --config_path example_configs/llava/concept_embedding_training_vqa/concept/cat_statue_concept_embedding_training_vqa.yaml
# CUDA_VISIBLE_DEVICES=3,4,5 python3 ./concept_embedding_training/train.py --config_path example_configs/llava/concept_embedding_training_vqa/concept/chicken_bean_bag_concept_embedding_training_vqa.yaml